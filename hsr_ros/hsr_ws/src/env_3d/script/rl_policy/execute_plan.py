# Copyright (c) 2018-2022, NVIDIA Corporation
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
#    list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
#    this list of conditions and the following disclaimer in the documentation
#    and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its
#    contributors may be used to endorse or promote products derived from
#    this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

#!/usr/bin/env/python3
import os
import sys
import yaml
import rospy
import torch
import numpy as np
from typing import Union
from scipy.spatial.transform import Rotation as R

# TODO: fix
sys.path.append('/root/tamp-hsr/hsr_tamp/experiments/env_3d/')
sys.path.append('..')
from tamp_planner import TAMPPlanner
from hsr_interface import HSRInterface
from force_sensor_interface import ForceSensorInterface
from post_process import PlanModifier
from mocap_interface import MocapInterface
from tf_interface import TfManager
from rl_agent import ResidualRL

sys.path.append('/root/tamp-hsr/')
from hsr_rl.utils.hydra_cfg.reformat import omegaconf_to_dict
from hsr_rl.tasks.utils.pinoc_utils import HSRIKSolver
from hsr_rl.tasks.utils.ik_utils import DifferentialInverseKinematicsCfg, DifferentialInverseKinematics

from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint


def load_config(policy_name: str = 'pick'):
    file_name = os.path.join('.', 'config', policy_name + '_config.yaml')
    with open(file_name, 'r') as f:
        config = yaml.load(f, Loader=yaml.SafeLoader)

    return config

def norm_diff_pos(p1: torch.Tensor, p2: torch.Tensor) -> torch.Tensor:
    # Calculate norm
    diff_norm = torch.norm(p1 - p2, p=2, dim=-1)

    return diff_norm

def norm_diff_xy(p1: torch.Tensor, p2: torch.Tensor) -> torch.Tensor:
    # Calculate norm
    diff_norm = torch.norm(p1[:2] - p2[:2], p=2, dim=-1)

    return diff_norm

def norm_diff_rot(q1: torch.Tensor, q2: torch.Tensor) -> torch.Tensor:
    # Calculate norm
    diff_norm1 = torch.norm(q1 - q2, p=2, dim=-1)
    diff_norm2 = torch.norm(q2 - q1, p=2, dim=-1)

    diff_norm = torch.min(diff_norm1, diff_norm2)

    return diff_norm

def normalize(x, eps: float = 1e-9):
    return x / x.norm(p=2, dim=-1).clamp(min=eps, max=None).unsqueeze(-1)

def quaternion_multiply(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    aw, ax, ay, az = torch.unbind(a, -1)
    bw, bx, by, bz = torch.unbind(b, -1)
    ow = aw * bw - ax * bx - ay * by - az * bz
    ox = aw * bx + ax * bw + ay * bz - az * by
    oy = aw * by - ax * bz + ay * bw + az * bx
    oz = aw * bz + ax * by - ay * bx + az * bw
    return torch.stack((ow, ox, oy, oz), -1)

def calc_diff_pos(p1, p2):
    return p1 - p2

def calc_diff_rot(q1, q2):
    q1 = normalize(q1)
    q2 = normalize(q2)

    scaling = torch.tensor([1, -1, -1, -1], device=q1.device)
    q1_inv = q1 * scaling
    q_diff = quaternion_multiply(q2, q1_inv)

    return q_diff


class ExecutePlan(object):
    def __init__(self, standalone=False):
        # Initialize ROS node
        rospy.init_node('execute_tamp_plan')

        # Feedback rate
        self.control_freq = 3.0
        self.rate = rospy.Rate(self.control_freq)

        self._device = 'cuda'
        self._command_type = 'position_rel'
        self._pd_control = False
        self._dt = torch.tensor(1.0, device=self._device)
        self._clip_obs = torch.tensor(5.0, device=self._device)

        # Core module
        self.tamp_planner = TAMPPlanner()
        self.hsr_interface = HSRInterface()
        self.tf_manager = TfManager()
        self.path_modifier = PlanModifier()
        self.mocap_interface = MocapInterface()
        self.ft_interface = ForceSensorInterface()
        self.hsr_ik_utils = HSRIKSolver()

        # Publisher
        self.arm_pub = rospy.Publisher('/hsrb/arm_trajectory_controller/command', JointTrajectory, queue_size=10)
        self.base_pub = rospy.Publisher('/hsrb/omni_base_controller/command', JointTrajectory, queue_size=10)

        pick_yaml = load_config('pick')
        place_yaml = load_config('place')
        insert_yaml = load_config('insert')

        pick_policy_cfg = omegaconf_to_dict(pick_yaml)
        place_policy_cfg = omegaconf_to_dict(place_yaml)
        insert_policy_cfg = omegaconf_to_dict(insert_yaml)

        # Skill based residual policy agents
        self.pick_agent = ResidualRL(pick_policy_cfg.get("params"))
        self.place_agent = ResidualRL(place_policy_cfg.get("params"))
        self.insert_agent = ResidualRL(insert_policy_cfg.get("params"))

        # Restore learned params
        self.pick_agent.restore(pick_policy_cfg["params"]["load_path"])
        self.place_agent.restore(place_policy_cfg["params"]["load_path"])
        self.insert_agent.restore(insert_policy_cfg["params"]["load_path"])

        # Get action scales for each skills
        self.pick_action_scale = torch.tensor(pick_policy_cfg["params"]["config"]["action_scale"], device=self._device)
        self.place_action_scale = torch.tensor(place_policy_cfg["params"]["config"]["action_scale"], device=self._device)
        self.insert_action_scale = torch.tensor(insert_policy_cfg["params"]["config"]["action_scale"], device=self._device)

        # Set ik controller
        self.ik_controller = self.set_ik_controller()

        # Initialize Robot
        self.initialize_robot()

        # Initialize TAMP
        self.initialize_tamp()

    def initialize_robot(self):
        self.check_status()

        # Set gripper to configuration position
        self.hsr_interface.open_gripper()

        # Set arm to configuration position
        self.hsr_interface.initialize_arm()

        # Set base to configuration position
        self.hsr_interface.initialize_base()

    def initialize_tamp(self):
        # Get object poses
        object_poses = self.mocap_interface.get_poses()

        # Get robot poses
        robot_poses = self.hsr_interface.get_joint_positions()

        # Initialize tamp simulator
        observations = (robot_poses, object_poses)
        self.tamp_planner.initialize(observations)

    #########################
    ####    ROS utils    ####
    #########################

    def set_base_pose(self, base_pose):
        base_traj = JointTrajectory()
        base_traj.joint_names = ['odom_x', 'odom_y', 'odom_t']

        # Set base trajectory
        assert len(base_pose) == 3, "Does not match the size of base pose"
        base_p = JointTrajectoryPoint()
        base_p.positions = base_pose
        base_p.velocities = np.zeros(len(base_pose)) # TODO: modify velocity
        base_p.time_from_start = rospy.Duration(1)
        base_traj.points = [base_p]

        return base_traj

    def set_arm_pose(self, arm_pose):
        arm_traj = JointTrajectory()
        arm_traj.joint_names = ['arm_lift_joint', 'arm_flex_joint',
                                'arm_roll_joint', 'wrist_flex_joint', 'wrist_roll_joint']

        # Set arm trajectory
        assert len(arm_pose) == 5, "Does not match the size of base pose"
        arm_p = JointTrajectoryPoint()
        arm_p.positions = arm_pose
        arm_p.velocities = np.zeros(len(arm_pose)) # TODO: modify velocity
        arm_p.time_from_start = rospy.Duration(1)
        arm_traj.points = [arm_p]

        return arm_traj

    def check_status(self):
        # Wait for publisher has built
        while self.base_pub.get_num_connections() == 0:
            rospy.sleep(0.1)
        while self.arm_pub.get_num_connections() == 0:
            rospy.sleep(0.1)

    ##########################
    ####    TAMP utils    ####
    ##########################

    def plan(self):
        # Run TAMP
        plan, _, _ = self.tamp_planner.plan()

        return plan

    def process(self, action_name, args):
        # Get object names
        object_names = self.tamp_planner.tamp_problem.body_names

        # Modify plan
        action_name, object_name, modified_action = self.path_modifier.post_process(action_name, object_names, args)

        return action_name, object_name, modified_action

    #########################
    ####     RL utils    ####
    #########################

    def get_pick_observation(self, obj_name, pick_pose, target_obj_pose) -> torch.Tensor:
        # Get joint pose
        joint_pose = self.hsr_interface.get_joint_positions(group='arm')

        # Get end effector and target object pose
        ee_pos, ee_rot = self.hsr_interface.get_link_pose('hand_palm_link')
        obj_pos, obj_rot = self.mocap_interface.get_pose(obj_name)

        # Calculate target end effector and object pose
        target_ee_pos, target_ee_rot = pick_pose[0], pick_pose[1]
        target_obj_pos, target_obj_rot = target_obj_pose[0], target_obj_pose[1]

        # To Tensor
        joint_pose = torch.tensor(joint_pose)
        ee_pos, ee_rot = torch.tensor(ee_pos), torch.tensor(ee_rot) # TODO: -torch.tensor(ee_rot)
        obj_pos, obj_rot = torch.tensor(obj_pos), torch.tensor(obj_rot)
        target_ee_pos, target_ee_rot = torch.tensor(target_ee_pos), torch.tensor(target_ee_rot)
        target_obj_pos, target_obj_rot = torch.tensor(target_obj_pos), torch.tensor(target_obj_rot)

        diff_ee_pos = calc_diff_pos(ee_pos, target_ee_pos) # difference ee_pos and pick_pos
        diff_ee_rot = calc_diff_rot(ee_rot, target_ee_rot) # difference ee_rot and pick_rot
        diff_obj_pos = calc_diff_pos(obj_pos, target_obj_pos) # difference obj_pos and target_pos
        diff_obj_rot = calc_diff_rot(obj_rot, target_obj_rot) # difference obj_rot and target_rot

        obs = torch.cat((joint_pose, diff_ee_pos, diff_ee_rot, diff_obj_pos, diff_obj_rot))
        obs = self._process_data(obs)

        return obs

    def get_place_observation(self, obj_name, target_obj_pose) -> torch.Tensor:
        # Get joint pose
        joint_pose = self.hsr_interface.get_joint_positions(group='arm')

        # Get end effector and target object pose
        ee_pos, ee_rot = self.hsr_interface.get_link_pose('hand_palm_link')
        obj_pos, obj_rot = self.mocap_interface.get_pose(obj_name)

        # Calculate target pose from base pose
        target_obj_pos, target_obj_rot = target_obj_pose[0], target_obj_pose[1]

        # To Tensor
        joint_pose = torch.tensor(joint_pose)
        ee_pos, ee_rot = torch.tensor(ee_pos), torch.tensor(ee_rot) # TODO: -torch.tensor(ee_rot)
        obj_pos, obj_rot = torch.tensor(obj_pos), torch.tensor(obj_rot)
        target_obj_pos, target_obj_rot = torch.tensor(target_obj_pos), torch.tensor(target_obj_rot)

        diff_ee_pos = calc_diff_pos(ee_pos, obj_pos) # difference ee_pos and obj_pos
        diff_ee_rot = calc_diff_rot(ee_rot, obj_rot) # difference ee_rot and obj_rot
        diff_obj_pos = calc_diff_pos(obj_pos, target_obj_pos) # difference obj_pos and target_pos
        diff_obj_rot = calc_diff_rot(obj_rot, target_obj_rot) # difference obj_rot and target_rot

        obs = torch.cat((joint_pose, diff_ee_pos, diff_ee_rot, diff_obj_pos, diff_obj_rot))
        obs = self._process_data(obs)

        return obs

    def get_insert_observation(self, obj_name, target_obj_pose) -> torch.Tensor:
        # Get joint pose
        joint_pose = self.hsr_interface.get_joint_positions(group='arm')

        # Get end effector and target object pose
        ee_pos, ee_rot = self.hsr_interface.get_link_pose('hand_palm_link')
        obj_pos, obj_rot = self.mocap_interface.get_pose(obj_name)

        # Calculate target pose from base pose
        target_obj_pos, target_obj_rot = target_obj_pose[0], target_obj_pose[1]

        # To Tensor
        joint_pose = torch.tensor(joint_pose)
        ee_pos, ee_rot = torch.tensor(ee_pos), torch.tensor(ee_rot) # TODO: -torch.tensor(ee_rot)
        obj_pos, obj_rot = torch.tensor(obj_pos), torch.tensor(obj_rot)
        target_obj_pos, target_obj_rot = torch.tensor(target_obj_pos), torch.tensor(target_obj_rot)

        diff_ee_pos = calc_diff_pos(ee_pos, obj_pos) # difference ee_pos and obj_pos
        diff_ee_rot = calc_diff_rot(ee_rot, obj_rot) # difference ee_rot and obj_rot
        diff_obj_pos = calc_diff_pos(obj_pos, target_obj_pos) # difference obj_pos and target_pos
        diff_obj_rot = calc_diff_rot(obj_rot, target_obj_rot) # difference obj_rot and target_rot

        obs = torch.cat((joint_pose, diff_ee_pos, diff_ee_rot, diff_obj_pos, diff_obj_rot))
        obs = self._process_data(obs)

        return obs

    def _process_data(self, obs: torch.Tensor) -> torch.Tensor:
        # To device
        obs = obs.to(self._device)

        # To torch.float32
        obs = obs.to(torch.float32)

        # Clamp observation
        obs = torch.clamp(obs, -self._clip_obs, self._clip_obs).to(self._device).clone()

        return obs

    def check_pick_status(self, obj_name):
        obj_pos, obj_rot = self.mocap_interface.get_pose(obj_name)
        ee_pos, ee_rot = self.hsr_interface.get_link_pose('hand_palm_link')

        obj_pos, obj_rot = torch.tensor(obj_pos), torch.tensor(obj_rot)
        ee_pos, ee_rot = torch.tensor(ee_pos), torch.tensor(ee_rot)

        # Calculate norm distance
        pos_dist = norm_diff_pos(ee_pos, obj_pos)
        print('pick distance:', pos_dist)

        pick_success = torch.where(
            pos_dist < torch.tensor([0.1]),
            torch.ones((1,)),
            torch.zeros((1,))
        )
        return True

    def check_place_status(self, obj_name, target_obj_pose):
        obj_pos, obj_rot = self.mocap_interface.get_pose(obj_name)
        target_pos, target_rot = target_obj_pose[0], target_obj_pose[1]

        obj_pos, obj_rot = torch.tensor(obj_pos), torch.tensor(obj_rot)
        target_pos, target_rot = torch.tensor(target_pos), torch.tensor(target_rot)

        # Calculate norm distance
        pos_dist = norm_diff_pos(obj_pos, target_pos)
        print('place distance:', pos_dist)

        place_success = torch.where(
            pos_dist < torch.tensor([0.03]),
            torch.ones((1,)),
            torch.zeros((1,))
        )
        return True

    def check_insert_status(self, obj_name, target_obj_pose):
        obj_pos, obj_rot = self.mocap_interface.get_pose(obj_name)
        target_pos, target_rot = target_obj_pose[0], target_obj_pose[1]

        obj_pos, obj_rot = torch.tensor(obj_pos), torch.tensor(obj_rot)
        target_pos, target_rot = torch.tensor(target_pos), torch.tensor(target_rot)

        # Calculate norm distance
        pos_dist = norm_diff_pos(obj_pos, target_pos)
        print('insert distance:', pos_dist)

        insert_success = torch.where(
            pos_dist < torch.tensor([0.02]),
            torch.ones((1,)),
            torch.zeros((1,))
        )
        return insert_success

    ##########################
    #### Real Robot utils ####
    ##########################

    def set_ik_controller(self):
        ik_control_cfg = DifferentialInverseKinematicsCfg(
            command_type=self._command_type,
            ik_method="dls",
            position_offset=(0.0, 0.0, 0.0),
            rotation_offset=(1.0, 0.0, 0.0, 0.0),
        )
        return DifferentialInverseKinematics(ik_control_cfg, 1, self._device)

    def calculate_base_command(self, target_pose):
        curr_pose = self.hsr_interface.get_joint_positions(group='base')
        curr_vel = self.hsr_interface.get_joint_velocities(group='base')
        diff_pose = np.array(target_pose) - np.array(curr_pose)
        diff_vel = np.array(curr_vel)
        kp = 1.25
        kd = 0.1
        command = kp * diff_pose + kd * diff_vel
        command += np.array(curr_pose)
        return command

    def calculate_arm_command(self, target_pose):
        curr_pose = self.hsr_interface.get_joint_positions(group='arm')
        curr_vel = self.hsr_interface.get_joint_velocities(group='arm')
        diff_pose = np.array(target_pose) - np.array(curr_pose)
        diff_vel = np.array(curr_vel)
        kp = 1.25
        kd = 0.1
        command = kp * diff_pose + kd * diff_vel
        command += np.array(curr_pose)
        return command

    def execute(self):
        plan = self.plan()

        if plan is None:
            return None

        for i, (action_name, args) in enumerate(plan):
            # Post process TAMP commands to hsr executable actions
            action_name, object_name, modified_action = self.process(action_name, args)

            if action_name == 'move_base':
                for target_pose in modified_action:
                    if self._pd_control:
                        target_base_pose = self.calculate_base_command(target_pose[:3])
                    else:
                        target_base_pose = target_pose[:3]
                    base_traj = self.set_base_pose(target_base_pose)
                    self.base_pub.publish(base_traj)
                    self.rate.sleep()

            elif action_name == 'pick':
                finish = False
                pick_traj, return_traj = modified_action

                for target_pose in pick_traj:
                    if self._pd_control:
                        target_base_pose = self.calculate_base_command(target_pose[:3])
                        target_arm_pose = self.calculate_arm_command(target_pose[3:])
                    else:
                        target_base_pose = target_pose[:3]
                        target_arm_pose = target_pose[3:]

                    # Get observation
                    obs = self.get_pick_observation(object_name, pick_pose, target_obj_pose)

                    # Residual action
                    with torch.no_grad():
                        actions = self.pick_agent.get_action(obs)

                    # Multiply target 6D pose and residual 6D pose
                    ik_action = self._dt * self.pick_action_scale * actions.to(self._device) # (dx, dy, dz)
                    self.ik_controller.set_command(ik_action)

                    # Calculate robot jacobian
                    ee_pos, ee_rot = self.hsr_interface.get_link_pose('hand_palm_link')
                    joint_positions = np.array(self.hsr_interface.get_joint_positions())
                    robot_jacobian = self.hsr_ik_utils.get_jacobian(joint_positions)

                    # To tensor and device
                    ee_pos = torch.tensor(ee_pos, dtype=torch.float32, device=self._device).view(1, 3)
                    ee_rot = -torch.tensor(ee_rot, dtype=torch.float32, device=self._device).view(1, 4)
                    robot_jacobian = torch.tensor(robot_jacobian, dtype=torch.float32, device=self._device).view(1, -1, 8)

                    # Calcurate delta pose
                    delta_pose = self.ik_controller.compute_delta(ee_pos, ee_rot, robot_jacobian)
                    delta_pose = torch.squeeze(delta_pose, dim=0)
                    delta_pose = delta_pose.to('cpu').detach().numpy().copy() # 8 dim

                    # Add delta pose to reference trajectory
                    target_base_pose += delta_pose[:3]
                    target_arm_pose += delta_pose[3:]

                    # Set target pose
                    base_traj = self.set_base_pose(target_base_pose)
                    arm_traj = self.set_arm_pose(target_arm_pose)

                    # Publish command
                    self.base_pub.publish(base_traj)
                    self.arm_pub.publish(arm_traj)
                    self.rate.sleep()

                rospy.sleep(2.0)
                self.hsr_interface.close_gripper()

                for target_pose in return_traj: # return
                    if self._pd_control:
                        target_base_pose = self.calculate_base_command(target_pose[:3])
                        target_arm_pose = self.calculate_arm_command(target_pose[3:])
                    else:
                        target_base_pose = target_pose[:3]
                        target_arm_pose = target_pose[3:]
                    base_traj = self.set_base_pose(target_base_pose)
                    arm_traj = self.set_arm_pose(target_arm_pose)
                    self.base_pub.publish(base_traj)
                    self.arm_pub.publish(arm_traj)
                    self.rate.sleep()

            elif action_name == 'place':
                finish = False
                place_traj = modified_action

                for target_pose in place_traj:
                    if self._pd_control:
                        target_base_pose = self.calculate_base_command(target_pose[:3])
                        target_arm_pose = self.calculate_arm_command(target_pose[3:])
                    else:
                        target_base_pose = target_pose[:3]
                        target_arm_pose = target_pose[3:]

                    # Get observation
                    obs = self.get_place_observation(object_name, target_obj_pose)

                    # Residual action
                    with torch.no_grad():
                        actions = self.place_agent.get_action(obs)

                    # Multiply target 6D pose and residual 6D pose
                    ik_action = self._dt * self.place_action_scale * actions.to(self._device) # (dx, dy, dz)
                    self.ik_controller.set_command(ik_action)

                    # Calculate robot jacobian
                    ee_pos, ee_rot = self.hsr_interface.get_link_pose('hand_palm_link')
                    joint_positions = np.array(self.hsr_interface.get_joint_positions())
                    robot_jacobian = self.hsr_ik_utils.get_jacobian(joint_positions)

                    # To tensor and device
                    ee_pos = torch.tensor(ee_pos, dtype=torch.float32, device=self._device).view(1, 3)
                    ee_rot = -torch.tensor(ee_rot, dtype=torch.float32, device=self._device).view(1, 4)
                    robot_jacobian = torch.tensor(robot_jacobian, dtype=torch.float32, device=self._device).view(1, -1, 8)

                    # Calcurate delta pose
                    delta_pose = self.ik_controller.compute_delta(ee_pos, ee_rot, robot_jacobian)
                    delta_pose = torch.squeeze(delta_pose, dim=0)
                    delta_pose = delta_pose.to('cpu').detach().numpy().copy() # 8 dim

                    # Add delta pose to reference trajectory
                    target_base_pose += delta_pose[:3]
                    target_arm_pose += delta_pose[3:]

                    # Set target pose
                    base_traj = self.set_base_pose(target_base_pose)
                    arm_traj = self.set_arm_pose(target_arm_pose)

                    # Publish command
                    self.base_pub.publish(base_traj)
                    self.arm_pub.publish(arm_traj)
                    self.rate.sleep()

            elif action_name == 'insert':
                finish = False
                loop_count = 0
                insert_traj, depart_traj, return_traj = modified_action

                while not finish: # insert
                    target_pose = insert_traj[loop_count]
                    if self._pd_control:
                        target_base_pose = self.calculate_base_command(target_pose[:3])
                        target_arm_pose = self.calculate_arm_command(target_pose[3:])
                    else:
                        target_base_pose = target_pose[:3]
                        target_arm_pose = target_pose[3:]

                    # Get observation
                    obs = self.get_insert_observation(object_name, target_obj_pose)

                    # Residual action
                    with torch.no_grad():
                        actions = self.insert_agent.get_action(obs)

                    # Multiply target 6D pose and residual 6D pose
                    ik_action = self._dt * self.insert_action_scale * actions.to(self._device) # (dx, dy, dz)
                    self.ik_controller.set_command(ik_action)

                    # Calculate robot jacobian
                    ee_pos, ee_rot = self.hsr_interface.get_link_pose('hand_palm_link')
                    joint_positions = np.array(self.hsr_interface.get_joint_positions())
                    robot_jacobian = self.hsr_ik_utils.get_jacobian(joint_positions)

                    # To tensor and device
                    ee_pos = torch.tensor(ee_pos, dtype=torch.float32, device=self._device).view(1, 3)
                    ee_rot = -torch.tensor(ee_rot, dtype=torch.float32, device=self._device).view(1, 4)
                    robot_jacobian = torch.tensor(robot_jacobian, dtype=torch.float32, device=self._device).view(1, -1, 8)

                    # Calcurate delta pose
                    delta_pose = self.ik_controller.compute_delta(ee_pos, ee_rot, robot_jacobian)
                    delta_pose = torch.squeeze(delta_pose, dim=0)
                    delta_pose = delta_pose.to('cpu').detach().numpy().copy() # 8 dim

                    # Add delta pose to reference trajectory
                    target_base_pose += delta_pose[:3]
                    target_arm_pose += delta_pose[3:]

                    # Set target pose
                    base_traj = self.set_base_pose(target_base_pose)
                    arm_traj = self.set_arm_pose(target_arm_pose)

                    prev_ft_data = self.ft_interface.get_current_force()

                    # Publish command
                    self.base_pub.publish(base_traj)
                    self.arm_pub.publish(arm_traj)
                    self.rate.sleep()

                    loop_count += 1
                    if loop_count >= len(insert_traj):
                        loop_count = len(insert_traj)-1
                        current_ft_data = self.ft_interface.get_current_force()
                        force_difference = self.ft_interface.compute_difference(prev_ft_data, current_ft_data)
                        weight = round(force_difference / 9.81 * 1000, 1)

                        finish = bool(self.check_insert_status(object_name))
                        finish |= True if weight > 500 else False

                rospy.sleep(2.0)
                self.hsr_interface.open_gripper()

                for target_pose in depart_traj: # depart
                    if self._pd_control:
                        target_base_pose = self.calculate_base_command(target_pose[:3])
                        target_arm_pose = self.calculate_arm_command(target_pose[3:])
                    else:
                        target_base_pose = target_pose[:3]
                        target_arm_pose = target_pose[3:]
                    base_traj = self.set_base_pose(target_base_pose)
                    arm_traj = self.set_arm_pose(target_arm_pose)
                    self.base_pub.publish(base_traj)
                    self.arm_pub.publish(arm_traj)
                    self.rate.sleep()

                for target_pose in return_traj: # return
                    if self._pd_control:
                        target_base_pose = self.calculate_base_command(target_pose[:3])
                        target_arm_pose = self.calculate_arm_command(target_pose[3:])
                    else:
                        target_base_pose = target_pose[:3]
                        target_arm_pose = target_pose[3:]
                    base_traj = self.set_base_pose(target_base_pose)
                    arm_traj = self.set_arm_pose(target_arm_pose)
                    self.base_pub.publish(base_traj)
                    self.arm_pub.publish(arm_traj)
                    self.rate.sleep()

            else:
                continue


if __name__ == '__main__':
    exec_plan = ExecutePlan()
    exec_plan.execute()