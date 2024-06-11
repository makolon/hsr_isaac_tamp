#!/usr/bin/env/python3
import os
import sys
import yaml
import rospy
import torch
import numpy as np
from scipy.spatial.transform import Rotation as R

# TODO: fix
sys.path.append('/root/tamp-hsr/hsr_tamp/experiments/gearbox_3d/')
sys.path.append('..')
from base_controller.controller import BaseController
from rl_agent import ResidualRL

sys.path.append('/root/tamp-hsr/')
from hsr_rl.utils.hydra_cfg.reformat import omegaconf_to_dict
from hsr_rl.tasks.utils.pinoc_utils import HSRIKSolver
from hsr_rl.tasks.utils.ik_utils import DifferentialInverseKinematicsCfg, DifferentialInverseKinematics


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

def modify_target_pose(target_obj_pose, gearbox_base_pose, object_name):
    # Calculate position difference from motion capture
    #   difference from left_hole: [ 0.16399157 -0.09748133 -0.03273092]
    #   difference from right_hole: [ 0.16733813  0.09592897 -0.04950504]
    #   difference from middle_shaft: [ 0.2633127  -0.00121711 -0.11001092]
    gearbox_base_pos, _ = gearbox_base_pose
    left_offset = (-0.164, 0.0975, 0.0)
    right_offset = (-0.167, -0.0960, 0.0)
    middle_offset = (-0.263, 0.0, 0.0)
    if object_name == 'gear1':
        return ((gearbox_base_pos[0]+left_offset[0],
                 gearbox_base_pos[1]+left_offset[1],
                 target_obj_pose[0][2]),
                 target_obj_pose[1])
    elif object_name == 'gear2':
        return ((gearbox_base_pos[0]+right_offset[0],
                 gearbox_base_pos[1]+right_offset[1],
                 target_obj_pose[0][2]),
                 target_obj_pose[1])
    elif object_name == 'gear3':
        return ((gearbox_base_pos[0]+middle_offset[0],
                 gearbox_base_pos[1]+middle_offset[1],
                 target_obj_pose[0][2]),
                 target_obj_pose[1])
    elif object_name == 'shaft1':
        return ((gearbox_base_pos[0]+left_offset[0],
                 gearbox_base_pos[1]+left_offset[1],
                 target_obj_pose[0][2]),
                 target_obj_pose[1])
    elif object_name == 'shaft2':
        return ((gearbox_base_pos[0]+right_offset[0],
                 gearbox_base_pos[1]+right_offset[1],
                 target_obj_pose[0][2]),
                 target_obj_pose[1])


class ExecutePlan(BaseController):
    def __init__(self, standalone=False):
        super(ExecutePlan, self).__init__()

        self._device = 'cuda'
        self._command_type = 'position_rel'
        self._dt = torch.tensor(1.0/self.control_freq, device=self._device)
        self._gamma = torch.tensor(0.3, device=self._device)
        self._clip_obs = torch.tensor(5.0, device=self._device)

        self.move_base_kp = 4.0
        self.pick_base_kp = 4.0
        self.place_base_kp = 1.0
        self.insert_base_kp = 1.0

        self.move_arm_kp = 2.0
        self.pick_arm_kp = 2.0
        self.place_arm_kp = 1.0
        self.insert_arm_kp = 1.0

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

    def augment_plan(self, plan):
        # Replay_trajectory
        return self.tamp_planner.execute(plan)

    def get_pick_observation(self, obj_name, pick_pose, target_obj_pose) -> torch.Tensor:
        # Get joint pose
        joint_pose = self.hsr_interface.get_joint_positions(group='arm')

        # Get end effector and target object pose
        ee_pos, ee_rot = self.mocap_interface.get_pose('end_effector')
        obj_pos, obj_rot = self.mocap_interface.get_pose(obj_name)

        # Calculate target end effector and object pose
        target_ee_pos, target_ee_rot = pick_pose[0], pick_pose[1]
        target_obj_pos, target_obj_rot = target_obj_pose[0], target_obj_pose[1]

        # To Tensor
        joint_pose = torch.tensor(joint_pose)
        ee_pos, ee_rot = torch.tensor(ee_pos), torch.tensor(ee_rot)
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

    def get_place_observation(self, obj_name, place_pose, target_obj_pose) -> torch.Tensor:
        # Get joint pose
        joint_pose = self.hsr_interface.get_joint_positions(group='arm')

        # Get end effector and target object pose
        ee_pos, ee_rot = self.mocap_interface.get_pose('end_effector')
        obj_pos, obj_rot = self.mocap_interface.get_pose(obj_name)

        # Calculate target pose from base pose
        target_obj_pos, target_obj_rot = target_obj_pose[0], target_obj_pose[1]

        # To Tensor
        joint_pose = torch.tensor(joint_pose)
        ee_pos, ee_rot = torch.tensor(ee_pos), torch.tensor(ee_rot)
        obj_pos, obj_rot = torch.tensor(obj_pos), torch.tensor(obj_rot)
        target_obj_pos, target_obj_rot = torch.tensor(target_obj_pos), torch.tensor(target_obj_rot)

        diff_ee_pos = calc_diff_pos(ee_pos, obj_pos) # difference ee_pos and obj_pos
        diff_ee_rot = calc_diff_rot(ee_rot, obj_rot) # difference ee_rot and obj_rot
        diff_obj_pos = calc_diff_pos(obj_pos, target_obj_pos) # difference obj_pos and target_pos
        diff_obj_rot = calc_diff_rot(obj_rot, target_obj_rot) # difference obj_rot and target_rot

        obs = torch.cat((joint_pose, diff_ee_pos, diff_ee_rot, diff_obj_pos, diff_obj_rot))
        obs = self._process_data(obs)

        return obs

    def get_insert_observation(self, obj_name, insert_pose, target_obj_pose) -> torch.Tensor:
        # Get joint pose
        joint_pose = self.hsr_interface.get_joint_positions(group='arm')

        # Get end effector and target object pose
        ee_pos, ee_rot = self.mocap_interface.get_pose('end_effector')
        obj_pos, obj_rot = self.mocap_interface.get_pose(obj_name)

        # Calculate target pose from base pose
        target_obj_pos, target_obj_rot = target_obj_pose[0], target_obj_pose[1]

        # To Tensor
        joint_pose = torch.tensor(joint_pose)
        ee_pos, ee_rot = torch.tensor(ee_pos), torch.tensor(ee_rot)
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

    def check_move_status(self, target_ee_pose):
        target_ee_pos, target_ee_rot = target_ee_pose
        curr_ee_pos, curr_ee_rot = self.mocap_interface.get_pose('end_effector')

        target_ee_pos, target_ee_rot = torch.tensor(target_ee_pos), torch.tensor(target_ee_rot)
        curr_ee_pos, curr_ee_rot = torch.tensor(curr_ee_pos), torch.tensor(curr_ee_rot)

        # Calculate norm distance
        pos_ee_dist = norm_diff_xy(target_ee_pos, curr_ee_pos)
        print('move ee distance:', pos_ee_dist)

        move_success = torch.where(
            pos_ee_dist < torch.tensor([0.015]),
            torch.ones((1,)),
            torch.zeros((1,))
        )

        return move_success

    def check_pick_status(self, target_ee_pose):
        target_ee_pos, target_ee_rot = target_ee_pose
        curr_ee_pos, curr_ee_rot = self.mocap_interface.get_pose('end_effector')

        target_ee_pos, target_ee_rot = torch.tensor(target_ee_pos), torch.tensor(target_ee_rot)
        curr_ee_pos, curr_ee_rot = torch.tensor(curr_ee_pos), torch.tensor(curr_ee_rot)

        # Calculate norm distance
        pos_ee_dist = norm_diff_xy(target_ee_pos, curr_ee_pos)
        print('pick ee distance:', pos_ee_dist)

        pick_success = torch.where(
            pos_ee_dist < torch.tensor([0.015]),
            torch.ones((1,)),
            torch.zeros((1,))
        )
        return pick_success

    def check_place_status(self, obj_name, target_obj_pose):
        obj_pos, obj_rot = self.mocap_interface.get_pose(obj_name)
        target_pos, target_rot = target_obj_pose[0], target_obj_pose[1]

        obj_pos, obj_rot = torch.tensor(obj_pos), torch.tensor(obj_rot)
        target_pos, target_rot = torch.tensor(target_pos), torch.tensor(target_rot)

        # Calculate norm distance
        pos_dist = norm_diff_xy(obj_pos, target_pos)
        print('place distance:', pos_dist)

        place_success = torch.where(
            pos_dist < torch.tensor([0.015]),
            torch.ones((1,)),
            torch.zeros((1,))
        )
        return place_success

    def check_insert_status(self, obj_name, target_obj_pose):
        obj_pos, obj_rot = self.mocap_interface.get_pose(obj_name)
        target_pos, target_rot = target_obj_pose[0], target_obj_pose[1]

        obj_pos, obj_rot = torch.tensor(obj_pos), torch.tensor(obj_rot)
        target_pos, target_rot = torch.tensor(target_pos), torch.tensor(target_rot)

        # Calculate norm distance
        pos_dist = norm_diff_xy(obj_pos, target_pos)
        print('insert distance:', pos_dist)

        insert_success = torch.where(
            pos_dist < torch.tensor([0.003]),
            torch.ones((1,)),
            torch.zeros((1,))
        )
        return insert_success

    def execute(self):
        plan = self.plan()

        if plan is None:
            return None

        # Augment plan
        move_metadata, pick_metadata, place_metadata, insert_metadata = self.augment_plan(plan)

        # For metadata
        move_cnt, pick_cnt, place_cnt, insert_cnt = 0, 0, 0, 0

        for i, (action_name, args) in enumerate(plan):
            # Post process TAMP commands to hsr executable actions
            action_name, object_name, modified_action = self.process(action_name, args)

            if action_name == 'move_base':
                finish = False
                loop_count = 0
                move_traj = modified_action

                # Get target poses from pick_metadata
                move_pose = move_metadata['target_robot_pose'][move_cnt]

                while not finish: # move
                    target_pose = move_traj[loop_count]

                    target_base_pose = self.calculate_base_command(target_pose[:3], self.move_base_kp)
                    base_traj = self.set_base_pose(target_base_pose)
                    self.base_pub.publish(base_traj)

                    loop_count += 1
                    if loop_count >= len(move_traj):
                        loop_count = len(move_traj)-1
                        finish = bool(self.check_move_status(move_pose))
                        if finish:
                            break

                    self.rate.sleep()

                move_cnt += 1

            elif action_name == 'pick':
                finish = False
                loop_count = 0
                pick_traj, return_traj = modified_action

                # Get target poses from pick_metadata
                pick_pose = pick_metadata['target_robot_pose'][pick_cnt]
                target_obj_pose = pick_metadata['target_object_pose'][pick_cnt]

                while not finish: # pick
                    target_pose = pick_traj[loop_count]

                    # Get observation
                    obs = self.get_pick_observation(object_name, pick_pose, target_obj_pose)

                    # Residual action
                    with torch.no_grad():
                        actions = self.pick_agent.get_action(obs)

                    # Multiply target 6D pose and residual 6D pose
                    jt_action = self._dt * self.pick_action_scale * actions.to(self._device)

                    delta_pose = torch.squeeze(jt_action, dim=0)
                    delta_pose = delta_pose.to('cpu').detach().numpy().copy() # 8 dim

                    # P control / PD control
                    target_base_pose = self.calculate_base_command(target_pose[:3]+delta_pose[:3], kp=self.pick_base_kp)
                    target_arm_pose = self.calculate_arm_command(target_pose[3:]+delta_pose[3:], kp=self.pick_arm_kp)

                    # Set target pose
                    base_traj = self.set_base_pose(target_base_pose)
                    arm_traj = self.set_arm_pose(target_arm_pose)

                    # Publish command
                    self.base_pub.publish(base_traj)
                    self.arm_pub.publish(arm_traj)

                    loop_count += 1
                    if loop_count >= len(pick_traj):
                        loop_count = len(pick_traj)-1
                        finish = bool(self.check_pick_status(pick_pose))
                        if finish:
                            break

                    self.rate.sleep()

                rospy.sleep(3.0)
                self.hsr_interface.close_gripper()

                for target_pose in return_traj: # return
                    target_base_pose = self.calculate_base_command(target_pose[:3])
                    target_arm_pose = self.calculate_arm_command(target_pose[3:])
                    base_traj = self.set_base_pose(target_base_pose)
                    arm_traj = self.set_arm_pose(target_arm_pose)
                    self.base_pub.publish(base_traj)
                    self.arm_pub.publish(arm_traj)
                    self.rate.sleep()

                # Add pick count
                pick_cnt += 1

            elif action_name == 'place':
                finish = False
                loop_count = 0
                place_traj = modified_action

                # Get target poses from place_metadata
                place_pose = place_metadata['target_robot_pose'][place_cnt]
                target_obj_pose = place_metadata['target_object_pose'][place_cnt]

                gearbox_base_pose = self.mocap_interface.get_pose('base')
                target_obj_pose = modify_target_pose(target_obj_pose, gearbox_base_pose, object_name)

                while not finish: # place
                    target_pose = place_traj[loop_count]

                    # Get observation
                    obs = self.get_place_observation(object_name, place_pose, target_obj_pose)

                    # Residual action
                    with torch.no_grad():
                        actions = self.place_agent.get_action(obs)

                    # Multiply target 6D pose and residual 6D pose
                    jt_action = self._dt * self.place_action_scale * actions.to(self._device)

                    delta_pose = torch.squeeze(jt_action, dim=0)
                    delta_pose = delta_pose.to('cpu').detach().numpy().copy() # 8 dim

                    print('place action:', delta_pose)

                    # P control / PD control
                    target_base_pose = self.calculate_base_command(target_pose[:3]+delta_pose[:3], kp=self.place_base_kp)
                    target_arm_pose = self.calculate_arm_command(target_pose[3:]+delta_pose[3:], kp=self.place_arm_kp)

                    # Set target pose
                    base_traj = self.set_base_pose(target_base_pose)
                    arm_traj = self.set_arm_pose(target_arm_pose)

                    # Publish command
                    self.base_pub.publish(base_traj)
                    self.arm_pub.publish(arm_traj)

                    loop_count += 1
                    if loop_count >= len(place_traj):
                        loop_count = len(place_traj)-1
                        finish = bool(self.check_place_status(object_name, target_obj_pose))
                        if finish:
                            break

                    self.rate.sleep()

                # Add place count
                place_cnt += 1

            elif action_name == 'insert':
                finish = False
                loop_count = 0
                insert_traj, depart_traj, return_traj = modified_action

                # Get target poses from insert_metadata
                insert_pose = insert_metadata['target_robot_pose'][insert_cnt]
                target_obj_pose = insert_metadata['target_object_pose'][insert_cnt]

                gearbox_base_pose = self.mocap_interface.get_pose('base')
                target_obj_pose = modify_target_pose(target_obj_pose, gearbox_base_pose, object_name)

                while not finish: # insert
                    target_pose = insert_traj[loop_count]

                    # Get observation
                    obs = self.get_insert_observation(object_name, insert_pose, target_obj_pose)

                    # Residual action
                    with torch.no_grad():
                        actions = self.insert_agent.get_action(obs)

                    # Multiply target 6D pose and residual 6D pose
                    jt_action = self._dt * self.insert_action_scale * actions.to(self._device)

                    delta_pose = torch.squeeze(jt_action, dim=0)
                    delta_pose = delta_pose.to('cpu').detach().numpy().copy() # 8 dim

                    print('insert action:', delta_pose)

                    # P control / PD control
                    target_base_pose = self.calculate_base_command(target_pose[:3]+delta_pose[:3], kp=self.insert_base_kp)
                    target_arm_pose = self.calculate_arm_command(target_pose[3:]+delta_pose[3:], kp=self.insert_arm_kp)

                    # Set target pose
                    base_traj = self.set_base_pose(target_base_pose)
                    arm_traj = self.set_arm_pose(target_arm_pose)

                    # Publish command
                    self.base_pub.publish(base_traj)
                    self.arm_pub.publish(arm_traj)

                    loop_count += 1
                    if loop_count >= len(insert_traj):
                        loop_count = len(insert_traj)-1
                        finish = bool(self.check_insert_status(object_name, target_obj_pose))
                        if finish:
                            break

                    self.rate.sleep()

                rospy.sleep(3.0)
                self.hsr_interface.open_gripper()

                for target_pose in depart_traj: # depart
                    target_base_pose = self.calculate_base_command(target_pose[:3])
                    target_arm_pose = self.calculate_arm_command(target_pose[3:])
                    base_traj = self.set_base_pose(target_base_pose)
                    arm_traj = self.set_arm_pose(target_arm_pose)
                    self.base_pub.publish(base_traj)
                    self.arm_pub.publish(arm_traj)
                    self.rate.sleep()

                for target_pose in return_traj: # return
                    target_base_pose = self.calculate_base_command(target_pose[:3])
                    target_arm_pose = self.calculate_arm_command(target_pose[3:])
                    base_traj = self.set_base_pose(target_base_pose)
                    arm_traj = self.set_arm_pose(target_arm_pose)
                    self.base_pub.publish(base_traj)
                    self.arm_pub.publish(arm_traj)
                    self.rate.sleep()

                # Add insert count
                insert_cnt += 1

            else:
                continue



if __name__ == '__main__':
    exec_plan = ExecutePlan()
    exec_plan.execute()