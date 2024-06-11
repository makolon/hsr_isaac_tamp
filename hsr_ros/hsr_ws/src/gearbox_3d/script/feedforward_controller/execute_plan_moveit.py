#!/usr/bin/env/python3
import os
import sys
import yaml
import rospy
import torch
import moveit_commander
import numpy as np
from scipy.spatial.transform import Rotation as R
from geometry_msgs.msg import Pose

# TODO: fix
sys.path.append('/root/tamp-hsr/hsr_tamp/experiments/gearbox_3d/')
sys.path.append('..')
from tamp_planner import TAMPPlanner
from hsr_interface import HSRInterface
from force_sensor_interface import ForceSensorInterface
from path_interface import PlanModifier
from mocap_interface import MocapInterface
from tf_interface import TfManager


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
    def __init__(self):
        # Initialize ROS node
        rospy.init_node('execute_tamp_plan')

        # Feedback rate
        self.control_freq = 10.0
        self.rate = rospy.Rate(self.control_freq)

        # Core module
        self.tamp_planner = TAMPPlanner()
        self.hsr_interface = HSRInterface()
        self.tf_manager = TfManager()
        self.path_modifier = PlanModifier()
        self.mocap_interface = MocapInterface()
        self.ft_interface = ForceSensorInterface()

        # Initialize moveit
        moveit_commander.roscpp_initialize(sys.argv)

        self.robot_moveit = moveit_commander.RobotCommander()
        self.arm_moveit = moveit_commander.MoveGroupCommander('arm', wait_for_servers=0.0)
        self.base_moveit = moveit_commander.MoveGroupCommander('base', wait_for_servers=0.0)
        self.gripper_moveit = moveit_commander.MoveGroupCommander('gripper', wait_for_servers=0.0)
        self.whole_body_moveit = moveit_commander.MoveGroupCommander('whole_body', wait_for_servers=0.0)

        # Planning parameters
        self.whole_body_moveit.allow_replanning(True)
        self.whole_body_moveit.set_planning_time(3)
        self.whole_body_moveit.set_pose_reference_frame('map')

        # Scene parameters
        self.scene = moveit_commander.PlanningSceneInterface()
        self.scene.remove_world_object()

        # Initialize Robot
        self.initialize_robot()

        # Initialize TAMP
        self.initialize_tamp()

    def initialize_robot(self):
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
        self.robot_joints = ['joint_x', 'joint_y', 'joint_rz', 'arm_lift_joint', 'arm_flex_joint',
                             'arm_roll_joint', 'wrist_flex_joint', 'wrist_roll_joint']
        robot_poses = self.hsr_interface.get_joint_positions()

        # Initialize tamp simulator
        observations = (robot_poses, object_poses)
        self.tamp_planner.initialize(observations)

    def plan(self):
        # Run TAMP
        plan, _, _ = self.tamp_planner.plan()

        return plan

    def augment_plan(self, plan):
        # Replay_trajectory
        return self.tamp_planner.execute(plan)

    def check_pick_status(self, target_ee_pose):
        target_ee_pos, target_ee_rot = target_ee_pose
        curr_ee_pos, curr_ee_rot = self.mocap_interface.get_pose('end_effector')

        target_ee_pos, target_ee_rot = torch.tensor(target_ee_pos), torch.tensor(target_ee_rot)
        curr_ee_pos, curr_ee_rot = torch.tensor(curr_ee_pos), torch.tensor(curr_ee_rot)

        # Calculate norm distance
        pos_dist = norm_diff_pos(target_ee_pos, curr_ee_pos)
        print('pick distance:', pos_dist)

        pick_success = torch.where(
            pos_dist < torch.tensor([0.03]),
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
            pos_dist < torch.tensor([0.03]),
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
            pos_dist < torch.tensor([0.01]),
            torch.ones((1,)),
            torch.zeros((1,))
        )
        return insert_success
    
    def set_goal_pose(self, target_pose):
        goal_pose = Pose()
        goal_pose.position.x = target_pose[0][0]
        goal_pose.position.y = target_pose[0][1]
        goal_pose.position.z = target_pose[0][2]
        goal_pose.orientation.x = target_pose[1][0]
        goal_pose.orientation.y = target_pose[1][1]
        goal_pose.orientation.z = target_pose[1][2]
        goal_pose.orientation.w = target_pose[1][3]

        return [goal_pose]

    def augment_plan(self, plan):
        # Replay_trajectory
        return self.tamp_planner.execute(plan)

    def execute(self):
        plan = self.plan()

        if plan is None:
            return None

        # Augment plan
        move_metadata, pick_metadata, place_metadata, insert_metadata = self.augment_plan(plan)

        # For metadata
        move_cnt, pick_cnt, place_cnt, insert_cnt = 0, 0, 0, 0

        for i, (action_name, args) in enumerate(plan):
            if action_name == 'move_base':
                for i in range(1):
                    move_pose = move_metadata['target_robot_pose'][move_cnt]

                    # Move
                    goal_pose = self.set_goal_pose(move_pose)
                    (traj, fraction) = self.whole_body_moveit.compute_cartesian_path(goal_pose, 0.01, 0.0, False)
                    self.whole_body_moveit.execute(traj, wait=True)   

                move_cnt += 1   

            elif action_name == 'pick':
                for i in range(3):
                    pick_pose = pick_metadata['target_robot_pose'][pick_cnt]

                    # Pick
                    goal_pose = self.set_goal_pose(pick_pose)
                    (traj, fraction) = self.whole_body_moveit.compute_cartesian_path(goal_pose, 0.01, 0.0, False)
                    self.whole_body_moveit.execute(traj, wait=True)

                    if i == 2:
                        rospy.sleep(3.0)
                        self.hsr_interface.close_gripper()

                    pick_cnt += 1

            elif action_name == 'place':
                for i in range(2):
                    place_pose = place_metadata['target_robot_pose'][place_cnt]

                    # Place
                    goal_pose = self.set_goal_pose(place_pose)
                    (traj, fraction) = self.whole_body_moveit.compute_cartesian_path(goal_pose, 0.01, 0.0, False)
                    self.whole_body_moveit.execute(traj, wait=True)

                place_cnt += 1

            elif action_name == 'insert':
                for i in range(3):
                    insert_pose = insert_metadata['target_robot_pose'][insert_cnt]

                    # Insert
                    goal_pose = self.set_goal_pose(insert_pose)
                    (traj, fraction) = self.whole_body_moveit.compute_cartesian_path(goal_pose, 0.01, 0.0, False)
                    self.whole_body_moveit.execute(traj, wait=True)

                    if i == 2:
                        rospy.sleep(3.0)
                        self.hsr_interface.open_gripper()

                    insert_cnt += 1

            else:
                continue


if __name__ == '__main__':
    exec_plan = ExecutePlan()
    exec_plan.execute()