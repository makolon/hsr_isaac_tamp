#!/usr/bin/env/python3
import sys
import rospy
import numpy as np
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

from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint


class ExecutePlan(object):
    def __init__(self):
        # Initialize ROS node
        rospy.init_node('execute_tamp_plan')

        # Feedback rate
        self.control_freq = 3
        self.rate = rospy.Rate(self.control_freq)

        # Core module
        self.tamp_planner = TAMPPlanner()
        self.hsr_interface = HSRInterface()
        self.tf_manager = TfManager()
        self.path_modifier = PlanModifier()
        self.mocap_interface = MocapInterface()
        self.ft_interface = ForceSensorInterface()

        # Publisher
        self.arm_pub = rospy.Publisher('/hsrb/arm_trajectory_controller/command', JointTrajectory, queue_size=10)
        self.base_pub = rospy.Publisher('/hsrb/omni_base_controller/command', JointTrajectory, queue_size=10)

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
        self.robot_joints = ['joint_x', 'joint_y', 'joint_rz', 'arm_lift_joint', 'arm_flex_joint',
                             'arm_roll_joint', 'wrist_flex_joint', 'wrist_roll_joint']
        robot_poses = self.hsr_interface.get_joint_positions()

        # Initialize tamp simulator
        observations = (robot_poses, object_poses)
        self.tamp_planner.initialize(observations)

    def set_base_pose(self, base_pose):
        base_traj = JointTrajectory()
        base_traj.joint_names = ['odom_x', 'odom_y', 'odom_t']

        # Set base trajectory
        assert len(base_pose) == 3, "Does not match the size of base pose"
        base_p = JointTrajectoryPoint()
        base_p.positions = base_pose
        base_p.velocities = np.zeros(len(base_pose))
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
        arm_p.velocities = np.zeros(len(arm_pose))
        arm_p.time_from_start = rospy.Duration(1)
        arm_traj.points = [arm_p]

        return arm_traj

    def check_status(self):
        # Wait for publisher has built
        while self.base_pub.get_num_connections() == 0:
            rospy.sleep(0.1)
        while self.arm_pub.get_num_connections() == 0:
            rospy.sleep(0.1)

    def plan(self):
        # Run TAMP
        plan, _, _ = self.tamp_planner.plan()

        return plan

    def process(self, action_name, args):
        # Modify plan
        action_name, modified_action = self.path_modifier.post_process(action_name, args)

        return action_name, modified_action

    def calculate_base_command(self, target_pose):
        curr_pose = self.hsr_interface.get_joint_positions(group='base')
        curr_vel = self.hsr_interface.get_joint_velocities(group='base')
        diff_pose = np.array(target_pose) - np.array(curr_pose)
        diff_vel = np.array(curr_vel)
        kp = 1.0
        D = 0.03
        kd = 2 * np.sqrt(kp) * D
        command = kp * diff_pose + kd * diff_vel
        command += np.array(curr_pose)
        return command

    def calculate_arm_command(self, target_pose):
        curr_pose = self.hsr_interface.get_joint_positions(group='arm')
        curr_vel = self.hsr_interface.get_joint_velocities(group='arm')
        diff_pose = np.array(target_pose) - np.array(curr_pose)
        diff_vel = np.array(curr_vel)
        kp = 1.0
        D = 0.03
        kd = 2 * np.sqrt(kp) * D
        command = kp * diff_pose + kd * diff_vel
        command += np.array(curr_pose)
        return command

    def execute(self):
        plan = self.plan()

        if plan is None:
            return None

        for i, (action_name, args) in enumerate(plan):
            # Post process TAMP commands to hsr executable actions
            action_name, modified_action = self.process(action_name, args)

            if action_name == 'move_base':
                for target_pose in modified_action:
                    target_base_pose = self.calculate_base_command(target_pose[:3])
                    base_traj = self.set_base_pose(target_base_pose)
                    self.base_pub.publish(base_traj)
                    self.rate.sleep()

            elif action_name == 'pick':
                pick_traj, return_traj = modified_action
                for target_pose in pick_traj: # pick
                    target_base_pose = self.calculate_base_command(target_pose[:3])
                    target_arm_pose = self.calculate_arm_command(target_pose[3:])
                    base_traj = self.set_base_pose(target_base_pose)
                    arm_traj = self.set_arm_pose(target_arm_pose)
                    self.base_pub.publish(base_traj)
                    self.arm_pub.publish(arm_traj)
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

            elif action_name == 'place':
                place_traj = modified_action
                for target_pose in place_traj: # place
                    target_base_pose = self.calculate_base_command(target_pose[:3])
                    target_arm_pose = self.calculate_arm_command(target_pose[3:])
                    base_traj = self.set_base_pose(target_base_pose)
                    arm_traj = self.set_arm_pose(target_arm_pose)
                    self.base_pub.publish(base_traj)
                    self.arm_pub.publish(arm_traj)
                    self.rate.sleep()

            elif action_name == 'insert':
                insert_traj, depart_traj, return_traj = modified_action
                for target_pose in insert_traj: # insert
                    target_base_pose = self.calculate_base_command(target_pose[:3])
                    target_arm_pose = self.calculate_arm_command(target_pose[3:])
                    base_traj = self.set_base_pose(target_base_pose)
                    arm_traj = self.set_arm_pose(target_arm_pose)
                    self.base_pub.publish(base_traj)
                    self.arm_pub.publish(arm_traj)
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

            else:
                continue


if __name__ == '__main__':
    exec_plan = ExecutePlan()
    exec_plan.execute()
