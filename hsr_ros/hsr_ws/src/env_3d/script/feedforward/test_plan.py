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
from post_process import PlanModifier
from mocap_interface import MocapInterface
from tf_interface import TfManager

from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint


class ExecutePlan(object):
    def __init__(self):
        # Initialize ROS node
        rospy.init_node('execute_tamp_plan')

        self.rate = rospy.Rate(3)

        # Core module
        self.tamp_planner = TAMPPlanner()
        self.hsr_interface = HSRInterface()
        self.tf_manager = TfManager()
        self.path_modifier = PlanModifier()
        self.mocap_interface = MocapInterface()

        # Publisher
        self.arm_pub = rospy.Publisher('/hsrb/arm_trajectory_controller/command', JointTrajectory, queue_size=10)
        self.base_pub = rospy.Publisher('/hsrb/omni_base_controller/command', JointTrajectory, queue_size=10)

        # Initialize Robot
        self.initialize_robot()

        # Initialize TAMP
        self.initialize_tamp()

        self.reset_dataset()

    def reset_dataset(self):
        self.measured_ee_traj = []
        self.true_ee_traj = []
        self.measured_joint_traj = []

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

    def execute(self):
        plan = self.plan()

        if plan is None:
            return None
        
        def make_tuple(pose_msg):
            return ((pose_msg.pose.position.x, pose_msg.pose.position.y, pose_msg.pose.position.z),
                    (pose_msg.pose.orientation.x, pose_msg.pose.orientation.y, pose_msg.pose.orientation.z, pose_msg.pose.orientation.w))

        for num_trial in range(5):
            for i, (action_name, args) in enumerate(plan):
                # Post process TAMP commands to hsr executable actions
                action_name, modified_action = self.process(action_name, args)

                if action_name == 'move_base':
                    for target_pose in modified_action:
                        base_traj = self.set_base_pose(target_pose)
                        self.base_pub.publish(base_traj)

                        # Get measured/true EE traj
                        measured_ee_pose = self.hsr_interface.get_link_pose('hand_palm_link')
                        self.measured_ee_traj.append(measured_ee_pose)

                        true_ee_pose = self.mocap_interface.get_pose('end_effector')
                        self.true_ee_traj.append(make_tuple(true_ee_pose))

                        # Get measured joint traj
                        measured_joint_pos = self.hsr_interface.get_joint_positions()
                        self.measured_joint_traj.append(measured_joint_pos)

                        self.rate.sleep()

                elif action_name == 'pick':
                    pick_traj, return_traj = modified_action
                    for target_pose in pick_traj: # forward
                        base_traj = self.set_base_pose(target_pose[:3])
                        arm_traj = self.set_arm_pose(target_pose[3:])
                        self.base_pub.publish(base_traj)
                        self.arm_pub.publish(arm_traj)

                        # Get measured/true EE traj
                        measured_ee_pose = self.hsr_interface.get_link_pose('hand_palm_link')
                        self.measured_ee_traj.append(measured_ee_pose)

                        true_ee_pose = self.mocap_interface.get_pose('end_effector')
                        self.true_ee_traj.append(make_tuple(true_ee_pose))

                        # Get measured joint traj
                        measured_joint_pos = self.hsr_interface.get_joint_positions()
                        self.measured_joint_traj.append(measured_joint_pos)

                        self.rate.sleep()

                    # rospy.sleep(5.0)
                    # self.hsr_interface.close_gripper()

                    for target_pose in return_traj: # reverse
                        base_traj = self.set_base_pose(target_pose[:3])
                        arm_traj = self.set_arm_pose(target_pose[3:])
                        self.base_pub.publish(base_traj)
                        self.arm_pub.publish(arm_traj)

                        # Get measured/true EE traj
                        measured_ee_pose = self.hsr_interface.get_link_pose('hand_palm_link')
                        self.measured_ee_traj.append(measured_ee_pose)

                        true_ee_pose = self.mocap_interface.get_pose('end_effector')
                        self.true_ee_traj.append(make_tuple(true_ee_pose))

                        # Get measured joint traj
                        measured_joint_pos = self.hsr_interface.get_joint_positions()
                        self.measured_joint_traj.append(measured_joint_pos)

                        self.rate.sleep()

                elif action_name == 'place':
                    place_traj = modified_action
                    for target_pose in place_traj: # forward
                        base_traj = self.set_base_pose(target_pose[:3])
                        arm_traj = self.set_arm_pose(target_pose[3:])
                        self.base_pub.publish(base_traj)
                        self.arm_pub.publish(arm_traj)

                        # Get measured/true EE traj
                        measured_ee_pose = self.hsr_interface.get_link_pose('hand_palm_link')
                        self.measured_ee_traj.append(measured_ee_pose)

                        true_ee_pose = self.mocap_interface.get_pose('end_effector')
                        self.true_ee_traj.append(make_tuple(true_ee_pose))

                        # Get measured joint traj
                        measured_joint_pos = self.hsr_interface.get_joint_positions()
                        self.measured_joint_traj.append(measured_joint_pos)

                        self.rate.sleep()

                elif action_name == 'insert':
                    insert_traj, depart_traj, return_traj = modified_action
                    for target_pose in insert_traj: # insert
                        base_traj = self.set_base_pose(target_pose[:3])
                        arm_traj = self.set_arm_pose(target_pose[3:])
                        self.base_pub.publish(base_traj)
                        self.arm_pub.publish(arm_traj)

                        # Get measured/true EE traj
                        measured_ee_pose = self.hsr_interface.get_link_pose('hand_palm_link')
                        self.measured_ee_traj.append(measured_ee_pose)

                        true_ee_pose = self.mocap_interface.get_pose('end_effector')
                        self.true_ee_traj.append(make_tuple(true_ee_pose))

                        # Get measured joint traj
                        measured_joint_pos = self.hsr_interface.get_joint_positions()
                        self.measured_joint_traj.append(measured_joint_pos)

                        self.rate.sleep()

                    # rospy.sleep(5.0)
                    # self.hsr_interface.open_gripper()

                    for target_pose in depart_traj: # depart
                        base_traj = self.set_base_pose(target_pose[:3])
                        arm_traj = self.set_arm_pose(target_pose[3:])
                        self.base_pub.publish(base_traj)
                        self.arm_pub.publish(arm_traj)

                        # Get measured/true EE traj
                        measured_ee_pose = self.hsr_interface.get_link_pose('hand_palm_link')
                        self.measured_ee_traj.append(measured_ee_pose)

                        true_ee_pose = self.mocap_interface.get_pose('end_effector')
                        self.true_ee_traj.append(make_tuple(true_ee_pose))

                        # Get measured joint traj
                        measured_joint_pos = self.hsr_interface.get_joint_positions()
                        self.measured_joint_traj.append(measured_joint_pos)

                        self.rate.sleep()

                    for target_pose in return_traj: # return
                        base_traj = self.set_base_pose(target_pose[:3])
                        arm_traj = self.set_arm_pose(target_pose[3:])
                        self.base_pub.publish(base_traj)
                        self.arm_pub.publish(arm_traj)

                        # Get measured/true EE traj
                        measured_ee_pose = self.hsr_interface.get_link_pose('hand_palm_link')
                        self.measured_ee_traj.append(measured_ee_pose)

                        true_ee_pose = self.mocap_interface.get_pose('end_effector')
                        self.true_ee_traj.append(make_tuple(true_ee_pose))

                        # Get measured joint traj
                        measured_joint_pos = self.hsr_interface.get_joint_positions()
                        self.measured_joint_traj.append(measured_joint_pos)

                        self.rate.sleep()

                else:
                    continue
            
            self.save_traj(num_trial)
            self.reset_dataset()
            self.hsr_interface.initialize_base()
            self.hsr_interface.initialize_arm()

    def save_traj(self, num_trial):
        np.save(f'measured_ee_traj_{num_trial}', self.measured_ee_traj)
        np.save(f'measured_joint_traj_{num_trial}', self.measured_joint_traj)
        # np.save(f'true_ee_traj_{num_trial}', self.true_ee_traj)


if __name__ == '__main__':
    exec_plan = ExecutePlan()
    exec_plan.execute()
