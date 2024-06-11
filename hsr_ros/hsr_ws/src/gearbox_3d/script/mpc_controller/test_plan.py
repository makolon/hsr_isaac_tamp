#!/usr/bin/env/python3
import sys
import rospy
import numpy as np
from scipy.spatial.transform import Rotation as R

# TODO: fix
sys.path.append('/root/tamp-hsr/hsr_tamp/experiments/gearbox_3d/')
sys.path.append('..')
from base_controller.controller import BaseController

from geometry_msgs.msg import PoseArray
from ocs_msgs.msg import mpc_observation, mpc_target_trajectories, mpc_flattened_controller


class ExecutePlan(BaseController):
    def __init__(self):
        super(BaseController, self).__init__()

        # Publisher
        self.mpc_target_pub = rospy.Publisher('/mobile_manipulator_mpc_target', mpc_target_trajectories, queue_size=1)
        self.mpc_policy_pub = rospy.Publisher('/mobile_manipulator_mpc_policy', mpc_flattened_controller, queue_size=1)
        self.mpc_observation_pub = rospy.Publisher('/mobile_manipulator_mpc_observation', mpc_observation, queue_size=1)

        # Subscriber
        self.traj_sub = rospy.Subscriber('/mobile_manipulator/optimizedPoseTrajectory', PoseArray, self._optimized_plan_cb)

    def _optimized_plan_cb(self, msg: PoseArray):
        self.target_traj = []
        for pos, orn in msg.poses:
            self.target_traj.append([pos, orn])

    def reset_dataset(self):
        self.measured_ee_traj = []
        self.measured_joint_traj = []

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

                    # Get measured/true EE traj
                    measured_ee_pose = self.hsr_interface.get_link_pose('hand_palm_link')
                    self.measured_ee_traj.append(measured_ee_pose)

                    # Get measured joint traj
                    measured_joint_pos = self.hsr_interface.get_joint_positions()
                    self.measured_joint_traj.append(measured_joint_pos)

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

                    # Get measured/true EE traj
                    measured_ee_pose = self.hsr_interface.get_link_pose('hand_palm_link')
                    self.measured_ee_traj.append(measured_ee_pose)

                    # Get measured joint traj
                    measured_joint_pos = self.hsr_interface.get_joint_positions()
                    self.measured_joint_traj.append(measured_joint_pos)

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

                    # Get measured/true EE traj
                    measured_ee_pose = self.hsr_interface.get_link_pose('hand_palm_link')
                    self.measured_ee_traj.append(measured_ee_pose)

                    # Get measured joint traj
                    measured_joint_pos = self.hsr_interface.get_joint_positions()
                    self.measured_joint_traj.append(measured_joint_pos)

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

                    # Get measured/true EE traj
                    measured_ee_pose = self.hsr_interface.get_link_pose('hand_palm_link')
                    self.measured_ee_traj.append(measured_ee_pose)

                    # Get measured joint traj
                    measured_joint_pos = self.hsr_interface.get_joint_positions()
                    self.measured_joint_traj.append(measured_joint_pos)

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

                    # Get measured/true EE traj
                    measured_ee_pose = self.hsr_interface.get_link_pose('hand_palm_link')
                    self.measured_ee_traj.append(measured_ee_pose)

                    # Get measured joint traj
                    measured_joint_pos = self.hsr_interface.get_joint_positions()
                    self.measured_joint_traj.append(measured_joint_pos)

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

                    # Get measured/true EE traj
                    measured_ee_pose = self.hsr_interface.get_link_pose('hand_palm_link')
                    self.measured_ee_traj.append(measured_ee_pose)

                    # Get measured joint traj
                    measured_joint_pos = self.hsr_interface.get_joint_positions()
                    self.measured_joint_traj.append(measured_joint_pos)

                    self.rate.sleep()

                for target_pose in return_traj: # return
                    target_base_pose = self.calculate_base_command(target_pose[:3])
                    target_arm_pose = self.calculate_arm_command(target_pose[3:])
                    base_traj = self.set_base_pose(target_base_pose)
                    arm_traj = self.set_arm_pose(target_arm_pose)
                    self.base_pub.publish(base_traj)
                    self.arm_pub.publish(arm_traj)

                    # Get measured/true EE traj
                    measured_ee_pose = self.hsr_interface.get_link_pose('hand_palm_link')
                    self.measured_ee_traj.append(measured_ee_pose)

                    # Get measured joint traj
                    measured_joint_pos = self.hsr_interface.get_joint_positions()
                    self.measured_joint_traj.append(measured_joint_pos)

                    self.rate.sleep()

            else:
                continue

        self.save_traj()

    def save_traj(self):
        np.save(f'measured_ee_traj', self.measured_ee_traj)
        np.save(f'measured_joint_traj', self.measured_joint_traj)


if __name__ == '__main__':
    exec_plan = ExecutePlan()
    exec_plan.execute()