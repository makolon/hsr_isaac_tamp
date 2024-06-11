#!/usr/bin/env/python3
import sys
import rospy
import numpy as np
from scipy.spatial.transform import Rotation as R

# TODO: fix
sys.path.append('/root/tamp-hsr/hsr_tamp/experiments/gearbox_3d/')
sys.path.append('..')
from base_controller.controller import BaseController


class ExecutePlan(BaseController):
    def __init__(self):
        super(BaseController, self).__init__()

        self.reset_dataset()

    def reset_dataset(self):
        self.measured_ee_traj = []
        self.true_ee_traj = []
        self.measured_joint_traj = []

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

                    rospy.sleep(3.0)
                    self.hsr_interface.close_gripper()

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

                    rospy.sleep(3.0)
                    self.hsr_interface.open_gripper()

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
        np.save(f'true_ee_traj_{num_trial}', self.true_ee_traj)


if __name__ == '__main__':
    exec_plan = ExecutePlan()
    exec_plan.execute()
