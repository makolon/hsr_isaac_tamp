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
        super(ExecutePlan, self).__init__()

    def execute(self):
        plan = self.plan()

        if plan is None:
            return None

        for i, (action_name, args) in enumerate(plan):
            # Post process TAMP commands to hsr executable actions
            action_name, object_name, modified_action = self.process(action_name, args)

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
