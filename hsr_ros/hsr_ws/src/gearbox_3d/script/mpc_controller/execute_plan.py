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
        super(ExecutePlan, self).__init__()

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

    def execute(self):
        plan = self.plan()

        if plan is None:
            return None

        for i, (action_name, args) in enumerate(plan):
            # Post process TAMP commands to hsr executable actions
            action_name, object_name, modified_action = self.process(action_name, args)

            if action_name == 'move_base':
                for target_pose in modified_action:
                    base_traj = self.set_base_pose(target_pose[:3])
                    self.base_pub.publish(base_traj)
                    self.rate.sleep()

            elif action_name == 'pick':
                pick_traj, return_traj = modified_action
                for target_pose in pick_traj: # pick
                    base_traj = self.set_base_pose(target_pose[:3])
                    arm_traj = self.set_arm_pose(target_pose[3:])
                    self.base_pub.publish(base_traj)
                    self.arm_pub.publish(arm_traj)
                    self.rate.sleep()

                rospy.sleep(2.0)
                self.hsr_interface.close_gripper()

                for target_pose in return_traj: # return
                    base_traj = self.set_base_pose(target_pose[:3])
                    arm_traj = self.set_arm_pose(target_pose[3:])
                    self.base_pub.publish(base_traj)
                    self.arm_pub.publish(arm_traj)
                    self.rate.sleep()

            elif action_name == 'place':
                place_traj = modified_action
                for target_pose in place_traj: # place
                    base_traj = self.set_base_pose(target_pose[:3])
                    arm_traj = self.set_arm_pose(target_pose[3:])
                    self.base_pub.publish(base_traj)
                    self.arm_pub.publish(arm_traj)
                    self.rate.sleep()

            elif action_name == 'insert':
                insert_traj, depart_traj, return_traj = modified_action
                for target_pose in insert_traj: # insert
                    base_traj = self.set_base_pose(target_pose[:3])
                    arm_traj = self.set_arm_pose(target_pose[3:])
                    self.base_pub.publish(base_traj)
                    self.arm_pub.publish(arm_traj)
                    self.rate.sleep()

                rospy.sleep(2.0)
                self.hsr_interface.open_gripper()

                for target_pose in depart_traj: # depart
                    base_traj = self.set_base_pose(target_pose[:3])
                    arm_traj = self.set_arm_pose(target_pose[3:])
                    self.base_pub.publish(base_traj)
                    self.arm_pub.publish(arm_traj)
                    self.rate.sleep()

                for target_pose in return_traj: # return
                    base_traj = self.set_base_pose(target_pose[:3])
                    arm_traj = self.set_arm_pose(target_pose[3:])
                    self.base_pub.publish(base_traj)
                    self.arm_pub.publish(arm_traj)
                    self.rate.sleep()

            else:
                continue


if __name__ == '__main__':
    exec_plan = ExecutePlan()
    exec_plan.execute()
