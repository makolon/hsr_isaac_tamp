#!/usr/bin/env python3
import sys
import time
import rospy
import numpy as np
import moveit_commander
from geometry_msgs.msg import Pose
from trajectory_msgs.msg import JointTrajectory
from controller_manager_msgs.srv import ListControllers

from hsr_py_interface import HSRPyInterface

class ExecutePlan(object):
    def __init__(self):
        # Initialize moveit
        moveit_commander.roscpp_initialize(sys.argv)

        # Initialize ROS node
        rospy.init_node('execute_tamp_plan')

        # HSR Python interface
        self.hsr_py = HSRPyInterface()

        # Publisher
        self.pub_arm = rospy.Publisher('/hsrb/arm_trajectory_controller/command', JointTrajectory, queue_size=10)
        self.pub_base = rospy.Publisher('/hsrb/omni_base_controller/command', JointTrajectory, queue_size=10) 

        # Wait for publisher has built
        while self.pub_arm.get_num_connections() == 0:
            rospy.sleep(0.1)
        while self.pub_base.get_num_connections() == 0:
            rospy.sleep(0.1)

        # Initialize robot
        self.initialize_robot()

    def initialize_robot(self):
        self.check_status()

        # Set arm to configuration position
        self.hsr_py.initialize_arm()

        # Set base to configuration position
        self.hsr_py.initialize_base()

        self.hsr_py.whole_body.move_to_neutral()

        self.whole_body = moveit_commander.MoveGroupCommander('whole_body', wait_for_servers=0.0)

    def check_status(self):
        # Make sure the controller is running
        rospy.wait_for_service('/hsrb/controller_manager/list_controllers')
        list_controllers = rospy.ServiceProxy('/hsrb/controller_manager/list_controllers', ListControllers)

        running = False
        while running is False:
            rospy.sleep(0.1)
            for c in list_controllers().controller:
                if c.name == 'arm_trajectory_controller' and c.state == 'running':
                    running |= True
                if c.name == 'head_trajectory_controller' and c.state == 'running':
                    running |= True
                if c.name == 'gripper_controller' and c.state == 'running':
                    running |= True
                if c.name == 'omni_base_controller' and c.state == 'running':
                    running |= True

        return running

    def create_trajectory(self):
        trajectory_x = np.ones(100) * 0.1
        trajectory_y = np.ones(100) * 0.1
        trajectory_z = np.ones(100) * 0.1
        self.trajectory = np.array([[trajectory_x[i], trajectory_y[i], trajectory_z[i]] 
                                    for i in range(len(trajectory_x))])

    def set_target_pose(self, count):
        ee_pose = self.whole_body.get_current_pose()

        goal_pose = Pose()
        goal_pose.position.x = ee_pose.pose.position.x + 0.0
        goal_pose.position.y = ee_pose.pose.position.y + self.trajectory[count][1]
        goal_pose.position.z = ee_pose.pose.position.z + 0.0
        goal_pose.orientation.x = ee_pose.pose.orientation.x
        goal_pose.orientation.y = ee_pose.pose.orientation.y
        goal_pose.orientation.z = ee_pose.pose.orientation.z
        goal_pose.orientation.w = ee_pose.pose.orientation.w

        return goal_pose

    def execute(self):
        traj_count = 0
        self.create_trajectory()
        input('wait_for_user')
        while not rospy.is_shutdown():
            start = time.time()
            goal_pose = self.set_target_pose(traj_count)
            (plan, fraction) = self.whole_body.compute_cartesian_path([goal_pose], 0.01, 0.0, True)
            self.whole_body.execute(plan, wait=False)

            traj_count += 1
            if traj_count >= len(self.trajectory):
                break

            print('Loop Hz:', 1 / (time.time() - start))


if __name__ == '__main__':
    exec_plan = ExecutePlan()
    exec_plan.execute()