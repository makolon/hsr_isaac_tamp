#!/usr/bin/env/python3
import tf
import sys
import rospy
import moveit_commander
import numpy as np
from scipy.spatial.transform import Rotation as R

sys.path.append('..')
sys.path.append('../..')
from tf_manager import TfManager
from hsr_py_interface import HSRPyInterface
from controller.ik_controller import IKController
from planner.linear_interpolation import LinearInterpolationPlanner

from controller_manager_msgs.srv import ListControllers


class ClosedLoopDemo(object):
    def __init__(self, grasp_type='side'):
        # Initialize moveit
        moveit_commander.roscpp_initialize(sys.argv)

        # Initialize ROS node
        rospy.init_node('execute_tamp_plan')

        # Feedback rate
        self.rate = rospy.Rate(50)

        # Core module
        self.hsr_py = HSRPyInterface()
        self.tf_manager = TfManager()
        self.planner = LinearInterpolationPlanner()
        self.controller = IKController()

        # Initialize Robot
        self.initialize_robot(grasp_type)

    def initialize_robot(self, grasp_type='side'):
        self.check_status()

        # Set arm to configuration position
        self.hsr_py.initialize_arm()

        # Set base to configuration position
        self.hsr_py.initialize_base()

        # Set arm to task pose
        self.hsr_py.set_task_pose()

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

    def set_target_pose(self, ee_traj, ee_mode='horizontal'):
        # Extract full trajectory
        diff_ee_traj = ee_traj[0]

        goal_way_points = []
        for i in range(1, len(diff_ee_traj)):
            # Transform difference pose on end effector frame to map frame
            target_map_pose = self.tf_manager.transform_coordinate(diff_ee_traj[i])

            if ee_mode == 'horizontal':
                goal_pose = ((target_map_pose[0],
                              target_map_pose[1],
                              target_map_pose[2]),
                             (0.5, 0.5, 0.5, -0.5),)
                goal_way_points.append(goal_pose)
            elif ee_mode == 'vertical':
                goal_pose = ((target_map_pose[0],
                              target_map_pose[1],
                              target_map_pose[2]),
                             (0.0, 0.707106781, 0.707106781, 0.0),)
                goal_way_points.append(goal_pose)

        return goal_way_points

    def execute(self):
        pose = np.array([0.0, 0.1, 0.0])

        # Plan trajectory
        ee_traj = self.planner.compute_path(pose, num_steps=100)
        traj = self.set_target_pose(ee_traj)

        # Execute
        while not rospy.is_shutdown():
            next_ee_pose = ((-0.8665181649519156, 1.060068345164372, 0.3912539360516797), (0.5, 0.5, 0.5, -0.5))
            self.controller.control(next_ee_pose)
            self.rate.sleep()


if __name__ == '__main__':
    closed_loop = ClosedLoopDemo()
    closed_loop.execute()