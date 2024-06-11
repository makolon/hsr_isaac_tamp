#!/usr/bin/env/python3
import sys
import time
import rospy
import hsrb_interface
import numpy as np
from controller_manager_msgs.srv import ListControllers


class ExecutePlan(object):
    def __init__(self):
        # Initialize ROS node
        rospy.init_node('execute_tamp_plan')

        # Initialize Robot
        self.initialize_robot()

    def initialize_robot(self):
        self.check_status()

        self.robot = hsrb_interface.Robot()
        self.omni_base = self.robot.get('omni_base')
        self.gripper = self.robot.get('gripper')
        self.whole_body = self.robot.get('whole_body')

        # Initialize arm position
        self.whole_body.move_to_go()

        # Initialize base position
        self.omni_base.go_abs(0.0, 0.0, 0.0, 300.0)

        # Set arm to configuration pose
        self.whole_body.move_to_neutral()

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
    
    def create_trajectory(self, num_steps=10):
        traj_x = np.linspace(0.0, 0.0, num_steps)
        traj_y = np.linspace(0.0, 0.0, num_steps)
        traj_z = np.linspace(0.0, 0.2, num_steps)
        trajectory = np.dstack((traj_x, traj_y, traj_z))[0]

        return trajectory

    def execute(self):
        trajectory = self.create_trajectory(num_steps=50)
        self.whole_body.linear_weight = 100.0
        self.whole_body.angular_weight = 100.0

        input('wait_for_user')
        while not rospy.is_shutdown():
            start = time.time()
            self.whole_body.move_end_effector_pose([
                hsrb_interface.geometry.pose(x=0.0,
                                             y=0.0,
                                             z=0.3)
            ], ref_frame_id='hand_palm_link', wait=False)

            print('Loop Hz: ', 1 / (time.time() - start))


if __name__ == '__main__':
    exec_plan = ExecutePlan()
    exec_plan.execute()
