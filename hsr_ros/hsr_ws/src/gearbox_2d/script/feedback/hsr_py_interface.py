#!/usr/bin/env/python3
import sys
import rospy
import hsrb_interface
import numpy as np
from controller_manager_msgs.srv import ListControllers


class HSRPyInterface(object):
    def __init__(self, standalone=False):
        # Initialize ROS node
        if standalone:
            rospy.init_node('hsr_python_interface')

        # Check server status
        self.check_status()

        self.robot = hsrb_interface.Robot()
        self.omni_base = self.robot.get('omni_base')
        self.gripper = self.robot.get('gripper')
        self.whole_body = self.robot.get('whole_body')

    def initialize_arm(self):
        # Initialize arm position
        self.whole_body.move_to_go()

    def initialize_base(self):
        # Initialize base position
        self.omni_base.go_abs(0.0, 0.0, 0.0, 300.0)

    def set_task_pose(self, grasp_type='side'):
        # Set base to configuration position
        self.omni_base.go_abs(-0.80, 0.75, np.pi/2, 300.0)

        # Set arm to configuration pose
        self.whole_body.move_to_neutral()

        if grasp_type == 'side':
            self.whole_body.move_end_effector_pose([
                hsrb_interface.geometry.pose(x=-0.3, y=0.0, z=0.0, ej=0.0),
            ], ref_frame_id='hand_palm_link')
        elif grasp_type == 'top':
            self.whole_body.move_end_effector_pose([
                hsrb_interface.geometry.pose(x=-0.3, y=0.0, z=0.0, ej=-1.57),
            ], ref_frame_id='hand_palm_link')

        # Open gripper
        self.gripper.command(1.2)

    def fix_task_pose(self, diff_pose):
        diff_x_pose, diff_y_pose, diff_z_pose = diff_pose[0], diff_pose[1], diff_pose[2]
        if np.abs(diff_z_pose) < 0.1:
            diff_z_pose = -0.12
        else:
            diff_z_pose = 0.0
        self.whole_body.move_end_effector_pose([
            hsrb_interface.geometry.pose(x=0.0, y=diff_y_pose, z=diff_z_pose)
        ], ref_frame_id='hand_palm_link')

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

    def close_gripper(self, force=0.5):
        self.gripper.apply_force(force)

    def open_gripper(self, width=1.0):
        self.gripper.command(width)

    def insert_object(self, ee_mode):
        if ee_mode == 'horizontal':
            self.whole_body.move_end_effector_pose([
                hsrb_interface.geometry.pose(x=-0.03, y=0.0, z=0.0),
            ], ref_frame_id='hand_palm_link')
        elif ee_mode == 'vertical':
            self.whole_body.move_end_effector_pose([
                hsrb_interface.geometry.pose(x=0.0, y=-0.03, z=0.0),
            ], ref_frame_id='hand_palm_link')

    def grasp_gear(self, diff_ee_pose, pick_offset=0.1):
        # Open gripper
        self.open_gripper(0.5)

        # Move to grasp
        self.whole_body.move_end_effector_by_line((0, 0, 1), -pick_offset)
        self.whole_body.move_end_effector_pose([
            hsrb_interface.geometry.pose(
                x=diff_ee_pose[0],
                y=diff_ee_pose[1],
                ek=-np.pi/2
            ),
        ], ref_frame_id='hand_palm_link')
        self.whole_body.move_end_effector_by_line((0, 0, 1), diff_ee_pose[2]+pick_offset)

        # Close gripper
        self.close_gripper(0.5)


if __name__ == '__main__':
    hsr_py = HSRPyInterface()
    hsr_py.initialize_arm()
    hsr_py.initialize_base()
