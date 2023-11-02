#!/usr/bin/env/python3
import sys
import rospy
import hsrb_interface
import numpy as np
from scipy.spatial.transform import Rotation as R

# TODO: fix
sys.path.append('/root/tamp-hsr/hsr_tamp/experiments/env_2d/')
sys.path.append('..')
from tamp_planner import TAMPPlanner
from post_process import PlanModifier
from mocap_interface import MocapInterface
from tf_interface import TfManager

from controller_manager_msgs.srv import ListControllers


class ExecutePlan(object):
    def __init__(self, grasp_type='side'):
        # Initialize ROS node
        rospy.init_node('execute_tamp_plan')

        # Core module
        self.tamp_planner = TAMPPlanner()
        self.path_modifier = PlanModifier()
        self.mocap_interface = MocapInterface()
        self.tf_manager = TfManager()

        # Initialize Robot
        self.initialize_robot(grasp_type)

    def initialize_robot(self, grasp_type='side'):
        self.check_status()

        # Get hsr python interface
        self.robot = hsrb_interface.Robot()
        self.omni_base = self.robot.get('omni_base')
        self.gripper = self.robot.get('gripper')
        self.whole_body = self.robot.get('whole_body')

        # Initialize arm position
        self.whole_body.move_to_go()

        # Initialize base position
        self.omni_base.go_abs(0.0, 0.0, 0.0, 300.0)

        # Set base to configuration position
        self.omni_base.go_abs(-0.8, 0.80, np.pi/2, 300.0)

        # Set arm to neutral
        self.whole_body.move_to_neutral()

        # Set hsr to configuration pose
        if grasp_type == 'side':
            self.whole_body.move_end_effector_pose([
                hsrb_interface.geometry.pose(x=-0.25, y=0.0, z=0.0, ej=0.0),
            ], ref_frame_id='hand_palm_link')

            # Fix base position using mocap data
            gearbox_poses = self.mocap_interface.get_poses()
            initial_pose = self.tf_manager.get_init_pose(gearbox_poses)
            diff_x_pose, diff_y_pose, diff_z_pose = initial_pose[1], initial_pose[0], initial_pose[2]
            self.whole_body.move_end_effector_pose([
                hsrb_interface.geometry.pose(x=0.0, y=-diff_y_pose, z=0.0)
            ], ref_frame_id='hand_palm_link')
        elif grasp_type == 'top':
            self.whole_body.move_end_effector_pose([
                hsrb_interface.geometry.pose(x=-0.25, y=0.0, z=0.0, ej=-np.pi/2),
            ], ref_frame_id='hand_palm_link')

            # Fix base position using mocap data
            gearbox_poses = self.mocap_interface.get_poses()
            initial_pose = self.tf_manager.get_init_pose(gearbox_poses)
            diff_x_pose, diff_y_pose, diff_z_pose = initial_pose[0], initial_pose[1], initial_pose[2]
            self.whole_body.move_end_effector_pose([
                hsrb_interface.geometry.pose(x=-diff_x_pose, y=-diff_y_pose, z=0.0)
            ], ref_frame_id='hand_palm_link')

        # Open gripper
        self.gripper.command(1.2)

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

    def plan(self):
        # Run TAMP
        plan, _, _ = self.tamp_planner.plan()

        return plan

    def process(self, action_name, args, grasp_type='side'):
        # Get rigid body poses from mocap
        mocap_poses = self.mocap_interface.get_poses()

        # Modify plan
        action_name, modified_action, ee_mode = self.path_modifier.post_process(action_name, args, mocap_poses, grasp_type)

        return action_name, modified_action, ee_mode

    def check_ee_mode(self):
        # Get end effector pose
        ee_pose = self.whole_body.get_end_effector_pose()

        ee_rot = np.array([
            ee_pose.ori.x,
            ee_pose.ori.y,
            ee_pose.ori.z,
            ee_pose.ori.w
        ])
        ee_euler = R.from_quat(ee_rot).as_euler('xyz')

        # Check whether end effector is vertical to ground plane, or horizontal
        if -0.8 < ee_euler[1] and 0.8 > ee_euler[1]:
            ee_mode = 'vertical'
        elif -2.0 < ee_euler[1] and -1.0 > ee_euler[1]:
            ee_mode = 'horizontal'

        return ee_mode

    def execute(self, grasp_type='side'):
        plan = self.plan()

        if plan is None:
            return None

        for i, (action_name, args) in enumerate(plan):
            # Post process TAMP commands to hsr executable actions
            action_name, diff_ee_pose, ee_mode = self.process(action_name, args, grasp_type)

            if action_name == 'move':
                # Set end effector mode
                ee_mode = self.check_ee_mode()

                # Move to next pose
                if ee_mode == 'horizontal':
                    self.whole_body.move_end_effector_pose([
                        hsrb_interface.geometry.pose(
                            x=diff_ee_pose[0],
                            y=diff_ee_pose[1],
                            z=diff_ee_pose[2]
                        ),
                    ], ref_frame_id='hand_palm_link')
                elif ee_mode == 'vertical':
                    self.whole_body.move_end_effector_pose([
                        hsrb_interface.geometry.pose(
                            x=-diff_ee_pose[1],
                            y=diff_ee_pose[0],
                            z=diff_ee_pose[2]
                        ),
                    ], ref_frame_id='hand_palm_link')

            elif action_name == 'pick':
                # Set impedance
                self.whole_body.impedance_config = 'compliance_hard'

                # Move to grasp pose
                if ee_mode == 'horizontal':
                    self.whole_body.move_end_effector_pose([
                        hsrb_interface.geometry.pose(
                            x=diff_ee_pose[0],
                            y=diff_ee_pose[1],
                        ),
                    ], ref_frame_id='hand_palm_link')
                elif ee_mode == 'vertical':
                    self.whole_body.move_end_effector_pose([
                        hsrb_interface.geometry.pose(
                            x=diff_ee_pose[0],
                            y=diff_ee_pose[1],
                            ek=-np.pi/2
                        ),
                    ], ref_frame_id='hand_palm_link')
                self.whole_body.move_end_effector_by_line((0, 0, 1), diff_ee_pose[2])

                # Grasp object
                self.gripper.apply_force(0.8)

                # Move to terminal pose
                if ee_mode == 'horizontal':
                    self.whole_body.move_end_effector_by_line((1, 0, 0), -diff_ee_pose[0])
                elif ee_mode == 'vertical':
                    self.whole_body.move_end_effector_by_line((0, 1, 0), -diff_ee_pose[0])

                # Remove impedance
                self.whole_body.impedance_config = None

            elif action_name == 'place':
                # Set impedance
                self.whole_body.impedance_config = 'compliance_hard'

                # Move to grasp pose
                if ee_mode == 'horizontal':
                    self.whole_body.move_end_effector_pose([
                        hsrb_interface.geometry.pose(
                            y=diff_ee_pose[1],
                            z=diff_ee_pose[2],
                        ),
                    ], ref_frame_id='hand_palm_link')
                    self.whole_body.move_end_effector_by_line((1, 0, 0), diff_ee_pose[0])
                elif ee_mode == 'vertical':
                    self.whole_body.move_end_effector_pose([
                        hsrb_interface.geometry.pose(
                            x=-diff_ee_pose[1],
                            z=diff_ee_pose[2],
                        ),
                    ], ref_frame_id='hand_palm_link')
                    self.whole_body.move_end_effector_by_line((0, 1, 0), diff_ee_pose[0])

                # Release object
                self.gripper.command(0.5)

                # Move to terminal pose
                if ee_mode == 'horizontal':
                    self.whole_body.move_end_effector_by_line((1, 0, 0), -diff_ee_pose[0])
                    self.whole_body.move_end_effector_pose([
                        hsrb_interface.geometry.pose(
                            y=-diff_ee_pose[1],
                            z=-diff_ee_pose[2]-0.15,
                        ),
                    ], ref_frame_id='hand_palm_link')
                elif ee_mode == 'vertical':
                    self.whole_body.move_end_effector_by_line((0, 1, 0), -diff_ee_pose[0])
                    self.whole_body.move_end_effector_pose([
                        hsrb_interface.geometry.pose(
                            x=diff_ee_pose[1],
                            z=-diff_ee_pose[2]-0.15,
                            ek=np.pi/2
                        ),
                    ], ref_frame_id='hand_palm_link')

                # Remove impedance
                self.whole_body.impedance_config = None

            elif action_name == 'stack':
                # Set impedance
                self.whole_body.impedance_config = 'compliance_hard'

                # Move to grasp pose
                if ee_mode == 'horizontal':
                    self.whole_body.move_end_effector_pose([
                        hsrb_interface.geometry.pose(
                            y=diff_ee_pose[1],
                            z=diff_ee_pose[2],
                        ),
                    ], ref_frame_id='hand_palm_link')
                    self.whole_body.move_end_effector_by_line((0, 1, 0), diff_ee_pose[0])
                elif ee_mode == 'vertical':
                    self.whole_body.move_end_effector_pose([
                        hsrb_interface.geometry.pose(
                            x=-diff_ee_pose[1],
                            z=diff_ee_pose[2],
                        ),
                    ], ref_frame_id='hand_palm_link')
                    self.whole_body.move_end_effector_by_line((0, 1, 0), diff_ee_pose[0])

                # Release object
                self.gripper.command(1.2)

                # Move to terminal pose
                if ee_mode == 'horizontal':
                    self.whole_body.move_end_effector_by_line((1, 0, 0), -diff_ee_pose[0])
                    self.whole_body.move_end_effector_pose([
                        hsrb_interface.geometry.pose(
                            y=-diff_ee_pose[1],
                            z=-diff_ee_pose[2]-0.15,
                        ),
                    ], ref_frame_id='hand_palm_link')
                elif ee_mode == 'vertical':
                    self.whole_body.move_end_effector_by_line((0, 1, 0), -diff_ee_pose[0])
                    self.whole_body.move_end_effector_pose([
                        hsrb_interface.geometry.pose(
                            x=diff_ee_pose[1],
                            z=-diff_ee_pose[2]-0.15,
                            ek=np.pi/2
                        ),
                    ], ref_frame_id='hand_palm_link')

                # Remove impedance
                self.whole_body.impedance_config = None

            else:
                continue


if __name__ == '__main__':
    exec_plan = ExecutePlan()
    exec_plan.execute()