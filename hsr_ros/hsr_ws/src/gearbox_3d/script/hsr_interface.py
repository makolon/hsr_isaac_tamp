#!/usr/bin/env/python3
import rospy
import hsrb_interface
import numpy as np
from scipy.spatial.transform import Rotation as R
from controller_manager_msgs.srv import ListControllers
from control_msgs.msg import JointTrajectoryControllerState


class HSRInterface(object):
    def __init__(self, standalone=False):
        # Initialize ROS node
        if standalone:
            rospy.init_node('hsr_interface')

        # Check server status
        self.check_status()

        self.robot = hsrb_interface.Robot()
        self.omni_base = self.robot.get('omni_base')
        self.gripper = self.robot.get('gripper')
        self.whole_body = self.robot.get('whole_body')

        self.base_sub = rospy.Subscriber('/hsrb/omni_base_controller/state', JointTrajectoryControllerState, self.base_callback)
        self.arm_sub = rospy.Subscriber('/hsrb/arm_trajectory_controller/state', JointTrajectoryControllerState, self.arm_callback)

        self.base_pos, self.base_vel, self.base_acc = None, None, None
        self.arm_pos, self.arm_vel, self.arm_acc = None, None, None

        self.base_joints = ['joint_x', 'joint_y', 'joint_rz']
        self.arm_joints = ['arm_lift_joint', 'arm_flex_joint', 'arm_roll_joint', 'wrist_flex_joint', 'wrist_roll_joint']
        self.joint_ids_to_name = {'joint_x': 0, 'joint_y': 1, 'joint_rz': 2,
                                  'arm_lift_joint': 0, 'arm_flex_joint': 1, 'arm_roll_joint': 2,
                                  'wrist_flex_joint': 3, 'wrist_roll_joint': 4}

    def initialize_arm(self):
        # Initialize arm position
        self.whole_body.move_to_go()

        # Set TAMP pose
        self.whole_body.move_to_joint_positions({
            'arm_lift_joint': 0.1,
            'arm_flex_joint': -np.pi/2,
            'arm_roll_joint': 0.0,
            'wrist_flex_joint': 0.0,
            'wrist_roll_joint': 0.0,
        })

    def initialize_base(self):
        # Initialize base position
        self.omni_base.go_abs(0.0, 0.0, 0.0, 30.0)

    def set_task_pose(self, grasp_type='side'):
        # Set base to configuration position
        self.omni_base.go_abs(-0.80, 0.75, np.pi/2, 30.0)

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

    def get_link_pose(self, link):
        tf_pose = self.whole_body._tf2_buffer.lookup_transform('map', link, rospy.Time(0))
        link_pose = [[tf_pose.transform.translation.x,
                      tf_pose.transform.translation.y,
                      tf_pose.transform.translation.z],
                     [tf_pose.transform.rotation.x,
                      tf_pose.transform.rotation.y,
                      tf_pose.transform.rotation.z,
                      tf_pose.transform.rotation.w]]
        return link_pose

    def get_joint_limits(self, joint):
        if joint == 'odom_x':
            limit = (-10.0, 10.0)
        elif joint == 'odom_y':
            limit = (-10.0, 10.0)
        elif joint == 'odom_t':
            limit = (-10.0, 10.0)
        else:
            limit = self.whole_body.joint_limits[joint]
        return limit

    def get_custom_limits(self, base_joints, arm_joints, custom_limits={}):
        joint_limits = []
        for joint in base_joints:
            if joint in custom_limits:
                joint_limits.append(custom_limits[joint])
            else:
                joint_limits.append(self.get_joint_limits(joint))
        for joint in arm_joints:
            if joint in custom_limits:
                joint_limits.append(custom_limits[joint])
            else:
                joint_limits.append(self.get_joint_limits(joint))
        return zip(*joint_limits)

    def get_joint_position(self, joint, group=None):
        if group == 'base':
            joint_position = self.base_pos[self.joint_ids_to_name[joint]]
        elif group == 'arm':
            joint_position = self.arm_pos[self.joint_ids_to_name[joint]]
        else:
            raise ValueError(joint)
        return joint_position

    def get_joint_positions(self, group=None):
        joint_positions = []
        if group == 'base':
            for base_joint in self.base_joints:
                base_pos = self.get_joint_position(base_joint, 'base')
                joint_positions.append(base_pos)
        elif group == 'arm':
            for arm_joint in self.arm_joints:
                arm_pos = self.get_joint_position(arm_joint, 'arm')
                joint_positions.append(arm_pos)
        else:
            for base_joint in self.base_joints:
                base_pos = self.get_joint_position(base_joint, 'base')
                joint_positions.append(base_pos)
            for arm_joint in self.arm_joints:
                arm_pos = self.get_joint_position(arm_joint, 'arm')
                joint_positions.append(arm_pos)    
        return joint_positions

    def get_joint_velocity(self, joint, group=None):
        if group == 'base':
            joint_velocity = self.base_vel[self.joint_ids_to_name[joint]]
        elif group == 'arm':
            joint_velocity = self.arm_vel[self.joint_ids_to_name[joint]]
        else:
            raise ValueError(joint)
        return joint_velocity

    def get_joint_velocities(self, group=None):
        joint_velocities = []
        if group == 'base':
            for base_joint in self.base_joints:
                joint_vel = self.get_joint_velocity(base_joint, 'base')
                joint_velocities.append(joint_vel)
        elif group == 'arm':
            for arm_joint in self.arm_joints:
                joint_vel = self.get_joint_velocity(arm_joint, 'arm')
                joint_velocities.append(joint_vel)
        else:
            for base_joint in self.base_joints:
                joint_vel = self.get_joint_velocity(base_joint, 'base')
                joint_velocities.append(joint_vel)
            for arm_joint in self.arm_joints:
                joint_vel = self.get_joint_velocity(arm_joint, 'arm')
                joint_velocities.append(joint_vel)
        return joint_velocities

    def get_joint_acceleration(self, joint, group=None):
        if group == 'base':
            joint_acceleration = self.base_acc[self.joint_ids_to_name[joint]]
        elif group == 'arm':
            joint_acceleration = self.arm_acc[self.joint_ids_to_name[joint]]
        else:
            raise ValueError(joint)
        return joint_acceleration

    def get_joint_accelerations(self, group=None):
        joint_acceralataions = []
        if group == 'base':
            for base_joint in self.base_joints:
                joint_acc = self.get_joint_acceleration(base_joint, 'base')
                joint_acceralataions.append(joint_acc)
        elif group == 'arm':
            for arm_joint in self.arm_joints:
                joint_acc = self.get_joint_acceleration(arm_joint, 'arm')
                joint_acceralataions.append(joint_acc)
        else:
            for base_joint in self.base_joints:
                joint_acc = self.get_joint_acceleration(base_joint, 'base')
                joint_acceralataions.append(joint_acc)
            for arm_joint in self.arm_joints:
                joint_acc = self.get_joint_acceleration(arm_joint, 'arm')
                joint_acceralataions.append(joint_acc)
        return joint_acceralataions

    def base_callback(self, data):
        self.base_pos = data.actual.positions
        self.base_vel = data.actual.velocities
        self.base_acc = data.actual.accelerations

    def arm_callback(self, data):
        self.arm_pos = data.actual.positions
        self.arm_vel = data.actual.velocities
        self.arm_acc = data.actual.accelerations


if __name__ == '__main__':
    hsr_interface = HSRInterface()
    hsr_interface.initialize_arm()
    hsr_interface.initialize_base()
    print('positions:', hsr_interface.get_joint_positions())
    print('velocities:', hsr_interface.get_joint_velocities())
