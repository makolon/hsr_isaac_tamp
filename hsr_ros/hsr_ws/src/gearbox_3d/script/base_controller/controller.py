#!/usr/bin/env/python3
import sys
import rospy
import numpy as np
from scipy.spatial.transform import Rotation as R

# TODO: fix
sys.path.append('/root/tamp-hsr/hsr_tamp/experiments/gearbox_3d/')
sys.path.append('..')
from tamp_planner import TAMPPlanner
from hsr_interface import HSRInterface
from tf_interface import TfManager
from path_interface import PlanModifier
from mocap_interface import MocapInterface
from force_sensor_interface import ForceSensorInterface

from geometry_msgs.msg import Twist
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint


class BaseController(object):
    def __init__(self):
        # Initialize ROS node
        rospy.init_node('execute_tamp_plan')

        # Feedback rate
        self.control_freq = 10.0 # 5Hz reso 0.02, 10Hz reso 0.01, 30Hz reso 0.005
        self.rate = rospy.Rate(self.control_freq)

        # Core module
        self.tamp_planner = TAMPPlanner()
        self.hsr_interface = HSRInterface()
        self.tf_manager = TfManager()
        self.path_modifier = PlanModifier()
        self.mocap_interface = MocapInterface()
        self.ft_interface = ForceSensorInterface()

        # Publisher
        self.arm_pub = rospy.Publisher('/hsrb/arm_trajectory_controller/command', JointTrajectory, queue_size=1)
        self.base_pub = rospy.Publisher('/hsrb/command_velocity', Twist, queue_size=1)

        # Initialize Robot
        self.initialize_robot()

        # Initialize TAMP
        self.initialize_tamp()

    def initialize_robot(self):
        self.check_status()

        # Set gripper to configuration position
        self.hsr_interface.open_gripper()

        # Set arm to configuration position
        self.hsr_interface.initialize_arm()

        # Set base to configuration position
        self.hsr_interface.initialize_base()

    def initialize_tamp(self):
        # Get object poses
        object_poses = self.mocap_interface.get_poses()

        # Get robot poses
        self.robot_joints = ['joint_x', 'joint_y', 'joint_rz', 'arm_lift_joint', 'arm_flex_joint',
                             'arm_roll_joint', 'wrist_flex_joint', 'wrist_roll_joint']
        robot_poses = self.hsr_interface.get_joint_positions()

        # Initialize tamp simulator
        observations = (robot_poses, object_poses)
        self.tamp_planner.initialize(observations)

    def set_base_pose(self, base_velocity):
        base_command = Twist()

        # Set base trajectory
        assert len(base_velocity) == 3, "Does not match the size of base pose"
        base_command.linear.x = base_velocity[0]
        base_command.linear.y = base_velocity[1]
        base_command.angular.z = base_velocity[2]

        return base_command

    def set_arm_pose(self, arm_pose, duration=0.0):
        arm_traj = JointTrajectory()
        arm_traj.joint_names = ['arm_lift_joint', 'arm_flex_joint',
                                'arm_roll_joint', 'wrist_flex_joint', 'wrist_roll_joint']

        # Set arm trajectory
        assert len(arm_pose) == 5, "Does not match the size of base pose"
        arm_p = JointTrajectoryPoint()
        arm_p.positions = arm_pose
        arm_p.velocities = np.ones(len(arm_pose)) * duration
        arm_p.time_from_start = rospy.Duration(1)
        arm_traj.points = [arm_p]

        return arm_traj

    def check_status(self):
        # Wait for publisher has built
        while self.base_pub.get_num_connections() == 0:
            rospy.sleep(0.1)
        while self.arm_pub.get_num_connections() == 0:
            rospy.sleep(0.1)

    def plan(self):
        # Run TAMP
        plan, _, _ = self.tamp_planner.plan()

        return plan

    def process(self, action_name, args):
        # Get object names
        object_names = self.tamp_planner.tamp_problem.body_names

        # Modify plan
        action_name, object_name, modified_action = self.path_modifier.post_process(action_name, object_names, args)

        return action_name, object_name, modified_action

    def calculate_base_command(self, target_pose, kp=5.0, kd=0.05):
        curr_pos, curr_rot = self.mocap_interface.get_pose('hsr_base')
        curr_rot = R.from_quat(curr_rot).as_euler('xyz')
        curr_pose = np.array([curr_pos[0], curr_pos[1], curr_rot[2]], dtype=np.float32)
        diff_pose = np.array(target_pose, dtype=np.float32) - np.array(curr_pose, dtype=np.float32)

        curr_vel = self.hsr_interface.get_joint_velocities(group='base')
        diff_vel = -np.array(curr_vel)

        # Calculate velocity command
        command = kp * diff_pose - kd * diff_vel
        command[2] = 0.2 * command[2]

        return command
    
    def calculate_arm_command(self, target_pose, kp=2.0, kd=0.05):
        curr_pose = self.hsr_interface.get_joint_positions(group='arm')
        diff_pose = np.array(target_pose, dtype=np.float32) - np.array(curr_pose, dtype=np.float32)

        # Get velocity
        curr_vel = self.hsr_interface.get_joint_velocities(group='arm')
        diff_vel = -np.array(curr_vel)

        # Calculate position command
        command = kp * diff_pose - kd * diff_vel
        command += np.array(curr_pose, dtype=np.float32)

        return command