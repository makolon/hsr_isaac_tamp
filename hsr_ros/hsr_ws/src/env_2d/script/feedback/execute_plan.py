#!/usr/bin/env/python3
import tf
import sys
import rospy
import moveit_commander
import numpy as np
from scipy.spatial.transform import Rotation as R

# TODO: fix
sys.path.append('/root/tamp-hsr/hsr_tamp/experiments/env_2d/')
sys.path.append('..')
from tamp_planner import TAMPPlanner
from hsr_py_interface import HSRPyInterface
from force_sensor_interface import ForceSensorInterface
from post_process import PlanModifier
from mocap_interface import MocapInterface
from tf_interface import TfManager
from controller.ik_controller import IKController

from geometry_msgs.msg import Pose
from controller_manager_msgs.srv import ListControllers


class ExecutePlan(object):
    def __init__(self, grasp_type='side'):
        # Initialize moveit
        moveit_commander.roscpp_initialize(sys.argv)

        # Initialize ROS node
        # rospy.init_node('execute_tamp_plan')

        # Feedback rate
        self.rate = rospy.Rate(50)

        # Threshold
        self.move_threshold = 0.05
        self.pick_threshold = 0.05
        self.place_threshold = 0.04 # 0.038
        self.stack_threshold = 0.032 # 0.038
        self.weight_threshold = 500

        # Core module
        self.tamp_planner = TAMPPlanner()
        self.hsr_py = HSRPyInterface()
        self.tf_manager = TfManager()
        self.path_modifier = PlanModifier()
        self.mocap_interface = MocapInterface()
        self.ft_interface = ForceSensorInterface()
        self.ik_controller = IKController()

        # Initialize Robot
        self.initialize_robot(grasp_type)

    def initialize_robot(self, grasp_type='side'):
        self.check_status()

        # Set moveit commander
        self.robot = moveit_commander.RobotCommander()
        self.arm = moveit_commander.MoveGroupCommander('arm', wait_for_servers=0.0)
        self.base = moveit_commander.MoveGroupCommander('base', wait_for_servers=0.0)
        self.gripper = moveit_commander.MoveGroupCommander('gripper', wait_for_servers=0.0)
        self.whole_body = moveit_commander.MoveGroupCommander('whole_body', wait_for_servers=0.0)

        # Set planning parameters
        self.set_planning_parameter()

        # Set arm to configuration position
        self.hsr_py.initialize_arm()

        # Set base to configuration position
        self.hsr_py.initialize_base()

        # Move to neutral
        self.hsr_py.set_task_pose(grasp_type)

    def set_planning_parameter(self):
        # Planning parameters
        self.whole_body.allow_replanning(True)
        self.whole_body.set_planning_time(3)
        self.whole_body.set_pose_reference_frame('map')

        # Scene parameters
        self.scene = moveit_commander.PlanningSceneInterface()
        self.scene.remove_world_object()

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

    def process(self, action_name, args):
        # Get rigid body poses from mocap
        mocap_poses = self.mocap_interface.get_poses()

        # Modify plan
        action_name, modified_action, ee_mode = self.path_modifier.post_process(action_name, args, mocap_poses)

        return modified_action, ee_mode

    def check_ee_mode(self):
        # Get end effector pose
        ee_pose = self.whole_body.get_current_pose()

        ee_rot = np.array([
            ee_pose.pose.orientation.x,
            ee_pose.pose.orientation.y,
            ee_pose.pose.orientation.z,
            ee_pose.pose.orientation.w
        ])
        ee_euler = R.from_quat(ee_rot).as_euler('xyz')

        # Check whether end effector is vertical to ground plane, or horizontal
        if -0.8 < ee_euler[1] and 0.8 > ee_euler[1]:
            ee_mode = 'vertical'
        elif -2.0 < ee_euler[1] and -1.0 > ee_euler[1]:
            ee_mode = 'horizontal'

        return ee_mode

    def create_trajectory(self, diff_pose, num_steps=2):
        # Temporally linear interpolation
        assert num_steps > 1, "Too shot to create waypoints"
        # num_steps = int(np.max(np.abs(diff_pose))//0.03)
        # if num_steps < 2:
        #     num_steps = 2
        traj_x = np.linspace(0.0, diff_pose[0], num_steps)
        traj_y = np.linspace(0.0, diff_pose[1], num_steps)
        traj_z = np.linspace(0.0, diff_pose[2], num_steps)
        trajectory = np.dstack((traj_x, traj_y, traj_z))

        return trajectory

    def set_target_pose(self, trajectory, move_mode='outward', ee_mode='horizontal'):
        if move_mode == 'outward':
            # Extract full trajectory
            diff_ee_traj = trajectory[0]

            goal_way_points = []
            for i in range(1, len(diff_ee_traj)):
                # Transform difference pose on end effector frame to map frame
                target_map_pose = self.tf_manager.transform_coordinate(diff_ee_traj[i])

                if ee_mode == 'horizontal':
                    goal_pose = Pose()
                    goal_pose.position.x = target_map_pose[0]
                    goal_pose.position.y = target_map_pose[1]
                    goal_pose.position.z = target_map_pose[2]
                    goal_pose.orientation.x = 0.5
                    goal_pose.orientation.y = 0.5
                    goal_pose.orientation.z = 0.5
                    goal_pose.orientation.w = -0.5
                    goal_way_points.append(goal_pose)
                elif ee_mode == 'vertical':
                    goal_pose = Pose()
                    goal_pose.position.x = target_map_pose[0]
                    goal_pose.position.y = target_map_pose[1]
                    goal_pose.position.z = target_map_pose[2]
                    goal_pose.orientation.x = 0.0
                    goal_pose.orientation.y = 0.707106781
                    goal_pose.orientation.z = 0.707106781
                    goal_pose.orientation.w = 0.0
                    goal_way_points.append(goal_pose)

            return goal_way_points

        elif move_mode == 'return':
            # Extract full trajectory
            diff_ee_traj = trajectory[0]

            goal_way_points = []
            for i in range(1, len(diff_ee_traj)):
                # Transform difference pose on end effector frame to map frame
                target_map_pose = self.tf_manager.transform_coordinate(diff_ee_traj[i])

                if ee_mode == 'horizontal':
                    goal_pose = Pose()
                    goal_pose.position.x = diff_ee_traj[i][0]
                    goal_pose.position.y = diff_ee_traj[i][1] - 0.12 # TODO: modify
                    goal_pose.position.z = diff_ee_traj[i][2]
                    goal_pose.orientation.x = 0.5
                    goal_pose.orientation.y = 0.5
                    goal_pose.orientation.z = 0.5
                    goal_pose.orientation.w = -0.5
                    goal_way_points.append(goal_pose)
                elif ee_mode == 'vertical':
                    goal_pose = Pose()
                    goal_pose.position.x = diff_ee_traj[i][0]
                    goal_pose.position.y = diff_ee_traj[i][1] - 0.12 # TODO: modify
                    goal_pose.position.z = diff_ee_traj[i][2] - 0.2 # TODO: modify
                    goal_pose.orientation.x = 0.0
                    goal_pose.orientation.y = 0.707106781
                    goal_pose.orientation.z = 0.707106781
                    goal_pose.orientation.w = 0.0
                    goal_way_points.append(goal_pose)

            return goal_way_points

    def create_goal(self, trajectory, ee_mode='horizontal'):
        # Extract full trajectory
        diff_ee_traj = trajectory[0][1]

        # Transform difference pose on end effector frame to map frame
        target_map_pose = self.tf_manager.transform_coordinate(diff_ee_traj)

        if ee_mode == 'horizontal':
            goal_pose = ((target_map_pose[0], target_map_pose[1], target_map_pose[2]),
                         (0.5, 0.5, 0.5, -0.5))
        elif ee_mode == 'vertical':
            goal_pose = ((target_map_pose[0], target_map_pose[1], target_map_pose[2]),
                         (0.0, 0.707106781, 0.707106781, 0.0))

        return goal_pose

    def modify_pose(self, diff_ee_pose, ee_mode):
        if ee_mode == 'horizontal':
            return (diff_ee_pose[0], diff_ee_pose[1], diff_ee_pose[2])
        elif ee_mode == 'vertical':
            return (-diff_ee_pose[1], diff_ee_pose[0], diff_ee_pose[2])

    def execute(self):
        plan = self.plan()

        if plan is None:
            return None

        for i, (action_name, args) in enumerate(plan):
            print('action_name:', action_name)
            if action_name == 'move':
                (_, diff_ee_pose), _ = self.process(action_name, args)

                # Set end effector mode
                ee_mode = self.check_ee_mode()

                # Move to next pose
                diff_ee_pose = self.modify_pose(diff_ee_pose, ee_mode)
                target_traj = self.create_trajectory(diff_ee_pose)
                goal_pose = self.set_target_pose(target_traj, ee_mode=ee_mode)
                (plan, fraction) = self.whole_body.compute_cartesian_path(goal_pose, 0.01, 0.0, False)
                self.whole_body.execute(plan, wait=True)

            elif action_name == 'pick':
                finish = False

                (init_ee_pose, diff_ee_pose), _ = self.process(action_name, args)
                self.hsr_py.fix_task_pose(diff_ee_pose)
                while not finish:
                    (_, diff_ee_pose), ee_mode = self.process(action_name, args)

                    # ee_mode = 'horizontal'
                    if ee_mode == 'horizontal':
                        # Crete trajectory
                        target_traj = self.create_trajectory(diff_ee_pose)
                        goal_pose = self.create_goal(target_traj, ee_mode=ee_mode)

                        # Move and plan to grasp pose
                        self.ik_controller.control(goal_pose)

                        if np.sum(np.absolute(diff_ee_pose)) < self.pick_threshold:
                            finish = True

                            # Grasp object
                            self.hsr_py.close_gripper()

                            # Move to terminal pose
                            target_traj = self.create_trajectory(init_ee_pose)
                            goal_pose = self.set_target_pose(target_traj, move_mode='return')
                            (plan, fraction) = self.whole_body.compute_cartesian_path(goal_pose, 0.01, 0.0, False)
                            self.whole_body.execute(plan, wait=True)

                    elif ee_mode == 'vertical':
                        finish = True

                        # Move to target object & grasp object
                        self.hsr_py.grasp_gear(diff_ee_pose)

                        target_traj = self.create_trajectory(init_ee_pose)
                        goal_pose = self.set_target_pose(target_traj, move_mode='return', ee_mode='vertical')
                        (plan, fraction) = self.whole_body.compute_cartesian_path(goal_pose, 0.01, 0.0, False)
                        self.whole_body.execute(plan, wait=True)                        

                    # Sleep up to rate
                    self.rate.sleep()

            elif action_name == 'place':
                finish = False

                (init_ee_pose, _), _ = self.process(action_name, args)
                while not finish:
                    (_, diff_ee_pose), ee_mode = self.process(action_name, args)

                    # Crete trajectory
                    target_traj = self.create_trajectory(diff_ee_pose)
                    goal_pose = self.create_goal(target_traj, ee_mode=ee_mode)

                    prev_ft_data = self.ft_interface.get_current_force()

                    # Move and plan to place pose
                    self.ik_controller.control(goal_pose)

                    current_ft_data = self.ft_interface.get_current_force()
                    force_difference = self.ft_interface.compute_difference(prev_ft_data, current_ft_data)
                    weight = round(force_difference / 9.81 * 1000, 1)

                    if np.sum(np.absolute(diff_ee_pose)) < self.place_threshold or weight > self.weight_threshold:
                        finish = True

                        # Sleep until stable
                        rospy.sleep(1.0)

                        # Put down end effector
                        self.hsr_py.insert_object(ee_mode)

                        # Release object
                        self.hsr_py.open_gripper()

                        # Move to terminal pose
                        target_traj = self.create_trajectory(init_ee_pose)
                        goal_pose = self.set_target_pose(target_traj, move_mode='return')
                        (plan, fraction) = self.whole_body.compute_cartesian_path(goal_pose, 0.01, 0.0, False)
                        self.whole_body.execute(plan, wait=True)

                    # Sleep up to rate
                    self.rate.sleep()

            elif action_name == 'stack':
                finish = False

                (init_ee_pose, _), _ = self.process(action_name, args)
                while not finish:
                    (_, diff_ee_pose), ee_mode = self.process(action_name, args)

                    # Crete trajectory
                    diff_ee_pose = self.modify_pose(diff_ee_pose, ee_mode)
                    target_traj = self.create_trajectory(diff_ee_pose)
                    goal_pose = self.create_goal(target_traj, ee_mode=ee_mode)

                    # Move and plan to stack pose
                    self.ik_controller.control(goal_pose)

                    if np.sum(np.absolute(diff_ee_pose)) < self.stack_threshold:
                        finish = True

                        # Sleep until stable
                        rospy.sleep(1.0)

                        # Put down end effector
                        self.hsr_py.insert_object(ee_mode)

                        # Release object
                        self.hsr_py.open_gripper()

                        # Move to terminal pose
                        target_traj = self.create_trajectory(init_ee_pose)
                        goal_pose = self.set_target_pose(target_traj, move_mode='return')
                        (plan, fraction) = self.whole_body.compute_cartesian_path(goal_pose, 0.01, 0.0, False)
                        self.whole_body.execute(plan, wait=True)

                    # Sleep up to rate
                    self.rate.sleep()

            else:
                continue


if __name__ == '__main__':
    exec_plan = ExecutePlan()
    exec_plan.execute()