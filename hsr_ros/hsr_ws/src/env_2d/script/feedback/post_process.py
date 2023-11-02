#!/usr/bin/env/python3
import numpy as np

class PlanModifier(object):
    def __init__(self, x_offset=0.0, y_offset=-0.065, z_offset=-0.04):
        self.mocap_x_offset = x_offset
        self.mocap_y_offset = y_offset
        self.mocap_z_offset = z_offset
        self.left_hole_x = -0.10
        self.left_hole_y = -0.18
        self.left_hole_z = 0.125 # 0.135 # 0.125 # 0.12
        self.right_hole_x = 0.095
        self.right_hole_y = -0.18
        self.right_hole_z = 0.142 # 0.15 # 0.145 # 0.14
        self.x_scale = 0.0 # 0.0~0.1
        self.y_scale = 0.065 # 0.0~0.1
        self.z_scale = 0.0 # 0.0~0.0
        self.ee_mode = 'horizontal'
        self.block_rigid_map = {
            'A' : 'red_shaft',
            'B' : 'green_gear',
            'C' : 'yellow_shaft',
            'D' : 'blue_gear'
        }

    def post_process(self, action_name, args, mocap_poses, grasp_type='side'):
        """
        Modify plan using sensor data.

        Args:
            plan (list): plan is trajectory of the tamp.
            robot_pose (list): robot_pose consists of base_pose, end_effector_pose, gripper.
            rigid_poses (dict): rigid_pose consists of captured rigid body poses

        Returns:
            commands (list): commands is modified plan
        """

        ee_pose = mocap_poses['end_effector']

        if grasp_type == 'side':
            if action_name == 'move':
                robot, init_robot_pose, way_point, term_robot_pose = args

                # Modify end effector position from mocap marker attached position
                if self.ee_mode == 'horizontal':
                    ee_pose_x = ee_pose.pose.position.x + self.mocap_x_offset
                    ee_pose_y = ee_pose.pose.position.y - self.mocap_y_offset
                    ee_pose_z = ee_pose.pose.position.z + self.mocap_z_offset
                elif self.ee_mode == 'vertical':
                    ee_pose_x = ee_pose.pose.position.x - self.mocap_z_offset
                    ee_pose_y = ee_pose.pose.position.y - self.mocap_y_offset
                    ee_pose_z = ee_pose.pose.position.z + self.mocap_x_offset

                # Get end effector orientation
                ee_ori_x = ee_pose.pose.orientation.x
                ee_ori_y = ee_pose.pose.orientation.y
                ee_ori_z = ee_pose.pose.orientation.z
                ee_ori_w = ee_pose.pose.orientation.w

                # Initial pose
                init_ee_pose = np.array([
                    ee_pose_x, ee_pose_y, ee_pose_z,
                    ee_ori_x, ee_ori_y, ee_ori_z, ee_ori_w
                ], dtype=np.float64)

                # Calculate difference from target pose in configuration space
                diff_ee_pose = np.array([
                    self.x_scale * 0.0,
                    self.y_scale * (term_robot_pose[0] - init_robot_pose[0]),
                    self.z_scale * (term_robot_pose[1] - init_robot_pose[1])
                ], dtype=np.float64)

                new_command = (action_name, [init_ee_pose, diff_ee_pose], self.ee_mode)

            elif action_name == 'pick':
                robot, block, init_block_pose, grasp_diff_pose, term_robot_pose = args

                # Set pick hyperparameters
                if block == 'A':
                    ee_x_offset = 0.03
                    ee_y_offset = 0.0
                    ee_z_offset = 0.04
                    self.ee_mode = 'horizontal'
                elif block == 'B':
                    ee_x_offset = -0.05
                    ee_y_offset = 0.0
                    ee_z_offset = 0.11
                    self.ee_mode = 'vertical'
                elif block == 'C':
                    ee_x_offset = 0.03
                    ee_y_offset = 0.0
                    ee_z_offset = 0.04
                    self.ee_mode = 'horizontal'
                elif block == 'D':
                    ee_x_offset = -0.05
                    ee_y_offset = 0.0
                    ee_z_offset = 0.13
                    self.ee_mode = 'vertical'

                # Modify end effector position from mocap marker attached position
                if self.ee_mode == 'horizontal':
                    ee_pose_x = ee_pose.pose.position.x + self.mocap_x_offset
                    ee_pose_y = ee_pose.pose.position.y - self.mocap_y_offset
                    ee_pose_z = ee_pose.pose.position.z + self.mocap_z_offset
                elif self.ee_mode == 'vertical':
                    ee_pose_x = ee_pose.pose.position.x - self.mocap_z_offset
                    ee_pose_y = ee_pose.pose.position.y - self.mocap_y_offset
                    ee_pose_z = ee_pose.pose.position.z + self.mocap_x_offset

                # Get end effector orientation
                ee_ori_x = ee_pose.pose.orientation.x
                ee_ori_y = ee_pose.pose.orientation.y
                ee_ori_z = ee_pose.pose.orientation.z
                ee_ori_w = ee_pose.pose.orientation.w

                # Initial pose
                init_ee_pose = np.array([
                    ee_pose_x, ee_pose_y, ee_pose_z,
                    ee_ori_x, ee_ori_y, ee_ori_z, ee_ori_w
                ], dtype=np.float64)

                # Map symbolic block name to real block name
                rigid_name = self.block_rigid_map[block]
                rigid_pose = mocap_poses[rigid_name]

                # Calculate grasp pose in configuration space
                rigid_pose_x = rigid_pose.pose.position.x
                rigid_pose_y = rigid_pose.pose.position.y
                rigid_pose_z = rigid_pose.pose.position.z
                diff_ee_pose = np.array([
                    rigid_pose_z - ee_pose_z - ee_x_offset,
                    rigid_pose_x - ee_pose_x - ee_y_offset,
                    rigid_pose_y - ee_pose_y - ee_z_offset,
                ], dtype=np.float64)

                new_command = (action_name, [init_ee_pose, diff_ee_pose], self.ee_mode)

            elif action_name == 'place':
                robot, block, init_block_pose, grasp_diff_pose, term_robot_pose = args

                # Get hole pose
                base_pose = mocap_poses['base']

                # Modify hole position from mocap marker
                if block == 'A':
                    hole_pose_x = base_pose.pose.position.x + self.left_hole_x
                    hole_pose_y = base_pose.pose.position.y + self.left_hole_y
                    hole_pose_z = base_pose.pose.position.z + self.left_hole_z
                    self.ee_mode = 'horizontal'
                elif block == 'B':
                    hole_pose_x = base_pose.pose.position.x + self.left_hole_x
                    hole_pose_y = base_pose.pose.position.y + self.left_hole_y
                    hole_pose_z = base_pose.pose.position.z + self.left_hole_z
                    self.ee_mode = 'vertical'
                elif block == 'C':
                    hole_pose_x = base_pose.pose.position.x + self.right_hole_x
                    hole_pose_y = base_pose.pose.position.y + self.right_hole_y
                    hole_pose_z = base_pose.pose.position.z + self.right_hole_z
                    self.ee_mode = 'horizontal'
                elif block == 'D':
                    hole_pose_x = base_pose.pose.position.x + self.right_hole_x
                    hole_pose_y = base_pose.pose.position.y + self.right_hole_y
                    hole_pose_z = base_pose.pose.position.z + self.right_hole_z
                    self.ee_mode = 'vertical'

                # Modify end effector position from mocap marker attached position
                if self.ee_mode == 'horizontal':
                    ee_pose_x = ee_pose.pose.position.x + self.mocap_x_offset
                    ee_pose_y = ee_pose.pose.position.y - self.mocap_y_offset
                    ee_pose_z = ee_pose.pose.position.z + self.mocap_z_offset
                elif self.ee_mode == 'vertical':
                    ee_pose_x = ee_pose.pose.position.x - self.mocap_z_offset
                    ee_pose_y = ee_pose.pose.position.y - self.mocap_y_offset
                    ee_pose_z = ee_pose.pose.position.z + self.mocap_x_offset

                # Get end effector orientation
                ee_ori_x = ee_pose.pose.orientation.x
                ee_ori_y = ee_pose.pose.orientation.y
                ee_ori_z = ee_pose.pose.orientation.z
                ee_ori_w = ee_pose.pose.orientation.w

                # Initial pose
                init_ee_pose = np.array([
                    ee_pose_x, ee_pose_y, ee_pose_z,
                    ee_ori_x, ee_ori_y, ee_ori_z, ee_ori_w
                ], dtype=np.float64)

                # Map symbolic block name to real block name
                rigid_name = self.block_rigid_map[block]
                rigid_pose = mocap_poses[rigid_name]

                # Calculate place pose in configuration space
                rigid_pose_x = rigid_pose.pose.position.x
                rigid_pose_y = rigid_pose.pose.position.y
                rigid_pose_z = rigid_pose.pose.position.z

                diff_ee_pose = np.array([
                    hole_pose_z - rigid_pose_z,
                    hole_pose_x - rigid_pose_x,
                    hole_pose_y - rigid_pose_y,
                ], dtype=np.float64)

                new_command = (action_name, [init_ee_pose, diff_ee_pose], self.ee_mode)

            elif action_name == 'stack':
                robot, u_block, u_pose, grasp_diff_pose, \
                    term_robot_pose, l_block, l_pose = args

                # Get hole pose
                base_pose = mocap_poses['base']

                # Modify hole position from mocap marker
                if u_block == 'A':
                    hole_pose_x = base_pose.pose.position.x + self.left_hole_x
                    hole_pose_y = base_pose.pose.position.y + self.left_hole_y
                    hole_pose_z = base_pose.pose.position.z + self.left_hole_z - 0.01 # TODO: modify
                    self.ee_mode = 'horizontal'
                elif u_block == 'B':
                    hole_pose_x = base_pose.pose.position.x + self.left_hole_x + 0.0175 # TODO: modify
                    hole_pose_y = base_pose.pose.position.y + self.left_hole_y
                    hole_pose_z = base_pose.pose.position.z + self.left_hole_z - 0.015 # TODO: modify
                    self.ee_mode = 'vertical'
                elif u_block == 'C':
                    hole_pose_x = base_pose.pose.position.x + self.right_hole_x
                    hole_pose_y = base_pose.pose.position.y + self.right_hole_y
                    hole_pose_z = base_pose.pose.position.z + self.right_hole_z - 0.01 # TODO: modify
                    self.ee_mode = 'horizontal'
                elif u_block == 'D':
                    hole_pose_x = base_pose.pose.position.x + self.right_hole_x + 0.0175 # TODO: modify
                    hole_pose_y = base_pose.pose.position.y + self.right_hole_y
                    hole_pose_z = base_pose.pose.position.z + self.right_hole_z - 0.015 # TODO: modify
                    self.ee_mode = 'vertical'

                # Modify end effector position from mocap marker attached position
                if self.ee_mode == 'horizontal':
                    ee_pose_x = ee_pose.pose.position.x + self.mocap_x_offset
                    ee_pose_y = ee_pose.pose.position.y - self.mocap_y_offset
                    ee_pose_z = ee_pose.pose.position.z + self.mocap_z_offset
                elif self.ee_mode == 'vertical':
                    ee_pose_x = ee_pose.pose.position.x - self.mocap_z_offset
                    ee_pose_y = ee_pose.pose.position.y - self.mocap_y_offset
                    ee_pose_z = ee_pose.pose.position.z + self.mocap_x_offset

                # Get end effector orientation
                ee_ori_x = ee_pose.pose.orientation.x
                ee_ori_y = ee_pose.pose.orientation.y
                ee_ori_z = ee_pose.pose.orientation.z
                ee_ori_w = ee_pose.pose.orientation.w

                # Initial pose
                init_ee_pose = np.array([
                    ee_pose_x, ee_pose_y, ee_pose_z,
                    ee_ori_x, ee_ori_y, ee_ori_z, ee_ori_w
                ], dtype=np.float64)

                # Map symbolic block name to real block name
                u_rigid_name = self.block_rigid_map[u_block]
                l_rigid_name = self.block_rigid_map[l_block]
                u_rigid_pose = mocap_poses[u_rigid_name]
                l_rigid_pose = mocap_poses[l_rigid_name]

                # Calculate stack pose in configuration space
                rigid_pose_x = u_rigid_pose.pose.position.x
                rigid_pose_y = u_rigid_pose.pose.position.y
                rigid_pose_z = u_rigid_pose.pose.position.z
                diff_ee_pose = np.array([
                    hole_pose_z - rigid_pose_z,
                    hole_pose_x - rigid_pose_x,
                    hole_pose_y - rigid_pose_y,
                ], dtype=np.float64)

                new_command = (action_name, [init_ee_pose, diff_ee_pose], self.ee_mode)

            else:
                pass

        elif grasp_type == 'top':
            ee_pose_x = ee_pose.pose.position.x + self.mocap_x_offset
            ee_pose_y = ee_pose.pose.position.y + self.mocap_y_offset
            ee_pose_z = ee_pose.pose.position.z + self.mocap_z_offset

            if action_name == 'move':
                robot, init_robot_pose, way_point, term_robot_pose = args

                # Calculate grasp pose in configuration space
                init_ee_pose = np.array([
                    ee_pose_x,
                    ee_pose_y,
                    ee_pose_z
                ])
                diff_ee_pose = np.array([
                    self.x_scale * 0.0,
                    self.y_scale * (term_robot_pose[0] - init_robot_pose[0]),
                    self.z_scale * (term_robot_pose[1] - init_robot_pose[1])
                ])
                term_ee_pose = init_ee_pose + diff_ee_pose

                new_command = (action_name, [init_ee_pose, diff_ee_pose, term_ee_pose])

            elif action_name == 'pick':
                robot, block, init_block_pose, grasp_diff_pose, term_robot_pose = args

                # Map symbolic block name to real block name
                rigid_name = self.block_rigid_map[block]
                rigid_pose = mocap_poses[rigid_name]

                # Get curent end effector pose
                init_ee_pose = np.array([
                    ee_pose_x,
                    ee_pose_y,
                    ee_pose_z
                ])

                # Calculate grasp pose in configuration space
                rigid_pose_x = rigid_pose.pose.position.x
                rigid_pose_y = rigid_pose.pose.position.y
                rigid_pose_z = rigid_pose.pose.position.z
                diff_ee_pose = np.array([
                    ee_pose_y - rigid_pose_y,
                    ee_pose_x - rigid_pose_x,
                    ee_pose_z - rigid_pose_z
                ])

                # Get terminal end effector pose as reverse of diff_ee_pose
                term_ee_pose = -diff_ee_pose

                new_command = (action_name, [init_ee_pose, diff_ee_pose, term_ee_pose])

            elif action_name == 'place':
                robot, block, init_block_pose, grasp_diff_pose, term_robot_pose = args

                # Get hole pose
                base_pose = mocap_poses['base']

                if block == 'A' or block == 'B':
                    hole_pose_x = base_pose.pose.position.x + self.left_hole_x
                    hole_pose_y = base_pose.pose.position.y + self.left_hole_y
                    hole_pose_z = base_pose.pose.position.z + self.left_hole_z

                elif block == 'C' or block == 'D':
                    hole_pose_x = base_pose.pose.position.x + self.right_hole_x
                    hole_pose_y = base_pose.pose.position.y + self.right_hole_y
                    hole_pose_z = base_pose.pose.position.z + self.right_hole_z

                # Get curent end effector pose
                init_ee_pose = np.array([
                    ee_pose_x,
                    ee_pose_y,
                    ee_pose_z
                ])

                # Calculate place pose in configuration space
                diff_ee_pose = np.array([
                    ee_pose_y - hole_pose_y,
                    ee_pose_x - hole_pose_x,
                    ee_pose_z - hole_pose_z
                ])

                # Get terminal end effector pose as reverse of diff_ee_pose
                term_ee_pose = -diff_ee_pose

                new_command = (action_name, [init_ee_pose, diff_ee_pose, term_ee_pose])

            elif action_name == 'stack':
                robot, u_block, u_pose, grasp_diff_pose, \
                    term_robot_pose, l_block, l_pose = args

                # Map symbolic block name to real block name
                u_rigid_name = self.block_rigid_map[u_block]
                l_rigid_name = self.block_rigid_map[l_block]
                u_rigid_pose = mocap_poses[u_rigid_name]
                l_rigid_pose = mocap_poses[l_rigid_name]

                # Get hole pose
                base_pose = mocap_poses['base']

                if u_block == 'A' or u_block == 'B':
                    hole_pose_x = base_pose.pose.position.x + self.left_hole_x
                    hole_pose_y = base_pose.pose.position.y + self.left_hole_y
                    hole_pose_z = base_pose.pose.position.z + self.left_hole_z

                elif u_block == 'C' or u_block == 'D':
                    hole_pose_x = base_pose.pose.position.x + self.right_hole_x
                    hole_pose_y = base_pose.pose.position.y + self.right_hole_y
                    hole_pose_z = base_pose.pose.position.z + self.right_hole_z

                # Get curent end effector pose
                init_ee_pose = np.array([
                    ee_pose_x,
                    ee_pose_y,
                    ee_pose_z
                ])

                # Calculate stack pose in configuration space
                diff_ee_pose = np.array([
                    ee_pose_y - hole_pose_y,
                    ee_pose_x - hole_pose_x,
                    ee_pose_z - hole_pose_z
                ])

                # Get terminal end effector pose as reverse of diff_ee_pose
                term_ee_pose = -diff_ee_pose

                new_command = (action_name, [init_ee_pose, diff_ee_pose, term_ee_pose])

            else:
                pass

        return new_command