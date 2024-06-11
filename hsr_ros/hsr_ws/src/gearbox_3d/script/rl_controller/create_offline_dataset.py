#!/usr/bin/env/python3
import os
import sys
import yaml
import h5py
import rospy
import torch
import d3rlpy
import numpy as np
from scipy.interpolate import interp1d
from scipy.spatial.transform import Rotation as R

# TODO: fix
sys.path.append('/root/tamp-hsr/hsr_tamp/experiments/gearbox_3d/')
sys.path.append('..')
from tamp_planner import TAMPPlanner
from execute_plan import ExecutePlan, norm_diff_pos, norm_diff_rot

from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint



class CreateDataset(ExecutePlan):
    def __init__(self, standalone=False):
        super(CreateDataset, self).__init__()

        # Set params
        self.pick_distance_scale = 0.01
        self.place_distance_scale = 0.01
        self.insert_distance_scale = 0.01
        self.pick_success_bonus = 10.0
        self.place_success_bonus = 10.0
        self.insert_success_bonus = 10.0

        # Initialize dataset
        self.initialize_dataset()

    def initialize_dataset(self):
        self.pick_dataset = {'observations': [],
                             'actions': [],
                             'rewards': [],
                             'terminals': []}
        self.place_dataset = {'observations': [],
                              'actions': [],
                              'rewards': [],
                              'terminals': []}
        self.insert_dataset = {'observations': [],
                               'actions': [],
                               'rewards': [],
                               'terminals': []}

    def check_skill(self):
        flag = input('reset?')
        if flag == 'y' or flag == 'yes':
            return True
        else:
            return False

    def create_dataset(self):
        # Create dataset
        pick_dataset = d3rlpy.dataset.MDPDataset(
            observations=np.array(self.pick_dataset['observations']),
            actions=np.array(self.pick_dataset['actions']),
            rewards=np.array(self.pick_dataset['rewards']),
            terminals=np.array(self.pick_dataset['terminals'])
        )
        place_dataset = d3rlpy.dataset.MDPDataset(
            observations=np.array(self.place_dataset['observations']),
            actions=np.array(self.place_dataset['actions']),
            rewards=np.array(self.place_dataset['rewards']),
            terminals=np.array(self.place_dataset['terminals'])
        )
        insert_dataset = d3rlpy.dataset.MDPDataset(
            observations=np.array(self.insert_dataset['observations']),
            actions=np.array(self.insert_dataset['actions']),
            rewards=np.array(self.insert_dataset['rewards']),
            terminals=np.array(self.insert_dataset['terminals'])
        )

        # Save dataset
        pick_dataset.dump('dataset/pick_dataset.h5')
        place_dataset.dump('dataset/place_dataset.h5')
        insert_dataset.dump('dataset/insert_dataset.h5')

    #########################
    ####     RL utils    ####
    #########################

    def calc_pick_reward(self, obj_name, target_parts_pos, target_parts_rot):
        ee_pos, ee_rot = self.mocap_interface.get_pose('end_effector')
        obj_pos, obj_rot = self.mocap_interface.get_pose(obj_name)

        # To torch.Tensor & device
        ee_pos = torch.tensor(ee_pos, device=self._device)
        ee_rot = torch.tensor(ee_rot, device=self._device)
        obj_pos = torch.tensor(obj_pos, device=self._device)
        obj_rot = torch.tensor(obj_rot, device=self._device)
        target_parts_pos = torch.tensor(target_parts_pos, device=self._device)
        target_parts_rot = torch.tensor(target_parts_rot, device=self._device)

        # Distance from hand to the target object
        dist = torch.norm(ee_pos - obj_pos, p=2, dim=-1)
        dist_reward = 1.0 / (1.0 + dist ** 2)
        dist_reward *= dist_reward
        if 'shaft' in obj_name:
            if dist <= 0.06:
                dist_reward *= 2
        elif 'gear' in obj_name:
            if dist <= 0.15:
                dist_reward *= 2

        reward = dist_reward * self.pick_distance_scale

        # Calculate difference between target object pose and final object pose
        target_pos_dist = norm_diff_pos(obj_pos, target_parts_pos)
        target_rot_dist = norm_diff_rot(obj_rot, target_parts_rot)

        target_pos_dist_reward = 1.0 / (1.0 + target_pos_dist ** 2)
        target_pos_dist_reward *= target_pos_dist_reward
        target_pos_dist_reward = torch.where(target_pos_dist <= 0.05, target_pos_dist_reward * 2, target_pos_dist_reward)

        target_rot_dist_reward = 1.0 / (1.0 + target_rot_dist ** 2)
        target_rot_dist_reward *= target_rot_dist_reward
        target_rot_dist_reward = torch.where(target_rot_dist <= 0.05, target_rot_dist_reward * 2, target_rot_dist_reward)

        reward += target_pos_dist_reward * self.pick_distance_scale
        reward += target_rot_dist_reward * self.pick_distance_scale

        # Check if block is picked up and close to target pose
        pick_success = self.check_pick_success(obj_name)
        reward += pick_success * self.pick_success_bonus

        return reward

    def calc_place_reward(self, obj_name, target_parts_pos, target_parts_rot):
        reward = torch.zeros(1, device=self._device)

        obj_pos, obj_rot = self.mocap_interface.get_pose(obj_name)

        # To torch.Tensor & device
        obj_pos = torch.tensor(obj_pos, device=self._device)
        obj_rot = torch.tensor(obj_rot, device=self._device)
        target_parts_pos = torch.tensor(target_parts_pos, device=self._device)
        target_parts_rot = torch.tensor(target_parts_rot, device=self._device)

        # Calculate difference between target object pose and final object pose
        target_pos_dist = norm_diff_pos(obj_pos, target_parts_pos)
        target_rot_dist = norm_diff_rot(obj_rot, target_parts_rot)

        target_pos_dist_reward = 1.0 / (1.0 + target_pos_dist ** 2)
        target_pos_dist_reward *= target_pos_dist_reward
        target_pos_dist_reward = torch.where(target_pos_dist <= 0.05, target_pos_dist_reward * 2, target_pos_dist_reward)

        target_rot_dist_reward = 1.0 / (1.0 + target_rot_dist ** 2)
        target_rot_dist_reward *= target_rot_dist_reward
        target_rot_dist_reward = torch.where(target_rot_dist <= 0.05, target_rot_dist_reward * 2, target_rot_dist_reward)

        reward += target_pos_dist_reward * self.place_distance_scale
        reward += target_rot_dist_reward * self.place_distance_scale

        # Check if block is picked up and above table
        place_success = self.check_place_success(obj_name, target_parts_pos)
        reward += place_success * self.insert_success_bonus

        return reward

    def calc_insert_reward(self, obj_name, target_parts_pos, target_parts_rot):
        reward = torch.zeros(1, device=self._device)

        obj_pos, obj_rot = self.mocap_interface.get_pose(obj_name)

        # To torch.Tensor & device
        obj_pos = torch.tensor(obj_pos, device=self._device)
        obj_rot = torch.tensor(obj_rot, device=self._device)
        target_parts_pos = torch.tensor(target_parts_pos, device=self._device)
        target_parts_rot = torch.tensor(target_parts_rot, device=self._device)

        # Calculate difference between target object pose and final object pose
        target_pos_dist = norm_diff_pos(obj_pos, target_parts_pos)
        target_rot_dist = norm_diff_rot(obj_rot, target_parts_rot)

        target_pos_dist_reward = 1.0 / (1.0 + target_pos_dist ** 2)
        target_pos_dist_reward *= target_pos_dist_reward
        target_pos_dist_reward = torch.where(target_pos_dist <= 0.05, target_pos_dist_reward * 2, target_pos_dist_reward)

        target_rot_dist_reward = 1.0 / (1.0 + target_rot_dist ** 2)
        target_rot_dist_reward *= target_rot_dist_reward
        target_rot_dist_reward = torch.where(target_rot_dist <= 0.05, target_rot_dist_reward * 2, target_rot_dist_reward)

        reward += target_pos_dist_reward * self.insert_distance_scale
        reward += target_rot_dist_reward * self.insert_distance_scale

        # Check if block is picked up and above table
        insert_success = self.check_insert_success(obj_name, target_parts_pos, target_parts_rot)
        reward += insert_success * self.insert_success_bonus

        return reward
    
    def check_pick_success(self, obj_name):
        obj_pos, _ = self.mocap_interface.get_pose(obj_name)

        return 1 if obj_pos[2] > 0.30 else 0

    def check_place_success(self, obj_name, target_parts_pos):
        obj_pos, _ = self.mocap_interface.get_pose(obj_name)

        # To torch.Tensor & device
        obj_pos = torch.tensor(obj_pos, device=self._device)
        target_parts_pos = torch.tensor(target_parts_pos, device=self._device)

        # Calculate difference
        pos_dist = norm_diff_pos(obj_pos, target_parts_pos)

        return 1 if pos_dist < 0.02 else 0

    def check_insert_success(self, obj_name, target_parts_pos, target_parts_rot):
        obj_pos, obj_rot = self.mocap_interface.get_pose(obj_name)

        # To torch.Tensor & device
        obj_pos = torch.tensor(obj_pos, device=self._device)
        obj_rot = torch.tensor(obj_rot, device=self._device)
        target_parts_pos = torch.tensor(target_parts_pos, device=self._device)
        target_parts_rot = torch.tensor(target_parts_rot, device=self._device)
    
        # Calculate difference
        pos_dist = norm_diff_pos(obj_pos, target_parts_pos)
        rot_dist = norm_diff_rot(obj_rot, target_parts_rot)

        return 1 if pos_dist < 0.03 and rot_dist < 0.03 else 0

    ##########################
    #### Real Robot utils ####
    ##########################

    def determine_terminate(self):
        command = input('Stop dataset collection: ')
        return True if 'yes' in command else False

    def execute(self, num_iter=10):
        for _ in range(num_iter):
            plan = self.plan()

            if plan is None:
                return None

            # Augment plan
            pick_metadata, place_metadata, insert_metadata = self.augment_plan(plan)

            # For metadata
            pick_cnt, place_cnt, insert_cnt = 0, 0, 0

            for i, (action_name, args) in enumerate(plan):
                # Post process TAMP commands to hsr executable actions
                action_name, object_name, modified_action = self.process(action_name, args)

                if action_name == 'move_base':
                    # Step1. Move base
                    for target_pose in modified_action:
                        target_base_pose = self.calculate_base_command(target_pose[:3])
                        base_traj = self.set_base_pose(target_base_pose)
                        self.base_pub.publish(base_traj)
                        self.rate.sleep()

                elif action_name == 'pick':
                    # Get target poses from pick_metadata
                    pick_pose = pick_metadata['target_robot_pose'][pick_cnt]
                    target_obj_pose = pick_metadata['target_object_pose'][pick_cnt]

                    pick_step = 0
                    for traj in modified_action:
                        for target_pose in traj:
                            target_base_pose = self.calculate_base_command(target_pose[:3])
                            target_arm_pose = self.calculate_arm_command(target_pose[3:])

                            # Get observation
                            obs = self.get_pick_observation(object_name, pick_pose, target_obj_pose)

                            # Residual action
                            with torch.no_grad():
                                actions = self.pick_agent.get_action(obs)

                            # Multiply target 6D pose and residual 6D pose
                            ik_action = self._dt * self.pick_action_scale * actions.to(self._device) # (dx, dy, dz)
                            self.ik_controller.set_command(ik_action)

                            # Calculate robot jacobian
                            ee_pos, ee_rot = self.mocap_interface.get_pose('end_effector')
                            joint_positions = np.array(self.hsr_interface.get_joint_positions())
                            robot_jacobian = self.hsr_ik_utils.get_jacobian(joint_positions)

                            # To tensor and device
                            ee_pos = torch.tensor(ee_pos, dtype=torch.float32, device=self._device).view(1, 3)
                            ee_rot = torch.tensor(ee_rot, dtype=torch.float32, device=self._device).view(1, 4)
                            ee_rot = ee_rot[:, [3, 0, 1, 2]]
                            robot_jacobian = torch.tensor(robot_jacobian, dtype=torch.float32, device=self._device).view(1, -1, 8)

                            # Calcurate delta pose
                            delta_pose = self.ik_controller.compute_delta(ee_pos, ee_rot, robot_jacobian)
                            delta_pose = torch.squeeze(delta_pose, dim=0)
                            delta_pose = delta_pose.to('cpu').detach().numpy().copy() # 8 dim

                            # Add delta pose to reference trajectory
                            target_base_pose += delta_pose[:3]
                            target_arm_pose += delta_pose[3:]

                            # Set target pose
                            base_traj = self.set_base_pose(target_base_pose)
                            arm_traj = self.set_arm_pose(target_arm_pose)

                            # Publish command
                            self.base_pub.publish(base_traj)
                            self.arm_pub.publish(arm_traj)
                            self.rate.sleep()

                            # Calculate RL status
                            rewards = self.calc_pick_reward(object_name, target_obj_pose[0], target_obj_pose[1])
                            terminals = self.check_pick_success(object_name)

                            self.pick_dataset['observations'].append(obs.to('cpu').detach().numpy().copy())
                            self.pick_dataset['actions'].append(actions.to('cpu').detach().numpy().copy())
                            self.pick_dataset['rewards'].append(rewards.to('cpu').detach().numpy().copy())
                            self.pick_dataset['terminals'].append(np.array(terminals))

                        # Close gripper
                        if pick_step == 0:
                            rospy.sleep(2.0)
                            self.hsr_interface.close_gripper()

                        # Determine whether skill continue or not
                        if self.check_pick_success(object_name): break

                        # Step one
                        pick_step += 1

                    if self.determine_terminate(): return

                    # Add pick count
                    pick_cnt += 1

                elif action_name == 'place':
                    # Get target poses from place_metadata
                    place_pose = place_metadata['target_robot_pose'][place_cnt]
                    target_obj_pose = place_metadata['target_object_pose'][place_cnt]

                    place_step = 0
                    for traj in modified_action:
                        for target_pose in traj:
                            target_base_pose = self.calculate_base_command(target_pose[:3])
                            target_arm_pose = self.calculate_arm_command(target_pose[3:])

                            # Get observation
                            obs = self.get_place_observation(object_name, place_pose, target_obj_pose)

                            # Residual action
                            with torch.no_grad():
                                actions = self.place_agent.get_action(obs)

                            # Multiply target 6D pose and residual 6D pose
                            ik_action = self._dt * self.place_action_scale * actions.to(self._device) # (dx, dy, dz)
                            self.ik_controller.set_command(ik_action)

                            # Calculate robot jacobian
                            ee_pos, ee_rot = self.mocap_interface.get_pose('end_effector')
                            joint_positions = np.array(self.hsr_interface.get_joint_positions())
                            robot_jacobian = self.hsr_ik_utils.get_jacobian(joint_positions)

                            # To tensor and device
                            ee_pos = torch.tensor(ee_pos, dtype=torch.float32, device=self._device).view(1, 3)
                            ee_rot = torch.tensor(ee_rot, dtype=torch.float32, device=self._device).view(1, 4)
                            ee_rot = ee_rot[:, [3, 0, 1, 2]]
                            robot_jacobian = torch.tensor(robot_jacobian, dtype=torch.float32, device=self._device).view(1, -1, 8)

                            # Calcurate delta pose
                            delta_pose = self.ik_controller.compute_delta(ee_pos, ee_rot, robot_jacobian)
                            delta_pose = torch.squeeze(delta_pose, dim=0)
                            delta_pose = delta_pose.to('cpu').detach().numpy().copy() # 8 dim

                            # Add delta pose to reference trajectory
                            target_base_pose += delta_pose[:3]
                            target_arm_pose += delta_pose[3:]

                            # Set target pose
                            base_traj = self.set_base_pose(target_base_pose)
                            arm_traj = self.set_arm_pose(target_arm_pose)

                            # Publish command
                            self.base_pub.publish(base_traj)
                            self.arm_pub.publish(arm_traj)
                            self.rate.sleep()

                            # Calculate RL status
                            rewards = self.calc_place_reward(object_name, target_obj_pose[0], target_obj_pose[1])
                            terminals = self.check_place_success(object_name, target_obj_pose[0])

                            self.place_dataset['observations'].append(obs.to('cpu').detach().numpy().copy())
                            self.place_dataset['actions'].append(actions.to('cpu').detach().numpy().copy())
                            self.place_dataset['rewards'].append(rewards.to('cpu').detach().numpy().copy())
                            self.place_dataset['terminals'].append(np.array(terminals))

                        # Determine whether skill continue or not
                        if self.check_place_success(object_name, target_obj_pose[0]): break

                        # Step one
                        place_step += 1

                    if self.determine_terminate(): return

                    # Add place count
                    place_cnt += 1

                elif action_name == 'insert':
                    # Get target poses from insert_metadata
                    insert_pose = insert_metadata['target_robot_pose'][insert_cnt]
                    target_obj_pose = insert_metadata['target_object_pose'][insert_cnt]

                    insert_step = 0
                    for traj in modified_action:
                        for target_pose in traj:
                            target_base_pose = self.calculate_base_command(target_pose[:3])
                            target_arm_pose = self.calculate_arm_command(target_pose[3:])

                            # Get observation
                            obs = self.get_insert_observation(object_name, insert_pose, target_obj_pose)

                            # Residual action
                            with torch.no_grad():
                                actions = self.insert_agent.get_action(obs)

                            # Multiply target 6D pose and residual 6D pose
                            ik_action = self._dt * self.insert_action_scale * actions.to(self._device) # (dx, dy, dz)
                            self.ik_controller.set_command(ik_action)

                            # Calculate robot jacobian
                            ee_pos, ee_rot = self.mocap_interface.get_pose('end_effector')
                            joint_positions = np.array(self.hsr_interface.get_joint_positions())
                            robot_jacobian = self.hsr_ik_utils.get_jacobian(joint_positions)

                            # To tensor and device
                            ee_pos = torch.tensor(ee_pos, dtype=torch.float32, device=self._device).view(1, 3)
                            ee_rot = torch.tensor(ee_rot, dtype=torch.float32, device=self._device).view(1, 4)
                            ee_rot = ee_rot[:, [3, 0, 1, 2]]
                            robot_jacobian = torch.tensor(robot_jacobian, dtype=torch.float32, device=self._device).view(1, -1, 8)

                            # Calcurate delta pose
                            delta_pose = self.ik_controller.compute_delta(ee_pos, ee_rot, robot_jacobian)
                            delta_pose = torch.squeeze(delta_pose, dim=0)
                            delta_pose = delta_pose.to('cpu').detach().numpy().copy() # 8 dim

                            # Add delta pose to reference trajectory
                            target_base_pose += delta_pose[:3]
                            target_arm_pose += delta_pose[3:]

                            # Set target pose
                            base_traj = self.set_base_pose(target_base_pose)
                            arm_traj = self.set_arm_pose(target_arm_pose)

                            # Publish command
                            self.base_pub.publish(base_traj)
                            self.arm_pub.publish(arm_traj)
                            self.rate.sleep()

                            # Calculate RL status
                            rewards = self.calc_insert_reward(object_name, target_obj_pose[0], target_obj_pose[1])
                            terminals = self.check_insert_success(object_name, target_obj_pose[0], target_obj_pose[1])

                            self.insert_dataset['observations'].append(obs.to('cpu').detach().numpy().copy())
                            self.insert_dataset['actions'].append(actions.to('cpu').detach().numpy().copy())
                            self.insert_dataset['rewards'].append(rewards.to('cpu').detach().numpy().copy())
                            self.insert_dataset['terminals'].append(np.array(terminals))

                        # Open gripper
                        if insert_step == 0:
                            rospy.sleep(2.0)
                            self.hsr_interface.open_gripper()

                        # Determine whether skill continue or not
                        if self.check_insert_success(object_name, target_obj_pose[0], target_obj_pose[1]): break

                        # Step one
                        insert_step += 1

                    if self.determine_terminate(): return

                    # Add insert count
                    insert_cnt += 1

            # Reset robot
            self.initialize_robot()

            # Wait for user in order to reset env
            input('Wait until user finish environment reset')

            # Reset TAMP
            self.initialize_tamp()


if __name__ == '__main__':
    exec_plan = CreateDataset()
    exec_plan.execute(num_iter=1)
    exec_plan.create_dataset()