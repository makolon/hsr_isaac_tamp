#!/usr/bin/env/python3
import os
import sys
import yaml
import h5py
import rospy
import torch
import argparse
import numpy as np
from scipy.interpolate import interp1d
from scipy.spatial.transform import Rotation as R

# TODO: fix
sys.path.append('/root/tamp-hsr/hsr_tamp/experiments/gearbox_3d/')
sys.path.append('..')
from execute_plan import ExecutePlan

def normalize(x, eps: float = 1e-9):
    norm = np.linalg.norm(x, ord=2, axis=-1, keepdims=True)
    return x / np.clip(norm, a_min=eps, a_max=None)

def quaternion_multiply(a, b):
    aw, ax, ay, az = np.split(a, 4, axis=-1)
    bw, bx, by, bz = np.split(b, 4, axis=-1)
    ow = aw * bw - ax * bx - ay * by - az * bz
    ox = aw * bx + ax * bw + ay * bz - az * by
    oy = aw * by - ax * bz + ay * bw + az * bx
    oz = aw * bz + ax * by - ay * bx + az * bw
    return np.concatenate((ow, ox, oy, oz), axis=-1)

def calc_diff_pos(p1, p2):
    return p1 - p2

def calc_diff_rot(q1, q2):
    q1 = normalize(q1)
    q2 = normalize(q2)

    scaling = np.array([1, -1, -1, -1])
    q1_inv = q1 * scaling
    q_diff = quaternion_multiply(q2, q1_inv)

    return q_diff


class CreateDataset(ExecutePlan):
    def __init__(self):
        super(CreateDataset, self).__init__()

        # Initialize dataset
        self.initialize_mlp_dataset()
        self.initialize_gp_dataset()

    def initialize_mlp_dataset(self):
        self.move_mlp_obs = []
        self.move_act_noise = []
        self.pick_mlp_obs = []
        self.pick_act_noise = []
        self.place_mlp_obs = []
        self.place_act_noise = []
        self.insert_mlp_obs = []
        self.insert_act_noise = []

    def initialize_gp_dataset(self):
        self.move_gp_obs = [] # (batch_size, max_length, input_dim)
        self.move_position_x = [] # (batch_size, max_length)
        self.move_position_y = [] # (batch_size, max_length)
        self.move_position_z = [] # (batch_size, max_length)
        self.move_rotation_x = [] # (batch_size, max_length)
        self.move_rotation_y = [] # (batch_size, max_length)
        self.move_rotation_z = [] # (batch_size, max_length)

        self.pick_gp_obs = [] # (batch_size, max_length, input_dim)
        self.pick_position_x = [] # (batch_size, max_length)
        self.pick_position_y = [] # (batch_size, max_length)
        self.pick_position_z = [] # (batch_size, max_length)
        self.pick_rotation_x = [] # (batch_size, max_length)
        self.pick_rotation_y = [] # (batch_size, max_length)
        self.pick_rotation_z = [] # (batch_size, max_length)

        self.place_gp_obs = [] # (batch_size, max_length, input_dim)
        self.place_position_x = [] # (batch_size, max_length)
        self.place_position_y = [] # (batch_size, max_length)
        self.place_position_z = [] # (batch_size, max_length)
        self.place_rotation_x = [] # (batch_size, max_length)
        self.place_rotation_y = [] # (batch_size, max_length)
        self.place_rotation_z = [] # (batch_size, max_length)

        self.insert_gp_obs = [] # (batch_size, max_length, input_dim)
        self.insert_position_x = [] # (batch_size, max_length)
        self.insert_position_y = [] # (batch_size, max_length)
        self.insert_position_z = [] # (batch_size, max_length)
        self.insert_rotation_x = [] # (batch_size, max_length)
        self.insert_rotation_y = [] # (batch_size, max_length)
        self.insert_rotation_z = [] # (batch_size, max_length)

    def create_mlp_dataset(self, num_trial=1):
        move_mlp_obs = np.array(self.move_mlp_obs)
        move_act_noise = np.array(self.move_act_noise)

        pick_mlp_obs = np.array(self.pick_mlp_obs)
        pick_act_noise = np.array(self.pick_act_noise)

        place_mlp_obs = np.array(self.place_mlp_obs)
        place_act_noise = np.array(self.place_act_noise)

        insert_mlp_obs = np.array(self.insert_mlp_obs)
        insert_act_noise = np.array(self.insert_act_noise)

        # Create dataset for multilayer perceptron
        move_file_name = 'move_mlp_dataset' + '_' + str(num_trial) + '.h5'
        move_file_path = os.path.join('./residual_dataset', move_file_name)
        with h5py.File(move_file_path, 'w') as f:
            f.create_dataset('data', data=move_mlp_obs)
            f.create_dataset('target', data=move_act_noise)

        pick_file_name = 'pick_mlp_dataset' + '_' + str(num_trial) + '.h5'
        pick_file_path = os.path.join('./residual_dataset', pick_file_name)
        with h5py.File(pick_file_path, 'w') as f:
            f.create_dataset('data', data=pick_mlp_obs)
            f.create_dataset('target', data=pick_act_noise)

        place_file_name = 'place_mlp_dataset' + '_' + str(num_trial) + '.h5'
        place_file_path = os.path.join('./residual_dataset', place_file_name)
        with h5py.File(place_file_path, 'w') as f:
            f.create_dataset('data', data=place_mlp_obs)
            f.create_dataset('target', data=place_act_noise)

        insert_file_name = 'insert_mlp_dataset' + '_' + str(num_trial) + '.h5'
        insert_file_path = os.path.join('./residual_dataset', insert_file_name)
        with h5py.File(insert_file_path, 'w') as f:
            f.create_dataset('data', data=insert_mlp_obs)
            f.create_dataset('target', data=insert_act_noise)

    def create_gp_dataset(self, file_names, input_dim=15, num_trial=1):
        self.move_gp_obs = np.array(self.move_gp_obs)
        move_gp_obs = self.move_gp_obs.reshape(-1, input_dim)
        move_position_x = np.array(self.move_position_x)
        move_position_y = np.array(self.move_position_y)
        move_position_z = np.array(self.move_position_z)
        move_rotation_x = np.array(self.move_rotation_x)
        move_rotation_y = np.array(self.move_rotation_y)
        move_rotation_z = np.array(self.move_rotation_z)
        move_act = [
            move_position_x, move_position_y, move_position_z,
            move_rotation_x, move_rotation_y, move_rotation_z
        ]

        self.pick_gp_obs = np.array(self.pick_gp_obs)
        pick_gp_obs = self.pick_gp_obs.reshape(-1, input_dim)
        pick_position_x = np.array(self.pick_position_x)
        pick_position_y = np.array(self.pick_position_y)
        pick_position_z = np.array(self.pick_position_z)
        pick_rotation_x = np.array(self.pick_rotation_x)
        pick_rotation_y = np.array(self.pick_rotation_y)
        pick_rotation_z = np.array(self.pick_rotation_z)
        pick_act = [
            pick_position_x, pick_position_y, pick_position_z,
            pick_rotation_x, pick_rotation_y, pick_rotation_z
        ]

        self.place_gp_obs = np.array(self.place_gp_obs)
        place_gp_obs = self.place_gp_obs.reshape(-1, input_dim)
        place_position_x = np.array(self.place_position_x)
        place_position_y = np.array(self.place_position_y)
        place_position_z = np.array(self.place_position_z)
        place_rotation_x = np.array(self.place_rotation_x)
        place_rotation_y = np.array(self.place_rotation_y)
        place_rotation_z = np.array(self.place_rotation_z)
        place_act = [
            place_position_x, place_position_y, place_position_z,
            place_rotation_x, place_rotation_y, place_rotation_z
        ]

        self.insert_gp_obs = np.array(self.insert_gp_obs)
        insert_gp_obs = self.insert_gp_obs.reshape(-1, input_dim)
        insert_position_x = np.array(self.insert_position_x)
        insert_position_y = np.array(self.insert_position_y)
        insert_position_z = np.array(self.insert_position_z)
        insert_rotation_x = np.array(self.insert_rotation_x)
        insert_rotation_y = np.array(self.insert_rotation_y)
        insert_rotation_z = np.array(self.insert_rotation_z)
        insert_act = [
            insert_position_x, insert_position_y, insert_position_z,
            insert_rotation_x, insert_rotation_y, insert_rotation_z
        ]

        # Create dataset for gaussian process
        for idx, file_name in enumerate(file_names):
            move_file_name = 'move_dataset' + '_' + file_name + '_' + str(num_trial) + '.h5'
            print('move_obs:', move_gp_obs.shape)
            print('move_act:', move_act[idx].shape)
            move_file_path = os.path.join('./residual_dataset', move_file_name)
            with h5py.File(move_file_path, 'w') as f:
                f.create_dataset('train_x', data=move_gp_obs)
                f.create_dataset('train_y', data=move_act[idx])

            pick_file_name = 'pick_dataset' + '_' + file_name + '_' + str(num_trial) + '.h5'
            print('pick_obs:', pick_gp_obs.shape)
            print('pick_act:', pick_act[idx].shape)
            pick_file_path = os.path.join('./residual_dataset', pick_file_name)
            with h5py.File(pick_file_path, 'w') as f:
                f.create_dataset('train_x', data=pick_gp_obs)
                f.create_dataset('train_y', data=pick_act[idx])
            
            place_file_name = 'place_dataset' + '_' + file_name + '_' + str(num_trial) + '.h5'
            print('place_obs:', place_gp_obs.shape)
            print('place_act:', place_act[idx].shape)
            place_file_path = os.path.join('./residual_dataset', place_file_name)
            with h5py.File(place_file_path, 'w') as f:
                f.create_dataset('train_x', data=place_gp_obs)
                f.create_dataset('train_y', data=place_act[idx])

            insert_file_name = 'insert_dataset' + '_' + file_name + '_' + str(num_trial) + '.h5'
            print('insert_obs:', insert_gp_obs.shape)
            print('insert_act:', insert_act[idx].shape)
            insert_file_path = os.path.join('./residual_dataset', insert_file_name)
            with h5py.File(insert_file_path, 'w') as f:
                f.create_dataset('train_x', data=insert_gp_obs)
                f.create_dataset('train_y', data=insert_act[idx])

    def get_observations(self):
        # Ground truth
        gt_base_pos, gt_base_quat = self.mocap_interface.get_pose('hsr_base') # 3, 4 dim
        gt_base_xy = gt_base_pos[:2] # 2 dim
        gt_base_yaw = R.from_quat(gt_base_quat).as_euler('xyz') # 1 dim
        gt_base_pose = np.array([*gt_base_xy, gt_base_yaw[1]], dtype=np.float32)

        gt_ee_pos, gt_ee_quat = self.mocap_interface.get_pose('end_effector') # 3, 4 dim
        gt_ee_pos = np.array(gt_ee_pos, dtype=np.float32)
        gt_ee_quat = np.array(gt_ee_quat, dtype=np.float32)

        gt_arm_pose = self.hsr_interface.get_joint_positions(group='arm')

        # Measured values
        mv_ee_pos, mv_ee_quat = self.hsr_interface.get_link_pose('hand_palm_link')
        mv_ee_pos = np.array(mv_ee_pos, dtype=np.float32)
        mv_ee_quat = -np.array(mv_ee_quat, dtype=np.float32)

        # Calculate difference
        diff_ee_pos = mv_ee_pos - gt_ee_pos
        diff_ee_rot = calc_diff_rot(gt_ee_quat, mv_ee_quat)
        diff_ee_rot = R.from_quat(diff_ee_rot[[3, 0, 1, 2]]).as_euler('xyz')
        for idx, diff_rot in enumerate(diff_ee_rot):
            if diff_rot > np.pi/2:
                diff_ee_rot[idx] = np.pi - diff_rot
            elif diff_rot < -np.pi/2:
                diff_ee_rot[idx] = -np.pi - diff_rot

        input_data = np.concatenate((gt_base_pose, gt_arm_pose, gt_ee_pos, gt_ee_quat)) # 15 dim
        output_data = np.concatenate((diff_ee_pos, diff_ee_rot)) # 6 dim

        return input_data, output_data
    
    def calculate_action(self, target_pose, group='all'):
        if group == 'all':
            # Action is defined as target_pose - current_gt_pose
            curr_base_pos, curr_base_quat = self.mocap_interface.get_pose('hsr_base') # Ground truth
            curr_base_euler = R.from_quat(curr_base_quat).as_euler('xyz')
            curr_base_pose = np.array([curr_base_pos[0], curr_base_pos[1], curr_base_euler[1]], dtype=np.float32)
            diff_base_pose = np.array(target_pose[:3], dtype=np.float32) - np.array(curr_base_pose, dtype=np.float32)

            curr_arm_pose = self.hsr_interface.get_joint_positions(group='arm') # NOTE: not ground truth but acceptable
            diff_arm_pose = np.array(target_pose[3:], dtype=np.float32) - np.array(curr_arm_pose, dtype=np.float32)

            diff_pose = np.concatenate((diff_base_pose, diff_arm_pose))
        elif group == 'base':
            curr_base_pos, curr_base_quat = self.mocap_interface.get_pose('hsr_base')
            curr_base_euler = R.from_quat(curr_base_quat).as_euler('xyz')
            curr_base_pose = np.array([curr_base_pos[0], curr_base_pos[1], curr_base_euler[1]], dtype=np.float32)
            diff_base_pose = np.array(target_pose[:3], dtype=np.float32) - np.array(curr_base_pose, dtype=np.float32)

            diff_arm_pose = np.zeros(5)
            diff_pose = np.concatenate((diff_base_pose, diff_arm_pose))
        elif group == 'arm':
            curr_arm_pose = self.hsr_interface.get_joint_positions(group='arm')
            diff_arm_pose = np.array(target_pose[3:], dtype=np.float32) - np.array(curr_arm_pose, dtype=np.float32)

            diff_base_pose = np.zeros(5)
            diff_pose = np.concatenate((diff_base_pose, diff_arm_pose))
        return diff_pose

    ##########################
    #### Real Robot utils ####
    ##########################

    def execute(self):
        plan = self.plan()

        if plan is None:
            return None

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

                    action = self.calculate_action(target_pose, group='base')
                    move_input_data, move_output_data = self.get_observations()

                    # Append MLP dataset
                    self.move_mlp_obs.append(move_input_data)
                    self.move_act_noise.append(action)

                    # Append GP dataset
                    self.move_gp_obs.append(move_input_data)
                    self.move_position_x.append(move_output_data[0])
                    self.move_position_y.append(move_output_data[1])
                    self.move_position_z.append(move_output_data[2])
                    self.move_rotation_x.append(move_output_data[3])
                    self.move_rotation_y.append(move_output_data[4])
                    self.move_rotation_z.append(move_output_data[5])

            elif action_name == 'pick':
                pick_traj, return_traj = modified_action
                for target_pose in pick_traj: # pick
                    target_base_pose = self.calculate_base_command(target_pose[:3])
                    target_arm_pose = self.calculate_arm_command(target_pose[3:])
                    base_traj = self.set_base_pose(target_base_pose)
                    arm_traj = self.set_arm_pose(target_arm_pose)
                    self.base_pub.publish(base_traj)
                    self.arm_pub.publish(arm_traj)
                    self.rate.sleep()

                    action = self.calculate_action(target_pose)
                    pick_input_data, pick_output_data = self.get_observations()

                    # Append MLP dataset
                    self.pick_mlp_obs.append(pick_input_data)
                    self.pick_act_noise.append(action)

                    # Append GP dataset
                    self.pick_gp_obs.append(pick_input_data)
                    self.pick_position_x.append(pick_output_data[0])
                    self.pick_position_y.append(pick_output_data[1])
                    self.pick_position_z.append(pick_output_data[2])
                    self.pick_rotation_x.append(pick_output_data[3])
                    self.pick_rotation_y.append(pick_output_data[4])
                    self.pick_rotation_z.append(pick_output_data[5])

                rospy.sleep(3.0)

                for target_pose in return_traj: # return
                    target_base_pose = self.calculate_base_command(target_pose[:3])
                    target_arm_pose = self.calculate_arm_command(target_pose[3:])
                    base_traj = self.set_base_pose(target_base_pose)
                    arm_traj = self.set_arm_pose(target_arm_pose)
                    self.base_pub.publish(base_traj)
                    self.arm_pub.publish(arm_traj)
                    self.rate.sleep()

                    action = self.calculate_action(target_pose)
                    pick_input_data, pick_output_data = self.get_observations()

                    # Append MLP dataset
                    self.pick_mlp_obs.append(pick_input_data)
                    self.pick_act_noise.append(action)

                    # Append GP dataset
                    self.pick_gp_obs.append(pick_input_data)
                    self.pick_position_x.append(pick_output_data[0])
                    self.pick_position_y.append(pick_output_data[1])
                    self.pick_position_z.append(pick_output_data[2])
                    self.pick_rotation_x.append(pick_output_data[3])
                    self.pick_rotation_y.append(pick_output_data[4])
                    self.pick_rotation_z.append(pick_output_data[5])

            elif action_name == 'place':
                place_traj = modified_action
                for target_pose in place_traj: # place
                    target_base_pose = self.calculate_base_command(target_pose[:3])
                    target_arm_pose = self.calculate_arm_command(target_pose[3:])
                    base_traj = self.set_base_pose(target_base_pose)
                    arm_traj = self.set_arm_pose(target_arm_pose)
                    self.base_pub.publish(base_traj)
                    self.arm_pub.publish(arm_traj)
                    self.rate.sleep()

                    action = self.calculate_action(target_pose)
                    place_input_data, place_output_data = self.get_observations()

                    # Append MLP dataset
                    self.place_mlp_obs.append(place_input_data)
                    self.place_act_noise.append(action)

                    # Append GP dataset
                    self.place_gp_obs.append(place_input_data)
                    self.place_position_x.append(place_output_data[0])
                    self.place_position_y.append(place_output_data[1])
                    self.place_position_z.append(place_output_data[2])
                    self.place_rotation_x.append(place_output_data[3])
                    self.place_rotation_y.append(place_output_data[4])
                    self.place_rotation_z.append(place_output_data[5])

            elif action_name == 'insert':
                insert_traj, depart_traj, return_traj = modified_action
                for target_pose in insert_traj: # insert
                    target_base_pose = self.calculate_base_command(target_pose[:3])
                    target_arm_pose = self.calculate_arm_command(target_pose[3:])
                    base_traj = self.set_base_pose(target_base_pose)
                    arm_traj = self.set_arm_pose(target_arm_pose)
                    self.base_pub.publish(base_traj)
                    self.arm_pub.publish(arm_traj)
                    self.rate.sleep()

                    action = self.calculate_action(target_pose)
                    insert_input_data, insert_output_data = self.get_observations()

                    # Append MLP dataset
                    self.insert_mlp_obs.append(insert_input_data)
                    self.insert_act_noise.append(action)

                    # Append GP dataset
                    self.insert_gp_obs.append(insert_input_data)
                    self.insert_position_x.append(insert_output_data[0])
                    self.insert_position_y.append(insert_output_data[1])
                    self.insert_position_z.append(insert_output_data[2])
                    self.insert_rotation_x.append(insert_output_data[3])
                    self.insert_rotation_y.append(insert_output_data[4])
                    self.insert_rotation_z.append(insert_output_data[5])

                rospy.sleep(3.0)

                for target_pose in depart_traj: # depart
                    target_base_pose = self.calculate_base_command(target_pose[:3])
                    target_arm_pose = self.calculate_arm_command(target_pose[3:])
                    base_traj = self.set_base_pose(target_base_pose)
                    arm_traj = self.set_arm_pose(target_arm_pose)
                    self.base_pub.publish(base_traj)
                    self.arm_pub.publish(arm_traj)
                    self.rate.sleep()

                    action = self.calculate_action(target_pose)
                    insert_input_data, insert_output_data = self.get_observations()

                    # Append MLP dataset
                    self.insert_mlp_obs.append(insert_input_data)
                    self.insert_act_noise.append(action)

                    # Append GP dataset
                    self.insert_gp_obs.append(insert_input_data)
                    self.insert_position_x.append(insert_output_data[0])
                    self.insert_position_y.append(insert_output_data[1])
                    self.insert_position_z.append(insert_output_data[2])
                    self.insert_rotation_x.append(insert_output_data[3])
                    self.insert_rotation_y.append(insert_output_data[4])
                    self.insert_rotation_z.append(insert_output_data[5])

                for target_pose in return_traj: # return
                    target_base_pose = self.calculate_base_command(target_pose[:3])
                    target_arm_pose = self.calculate_arm_command(target_pose[3:])
                    base_traj = self.set_base_pose(target_base_pose)
                    arm_traj = self.set_arm_pose(target_arm_pose)
                    self.base_pub.publish(base_traj)
                    self.arm_pub.publish(arm_traj)
                    self.rate.sleep()

                    action = self.calculate_action(target_pose)
                    insert_input_data, insert_output_data = self.get_observations()

                    # Append MLP dataset
                    self.insert_mlp_obs.append(insert_input_data)
                    self.insert_act_noise.append(action)

                    # Append GP dataset
                    self.insert_gp_obs.append(insert_input_data)
                    self.insert_position_x.append(insert_output_data[0])
                    self.insert_position_y.append(insert_output_data[1])
                    self.insert_position_z.append(insert_output_data[2])
                    self.insert_rotation_x.append(insert_output_data[3])
                    self.insert_rotation_y.append(insert_output_data[4])
                    self.insert_rotation_z.append(insert_output_data[5])


if __name__ == '__main__':
    file_names = ['position_x', 'position_y', 'position_z', 'rotation_x', 'rotation_y', 'rotation_z']
    exec_plan = CreateDataset()
    exec_plan.execute()
    exec_plan.create_mlp_dataset(num_trial=1)
    exec_plan.create_gp_dataset(file_names, num_trial=1)
