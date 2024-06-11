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
from execute_plan import ExecutePlan, modify_target_pose

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
        # Observation (input data for mlp)
        self.move_mlp_obs = []
        self.pick_mlp_obs = []
        self.place_mlp_obs = []
        self.insert_mlp_obs = []

        # Action noise (output data for mlp)
        self.move_act_noise = []
        self.pick_act_noise = []
        self.place_act_noise = []
        self.insert_act_noise = []

    def initialize_gp_dataset(self):
        # Observation (input data for gp)
        self.gp_obs = []

        # Ground truth (output data for gp)
        self.position_x = []
        self.position_y = []
        self.position_z = []
        self.rotation_x = []
        self.rotation_y = []
        self.rotation_z = []

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
        print('move_mlp_obs:', move_mlp_obs.shape)
        print('move_act_noise:', move_act_noise.shape)
        with h5py.File(move_file_path, 'w') as f:
            f.create_dataset('data', data=move_mlp_obs)
            f.create_dataset('target', data=move_act_noise)

        pick_file_name = 'pick_mlp_dataset' + '_' + str(num_trial) + '.h5'
        pick_file_path = os.path.join('./residual_dataset', pick_file_name)
        print('pick_mlp_obs:', pick_mlp_obs.shape)
        print('pick_act_noise:', pick_act_noise.shape)
        with h5py.File(pick_file_path, 'w') as f:
            f.create_dataset('data', data=pick_mlp_obs)
            f.create_dataset('target', data=pick_act_noise)

        place_file_name = 'place_mlp_dataset' + '_' + str(num_trial) + '.h5'
        place_file_path = os.path.join('./residual_dataset', place_file_name)
        print('place_mlp_obs:', place_mlp_obs.shape)
        print('place_act_noise:', place_act_noise.shape)
        with h5py.File(place_file_path, 'w') as f:
            f.create_dataset('data', data=place_mlp_obs)
            f.create_dataset('target', data=place_act_noise)

        insert_file_name = 'insert_mlp_dataset' + '_' + str(num_trial) + '.h5'
        insert_file_path = os.path.join('./residual_dataset', insert_file_name)
        print('insert_mlp_obs:', insert_mlp_obs.shape)
        print('insert_act_noise:', insert_act_noise.shape)
        with h5py.File(insert_file_path, 'w') as f:
            f.create_dataset('data', data=insert_mlp_obs)
            f.create_dataset('target', data=insert_act_noise)

    def create_gp_dataset(self, file_names, input_dim=15):
        self.gp_obs = np.array(self.gp_obs)
        gp_obs = self.gp_obs.reshape(-1, input_dim)

        gp_gt = [
            np.array(self.position_x), np.array(self.position_y), np.array(self.position_z),
            np.array(self.rotation_x), np.array(self.rotation_y), np.array(self.rotation_z)
        ]

        # Create dataset for gaussian process
        for idx, file_name in enumerate(file_names):
            file_name = 'gp_dataset' + '_' + file_name + '.h5'
            print('gp_obs:', gp_obs.shape)
            print('gp_gt:', gp_gt[idx].shape)
            file_path = os.path.join('./residual_dataset', file_name)
            with h5py.File(file_path, 'w') as f:
                f.create_dataset('train_x', data=gp_obs)
                f.create_dataset('train_y', data=gp_gt[idx])

    def get_gp_observations(self):
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

    def get_mlp_observations(self, delta_pose):
        # Ground truth
        gt_base_pos, gt_base_quat = self.mocap_interface.get_pose('hsr_base') # 3, 4 dim
        gt_base_xy = gt_base_pos[:2] # 2 dim
        gt_base_yaw = R.from_quat(gt_base_quat).as_euler('xyz') # 1 dim
        gt_base_pose = np.array([*gt_base_xy, gt_base_yaw[1]], dtype=np.float32)

        gt_ee_pos, gt_ee_quat = self.mocap_interface.get_pose('end_effector') # 3, 4 dim
        gt_ee_pos = np.array(gt_ee_pos, dtype=np.float32)
        gt_ee_quat = np.array(gt_ee_quat, dtype=np.float32)

        gt_arm_pose = self.hsr_interface.get_joint_positions(group='arm')

        input_data = np.concatenate((gt_base_pose, gt_arm_pose, gt_ee_pos, gt_ee_quat, delta_pose)) # 23 dim

        return input_data

    def calculate_action_noise(self, target_pose, group='all'):
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

        # Augment plan
        move_metadata, pick_metadata, place_metadata, insert_metadata = self.augment_plan(plan)

        # For metadata
        move_cnt, pick_cnt, place_cnt, insert_cnt = 0, 0, 0, 0

        for i, (action_name, args) in enumerate(plan):
            # Post process TAMP commands to hsr executable actions
            action_name, object_name, modified_action = self.process(action_name, args)

            if action_name == 'move_base':
                for target_pose in modified_action: # move
                    ### Save gp state estimation noise data

                    target_base_pose = self.calculate_base_command(target_pose[:3])
                    base_traj = self.set_base_pose(target_base_pose)
                    self.base_pub.publish(base_traj)
                    self.rate.sleep()

                    # Get observation data for MLP & GP
                    move_gp_obs, move_gp_noise = self.get_gp_observations()

                    # Append GP dataset
                    self.gp_obs.append(move_gp_obs)
                    self.position_x.append(move_gp_noise[0])
                    self.position_y.append(move_gp_noise[1])
                    self.position_z.append(move_gp_noise[2])
                    self.rotation_x.append(move_gp_noise[3])
                    self.rotation_z.append(move_gp_noise[4])
                    self.rotation_y.append(move_gp_noise[5])

            elif action_name == 'pick':
                pick_traj, return_traj = modified_action

                # Get target poses from pick_metadata
                pick_pose = pick_metadata['target_robot_pose'][pick_cnt]
                target_obj_pose = pick_metadata['target_object_pose'][pick_cnt]
                for target_pose in pick_traj: # pick
                    ### Save action noise data & gp state estimation noise data

                    # Get observation
                    obs = self.get_pick_observation(object_name, pick_pose, target_obj_pose)

                    # Residual action
                    with torch.no_grad():
                        actions = self.pick_agent.get_action(obs)

                    # Multiply target 6D pose and residual 6D pose
                    jt_action = self._dt * self.pick_action_scale * actions.to(self._device)
                    delta_pose = torch.squeeze(jt_action, dim=0)
                    delta_pose = delta_pose.to('cpu').detach().numpy().copy() # 8 dim

                    # P control
                    target_base_pose = self.calculate_base_command(target_pose[:3]+delta_pose[:3])
                    target_arm_pose = self.calculate_arm_command(target_pose[3:]+delta_pose[3:])

                    # Set target pose
                    base_traj = self.set_base_pose(target_base_pose)
                    arm_traj = self.set_arm_pose(target_arm_pose)

                    # Publish command
                    self.base_pub.publish(base_traj)
                    self.arm_pub.publish(arm_traj)
                    self.rate.sleep()

                    # Get observation data for MLP & GP
                    pick_mlp_obs = self.get_mlp_observations(delta_pose)
                    pick_gp_obs, pick_gp_noise = self.get_gp_observations()

                    # Get action data for MLP
                    action_noise = self.calculate_action_noise(target_pose+delta_pose)

                    # Append MLP dataset
                    self.pick_mlp_obs.append(pick_mlp_obs)
                    self.pick_act_noise.append(action_noise)

                    # Append GP dataset
                    self.gp_obs.append(pick_gp_obs)
                    self.position_x.append(pick_gp_noise[0])
                    self.position_y.append(pick_gp_noise[1])
                    self.position_z.append(pick_gp_noise[2])
                    self.rotation_x.append(pick_gp_noise[3])
                    self.rotation_z.append(pick_gp_noise[4])
                    self.rotation_y.append(pick_gp_noise[5])

                rospy.sleep(3.0)
                self.hsr_interface.close_gripper()

                for target_pose in return_traj: # return
                    ### Save gp state estimation noise data

                    target_base_pose = self.calculate_base_command(target_pose[:3])
                    target_arm_pose = self.calculate_arm_command(target_pose[3:])
                    base_traj = self.set_base_pose(target_base_pose)
                    arm_traj = self.set_arm_pose(target_arm_pose)
                    self.base_pub.publish(base_traj)
                    self.arm_pub.publish(arm_traj)
                    self.rate.sleep()

                    # Get observation data for MLP & GP
                    pick_gp_obs, pick_gp_noise = self.get_gp_observations()

                    # Append GP dataset
                    self.gp_obs.append(pick_gp_obs)
                    self.position_x.append(pick_gp_noise[0])
                    self.position_y.append(pick_gp_noise[1])
                    self.position_z.append(pick_gp_noise[2])
                    self.rotation_x.append(pick_gp_noise[3])
                    self.rotation_z.append(pick_gp_noise[4])
                    self.rotation_y.append(pick_gp_noise[5])

            elif action_name == 'place':
                place_traj = modified_action

                # Get target poses from place_metadata
                place_pose = place_metadata['target_robot_pose'][place_cnt]
                target_obj_pose = place_metadata['target_object_pose'][place_cnt]

                gearbox_base_pose = self.mocap_interface.get_pose('base')
                target_obj_pose = modify_target_pose(target_obj_pose, gearbox_base_pose, object_name)
                for target_pose in place_traj: # place
                    ### Save action noise data & gp state estimation noise data

                    # Get observation
                    obs = self.get_place_observation(object_name, place_pose, target_obj_pose)

                    # Residual action
                    with torch.no_grad():
                        actions = self.place_agent.get_action(obs)

                    # Multiply target 6D pose and residual 6D pose
                    jt_action = self._dt * self.place_action_scale * actions.to(self._device)
                    delta_pose = torch.squeeze(jt_action, dim=0)
                    delta_pose = delta_pose.to('cpu').detach().numpy().copy() # 8 dim

                    # P control
                    target_base_pose = self.calculate_base_command(target_pose[:3]+delta_pose[:3])
                    target_arm_pose = self.calculate_arm_command(target_pose[3:]+delta_pose[3:])

                    # Set target pose
                    base_traj = self.set_base_pose(target_base_pose)
                    arm_traj = self.set_arm_pose(target_arm_pose)

                    # Publish command
                    self.base_pub.publish(base_traj)
                    self.arm_pub.publish(arm_traj)
                    self.rate.sleep()

                    # Get observation data for MLP & GP
                    place_mlp_obs = self.get_mlp_observations(delta_pose)
                    place_gp_obs, place_gp_noise = self.get_gp_observations()

                    # Get action data for MLP
                    action_noise = self.calculate_action_noise(target_pose+delta_pose)

                    # Append MLP dataset
                    self.place_mlp_obs.append(place_mlp_obs)
                    self.place_act_noise.append(action_noise)

                    # Append GP dataset
                    self.gp_obs.append(place_gp_obs)
                    self.position_x.append(place_gp_noise[0])
                    self.position_y.append(place_gp_noise[1])
                    self.position_z.append(place_gp_noise[2])
                    self.rotation_x.append(place_gp_noise[3])
                    self.rotation_z.append(place_gp_noise[4])
                    self.rotation_y.append(place_gp_noise[5])

            elif action_name == 'insert':
                insert_traj, depart_traj, return_traj = modified_action

                # Get target poses from insert_metadata
                insert_pose = insert_metadata['target_robot_pose'][insert_cnt]
                target_obj_pose = insert_metadata['target_object_pose'][insert_cnt]

                gearbox_base_pose = self.mocap_interface.get_pose('base')
                target_obj_pose = modify_target_pose(target_obj_pose, gearbox_base_pose, object_name)

                for target_pose in insert_traj: # insert
                    ### Save action noise data & gp state estimation noise data

                    # Get observation
                    obs = self.get_insert_observation(object_name, insert_pose, target_obj_pose)

                    # Residual action
                    with torch.no_grad():
                        actions = self.insert_agent.get_action(obs)

                    # Multiply target 6D pose and residual 6D pose
                    jt_action = self._dt * self.insert_action_scale * actions.to(self._device)
                    delta_pose = torch.squeeze(jt_action, dim=0)
                    delta_pose = delta_pose.to('cpu').detach().numpy().copy() # 8 dim

                    # P control
                    target_base_pose = self.calculate_base_command(target_pose[:3]+delta_pose[:3])
                    target_arm_pose = self.calculate_arm_command(target_pose[3:]+delta_pose[3:])

                    # Set target pose
                    base_traj = self.set_base_pose(target_base_pose)
                    arm_traj = self.set_arm_pose(target_arm_pose)

                    # Publish command
                    self.base_pub.publish(base_traj)
                    self.arm_pub.publish(arm_traj)
                    self.rate.sleep()

                    # Get observation data for MLP & GP
                    insert_mlp_obs = self.get_mlp_observations(delta_pose)
                    insert_gp_obs, insert_gp_noise = self.get_gp_observations()

                    # Get action data for MLP
                    action_noise = self.calculate_action_noise(target_pose+delta_pose)

                    # Append MLP dataset
                    self.insert_mlp_obs.append(insert_mlp_obs)
                    self.insert_act_noise.append(action_noise)

                    # Append GP dataset
                    self.gp_obs.append(insert_gp_obs)
                    self.position_x.append(insert_gp_noise[0])
                    self.position_y.append(insert_gp_noise[1])
                    self.position_z.append(insert_gp_noise[2])
                    self.rotation_x.append(insert_gp_noise[3])
                    self.rotation_z.append(insert_gp_noise[4])
                    self.rotation_y.append(insert_gp_noise[5])

                rospy.sleep(3.0)
                self.hsr_interface.open_gripper()

                for target_pose in depart_traj: # depart
                    ### Save gp state estimation noise data

                    target_base_pose = self.calculate_base_command(target_pose[:3])
                    target_arm_pose = self.calculate_arm_command(target_pose[3:])
                    base_traj = self.set_base_pose(target_base_pose)
                    arm_traj = self.set_arm_pose(target_arm_pose)
                    self.base_pub.publish(base_traj)
                    self.arm_pub.publish(arm_traj)
                    self.rate.sleep()

                    # Get observation data for MLP & GP
                    insert_gp_obs, insert_gp_noise = self.get_gp_observations()

                    # Append GP dataset
                    self.gp_obs.append(insert_gp_obs)
                    self.position_x.append(insert_gp_noise[0])
                    self.position_y.append(insert_gp_noise[1])
                    self.position_z.append(insert_gp_noise[2])
                    self.rotation_x.append(insert_gp_noise[3])
                    self.rotation_z.append(insert_gp_noise[4])
                    self.rotation_y.append(insert_gp_noise[5])

                for target_pose in return_traj: # return
                    ### Save gp state estimation noise data

                    target_base_pose = self.calculate_base_command(target_pose[:3])
                    target_arm_pose = self.calculate_arm_command(target_pose[3:])
                    base_traj = self.set_base_pose(target_base_pose)
                    arm_traj = self.set_arm_pose(target_arm_pose)
                    self.base_pub.publish(base_traj)
                    self.arm_pub.publish(arm_traj)
                    self.rate.sleep()

                    # Get insert_gp_obs data for MLP & GP
                    insert_gp_obs, insert_gp_noise = self.get_gp_observations()

                    # Append GP dataset
                    self.gp_obs.append(insert_gp_obs)
                    self.position_x.append(insert_gp_noise[0])
                    self.position_y.append(insert_gp_noise[1])
                    self.position_z.append(insert_gp_noise[2])
                    self.rotation_x.append(insert_gp_noise[3])
                    self.rotation_z.append(insert_gp_noise[4])
                    self.rotation_y.append(insert_gp_noise[5])


if __name__ == '__main__':
    file_names = ['position_x', 'position_y', 'position_z', 'rotation_x', 'rotation_y', 'rotation_z']

    exec_plan = CreateDataset()
    exec_plan.execute()
    exec_plan.create_mlp_dataset(num_trial=3)
    exec_plan.create_gp_dataset(file_names)
