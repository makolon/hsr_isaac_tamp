import os
import json
import math
import numpy as np
from scipy.spatial.transform import Rotation as R


def load_dataset(problem_name, action_type='absolute', target_space='joint'):
    dataset_dir = os.path.join('/root/tamp-hsr/hsr_tamp/experiments/gearbox_3d/dataset', problem_name, 'full')

    # Trajectory and metadata directory per skill
    traj_dir = os.path.join(dataset_dir, 'trajectory')
    meta_dir = os.path.join(dataset_dir, 'metadata')

    # Calculate number of files
    num_traj_files = sum(os.path.isfile(os.path.join(traj_dir, name)) for name in os.listdir(traj_dir))
    num_meta_files = sum(os.path.isfile(os.path.join(meta_dir, name)) for name in os.listdir(meta_dir))

    # Get trajectory
    dataset_list = []
    for i in range(num_traj_files):
        data_name = 'trajectory_' + str(i) + '.json'
        dataset_path = os.path.join(traj_dir, data_name)
        with open(dataset_path, 'rb') as f:
            dataset = json.load(f)
        dataset_list.append(dataset)

    # Get metadata
    metadata_list = []
    for i in range(num_meta_files):
        metadata_name = 'metadata_' + str(i) + '.json'
        metadata_path = os.path.join(meta_dir, metadata_name)
        with open(metadata_path, 'rb') as f:
            metadata = json.load(f)
        metadata_list.append(metadata)

    action_dataset, _ = post_process(dataset_list, action_type, target_space)
    metadata = parse_metadata(metadata_list)

    return action_dataset, metadata

def load_skill_dataset(problem_name, skill_name, action_type='absolute', target_space='joint'):
    dataset_dir = os.path.join('/root/tamp-hsr/hsr_tamp/experiments/gearbox_3d/dataset', problem_name, skill_name)

    # Trajectory and metadata directory per skill
    traj_dir = os.path.join(dataset_dir, 'trajectory')
    meta_dir = os.path.join(dataset_dir, 'metadata')

    # Calculate number of files
    num_traj_files = sum(os.path.isfile(os.path.join(traj_dir, name)) for name in os.listdir(traj_dir))
    num_meta_files = sum(os.path.isfile(os.path.join(meta_dir, name)) for name in os.listdir(meta_dir))

    # Get trajectory
    dataset_list = []
    for i in range(num_traj_files):
        data_name = 'trajectory_' + str(i) + '.json'
        dataset_path = os.path.join(traj_dir, data_name)
        with open(dataset_path, 'rb') as f:
            dataset = json.load(f)
        dataset_list.append(dataset)

    # Get metadata
    metadata_list = []
    for i in range(num_meta_files):
        metadata_name = 'metadata_' + str(i) + '.json'
        metadata_path = os.path.join(meta_dir, metadata_name)
        with open(metadata_path, 'rb') as f:
            metadata = json.load(f)
        metadata_list.append(metadata)

    action_dataset, _ = post_skill_process(dataset_list, action_type, target_space)
    metadata = parse_skill_metadata(metadata_list)

    return action_dataset, metadata

def post_process(dataset, action_type='absolute', target_space='joint'):
    action_dataset = []
    object_dataset = []
    for data in dataset:
        # Get data for robot
        robot_poses = data['robot_pose'] # 10 joints
        gripper_poses = data['gripper_pose'] # 2 joints
        ee_poses = data['ee_pose'] # 7 dimensions
        diff_robot_poses = data['diff_robot_pose'] # 10 joints
        diff_gripper_poses = data['diff_gripper_pose'] # 2 joints
        diff_ee_poses = data['diff_ee_pose'] # 7 dimensions

        # Prepare action_dataset
        robot_traj = []
        if target_space == 'joint':
            if action_type == 'relative':
                for drp in diff_robot_poses:
                    robot_traj.append(drp) # 10 dimensions
            elif action_type == 'absolute':
                for rp in robot_poses:
                    robot_traj.append(rp) # 10 dimensions

        elif target_space == 'task':
            if action_type == 'relative':
                for dep, dgp in zip(diff_ee_poses, diff_gripper_poses):
                    action = dep + dgp
                    robot_traj.append(action) # 8 dimensions
            elif action_type == 'absolute':
                for ep, gp in zip(ee_poses, gripper_poses):
                    action = ep[0] + transpose(ep[1]) + gp
                    robot_traj.append(action) # 9 dimensions

        # Get data for object
        object_poses = data['object_pose']

        # Prepare object_dataset
        object_traj = []
        for op in object_poses:
            object_traj.append(op)

        action_dataset.append(robot_traj)
        object_dataset.append(object_traj)

    return action_dataset, object_dataset

def post_skill_process(dataset, action_type='absolute', target_space='joint'):
    action_dataset = []
    object_dataset = []
    for action_name, data in dataset:
        # Get data for robot
        robot_poses = data['robot_pose'] # 10 joints
        gripper_poses = data['gripper_pose'] # 2 joints
        ee_poses = data['ee_pose'] # 7 dimensions
        diff_robot_poses = data['diff_robot_pose'] # 10 joints
        diff_gripper_poses = data['diff_gripper_pose'] # 2 joints
        diff_ee_poses = data['diff_ee_pose'] # 7 dimensions

        # Prepare action_dataset
        robot_traj = []
        if target_space == 'joint':
            if action_type == 'relative':
                for drp in diff_robot_poses:
                    robot_traj.append(drp) # 10 dimensions
            elif action_type == 'absolute':
                for rp in robot_poses:
                    robot_traj.append(rp) # 10 dimensions

        elif target_space == 'task':
            if action_type == 'relative':
                for dep, dgp in zip(diff_ee_poses, diff_gripper_poses):
                    action = dep + dgp
                    robot_traj.append(action) # 8 dimensions
            elif action_type == 'absolute':
                for ep, gp in zip(ee_poses, gripper_poses):
                    action = ep[0] + transpose(ep[1]) + gp
                    robot_traj.append(action) # 9 dimensions

        # Get data for object
        object_poses = data['object_pose']

        # Prepare object_dataset
        object_traj = []
        for op in object_poses:
            object_traj.append(op)

        action_dataset.append([action_name, robot_traj])
        object_dataset.append([action_name, object_traj])

    return action_dataset, object_dataset

def parse_metadata(metadataset):
    env_info = {'initial_robot_pose': [],
                'initial_object_pose': [],
                'goal_robot_pose': [],
                'goal_object_pose': [],
                'target_object_name': [],
                'skill_name': []}
    for metadata in metadataset:
        # Get target objects
        target_object_name = metadata['target_object_name']
        env_info['target_object_name'].append(target_object_name)

        # Get data for object / Prepare object_dataset
        object_init_poses = metadata['object_init_pose']
        env_info['initial_object_pose'].append(object_init_poses)

        object_goal_poses = metadata['object_goal_pose']
        env_info['goal_object_pose'].append(object_goal_poses)

        # Get initial pose of robot / Prepare action_datase
        robot_init_poses = metadata['robot_init_pose'] # 10 joints
        env_info['initial_robot_pose'].append(robot_init_poses)

        robot_goal_poses = metadata['robot_goal_pose'] # 10 joints
        env_info['goal_robot_pose'].append(robot_goal_poses)

        env_info['skill_name'].append(metadata['skill_name'])

    return env_info

def parse_skill_metadata(metadataset):
    env_info = {'initial_robot_pose': [],
                'initial_object_pose': [],
                'goal_robot_pose': [],
                'goal_object_pose': [],
                'target_robot_pose': [],
                'target_object_name': [],
                'fixed_object_name': []}
    for metadata in metadataset:
        # Get target objects
        target_object_name = metadata['target_object_name']
        env_info['target_object_name'].append(target_object_name)

        # Get fixed objects
        fixed_object_name = metadata['fixed_object_name']
        env_info['fixed_object_name'].append(fixed_object_name)

        # Get data for object / Prepare object_dataset
        object_init_poses = metadata['object_init_pose']
        env_info['initial_object_pose'].append(object_init_poses)

        object_goal_poses = metadata['object_goal_pose']
        env_info['goal_object_pose'].append(object_goal_poses)

        # Get initial pose of robot / Prepare action_datase
        robot_init_poses = metadata['robot_init_pose'] # 10 joints
        env_info['initial_robot_pose'].append(np.array(robot_init_poses))

        robot_goal_poses = metadata['robot_goal_pose'] # 10 joints
        env_info['goal_robot_pose'].append(np.array(robot_goal_poses))

        robot_target_poses = metadata['robot_target_pose'] # 7 dim
        env_info['target_robot_pose'].append(np.array(robot_target_poses))

    return env_info

def transpose(quat):
    x, y, z, w = quat
    return [w, x, y, z]

def calculate_delta_theta(delta_quat):
    # Transform quaternion to rotation matrix
    rotation_matrix = np.zeros((3, 3))
    x, y, z, w = delta_quat
    rotation_matrix[0, 0] = 1 - 2 * (y**2 + z**2)
    rotation_matrix[0, 1] = 2 * (x*y - z*w)
    rotation_matrix[0, 2] = 2 * (x*z + y*w)
    rotation_matrix[1, 0] = 2 * (x*y + z*w)
    rotation_matrix[1, 1] = 1 - 2 * (x**2 + z**2)
    rotation_matrix[1, 2] = 2 * (y*z - x*w)
    rotation_matrix[2, 0] = 2 * (x*z - y*w)
    rotation_matrix[2, 1] = 2 * (y*z + x*w)
    rotation_matrix[2, 2] = 1 - 2 * (x**2 + y**2)

    # Transform rotation matrix to euler angle
    roll = math.atan2(rotation_matrix[2, 1], rotation_matrix[2, 2])
    pitch = math.atan2(-rotation_matrix[2, 0], math.sqrt(rotation_matrix[2, 1]**2 + rotation_matrix[2, 2]**2))
    yaw = math.atan2(rotation_matrix[1, 0], rotation_matrix[0, 0])

    # Reults of euler angle
    delta_euler = [roll, pitch, yaw]

    return delta_euler