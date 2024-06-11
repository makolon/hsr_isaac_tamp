import numpy as np
import itertools

link_name = ['base_footprint',
    'base_link',
    'base_roll_link',
    'base_r_drive_wheel_link',
    'base_l_drive_wheel_link',
    'base_r_passive_wheel_x_frame',
    'base_r_passive_wheel_y_frame',
    'base_r_passive_wheel_z_link',
    'base_l_passive_wheel_x_frame',
    'base_l_passive_wheel_y_frame',
    'base_l_passive_wheel_z_link',
    'base_range_sensor_link',
    'base_imu_frame',
    'base_f_bumper_link',
    'base_b_bumper_link',
    'torso_lift_link',
    'head_pan_link',
    'head_tilt_link',
    'head_l_stereo_camera_link',
    'head_l_stereo_camera_gazebo_frame',
    'head_r_stereo_camera_link',
    'head_r_stereo_camera_gazebo_frame',
    'head_center_camera_frame',
    'head_center_camera_gazebo_frame',
    'head_rgbd_sensor_link',
    'head_rgbd_sensor_gazebo_frame',
    'arm_lift_link',
    'arm_flex_link',
    'arm_roll_link',
    'wrist_flex_link',
    'wrist_ft_sensor_mount_link',
    'wrist_ft_sensor_frame',
    'wrist_roll_link',
    'hand_palm_link',
    'hand_motor_dummy_link',
    'hand_l_proximal_link',
    'hand_l_spring_proximal_link',
    'hand_l_mimic_distal_link',
    'hand_l_distal_link',
    'hand_l_finger_tip_frame',
    'hand_l_finger_vacuum_frame',
    'hand_r_proximal_link',
    'hand_r_spring_proximal_link',
    'hand_r_mimic_distal_link',
    'hand_r_distal_link',
    'hand_r_finger_tip_frame',
    'hand_camera_frame',
    'hand_camera_gazebo_frame']


def create_never_collision(link_name):
    results = []
    for pair in itertools.combinations(link_name, 2):
        results.append(tuple(pair))
    return results

if __name__ == '__main__':
    results = create_never_collision(link_name)
    print('results: ', results)
