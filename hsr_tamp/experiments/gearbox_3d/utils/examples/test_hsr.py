#!/usr/bin/env python

import os
import sys
import numpy as np
import pybullet as p

sys.path.append('..')
from pybullet_tools.hsrb_utils import TOP_HOLDING_ARM, SIDE_HOLDING_ARM, HSRB_URDF, \
    HSR_GROUPS, open_arm, get_disabled_collisions
from pybullet_tools.utils import set_base_values, joint_from_name, quat_from_euler, set_joint_position, \
    set_joint_positions, add_data_path, connect, plan_base_motion, plan_joint_motion, enable_gravity, \
    joint_controller, dump_body, load_model, joints_from_names, wait_if_gui, disconnect, get_joint_positions, \
    get_link_pose, link_from_name, get_pose, wait_if_gui, load_pybullet, set_quat, Euler, PI, RED, add_line, \
    wait_for_duration, LockRenderer, HideOutput

SLEEP = None

def test_base_motion(hsr, base_start, base_goal, obstacles=[]):
    #disabled_collisions = get_disabled_collisions(hsr)
    set_base_values(hsr, base_start)
    wait_if_gui('Plan Base?')
    base_limits = ((-2.5, -2.5), (2.5, 2.5))
    with LockRenderer(lock=False):
        base_path = plan_base_motion(hsr, base_goal, base_limits, obstacles=obstacles)
    if base_path is None:
        print('Unable to find a base path')
        return
    print(len(base_path))
    for bq in base_path:
        set_base_values(hsr, bq)
        if SLEEP is None:
            wait_if_gui('Continue?')
        else:
            wait_for_duration(SLEEP)

#####################################

def test_arm_motion(hsr, arm_joints, arm_goal):
    disabled_collisions = get_disabled_collisions(hsr)
    wait_if_gui('Plan Arm?')
    with LockRenderer(lock=False):
        arm_path = plan_joint_motion(hsr, arm_joints, arm_goal, disabled_collisions=disabled_collisions)
    if arm_path is None:
        print('Unable to find an arm path')
        return
    print(len(arm_path))
    for q in arm_path:
        set_joint_positions(hsr, arm_joints, q)
        wait_for_duration(0.01)

def test_arm_control(hsr, arm_joints, arm_start):
    wait_if_gui('Control Arm?')
    real_time = False
    enable_gravity()
    p.setRealTimeSimulation(real_time)
    for _ in joint_controller(hsr, arm_joints, arm_start):
        if not real_time:
            p.stepSimulation()

#####################################

def test_ikfast(hsr):
    from pybullet_tools.ikfast.hsrb.ik import get_tool_pose, get_ikfast_generator, hsr_inverse_kinematics
    arm_joints = joints_from_names(hsr, HSR_GROUPS['arm'])
    base_joints = joints_from_names(hsr, HSR_GROUPS['base'])
    base_arm_joints = base_joints + arm_joints

    arm = 'arm'
    pose = get_tool_pose(hsr, arm)

    print('get_link_pose: ', get_link_pose(hsr, link_from_name(hsr, 'hand_palm_link')))
    print('get_tool_pose: ', pose)
    for i in range(100):
        pose_x = 2.5
        pose_y = 2.0
        pose_z = 0.5

        rotation = np.random.choice(['foward', 'back', 'right', 'left'])
        if rotation == 'foward':
            angle = ([0.707107, 0.0, 0.707107, 0.0])
        elif rotation == 'back':
            angle = ([0.0, -0.70710678, 0.0, 0.70710678])
        elif rotation == 'right':
            angle = ([0.5, -0.5, 0.5, 0.5])
        elif rotation == 'left':
            angle = ([0.5, 0.5, 0.5, -0.5])
        tool_pose = ((pose_x, pose_y, pose_z), angle)
        ik_poses = hsr_inverse_kinematics(hsr, arm, tool_pose)
        if ik_poses != None:
            set_joint_positions(hsr, base_arm_joints, ik_poses)

        ee_pose = get_tool_pose(hsr, arm)
        print("ee_pose:", ee_pose)
        wait_if_gui()

#####################################

def main():
    connect(use_gui=True)
    add_data_path()

    plane = p.loadURDF("plane.urdf")

    directory = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    add_data_path(directory)

    table_path = "models/table_collision/table.urdf"
    table = load_pybullet(table_path, fixed_base=True)
    set_quat(table, quat_from_euler(Euler(yaw=PI/2)))
    obstacles = [table]

    hsr_urdf = HSRB_URDF
    with HideOutput():
        hsr = load_model(hsr_urdf, fixed_base=True)
    dump_body(hsr)

    base_start = (-2, -2, 0)
    base_goal = (2, 2, 0)
    arm_start = SIDE_HOLDING_ARM
    arm_goal = TOP_HOLDING_ARM

    arm_joints = joints_from_names(hsr, HSR_GROUPS['arm'])
    torso_joints = joints_from_names(hsr, HSR_GROUPS['torso'])
    gripper_joints = joints_from_names(hsr, HSR_GROUPS['gripper'])

    print('Set joints')
    set_joint_positions(hsr, arm_joints, arm_start)
    set_joint_positions(hsr, torso_joints, [0.0])
    set_joint_positions(hsr, gripper_joints, [0, 0])

    add_line(base_start, base_goal, color=RED)
    print(base_start, base_goal)

    print('Test base motion')
    test_base_motion(hsr, base_start, base_goal, obstacles=obstacles)

    print('Test arm motion')
    test_arm_motion(hsr, arm_joints, arm_goal)

    test_ikfast(hsr)

    while True:
        word = input('Input something: ')
        if word == 'Finish':
            disconnect()
            break

if __name__ == '__main__':
    main()