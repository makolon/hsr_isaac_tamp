import random
import numpy as np
import pybullet as p
from scipy.spatial.transform import Rotation as R

from ..utils import get_ik_limits, compute_forward_kinematics, compute_inverse_kinematics, select_solution, \
    USE_ALL, USE_CURRENT
from ...hsrb_utils import HSR_TOOL_FRAMES, get_torso_arm_joints, get_base_joints, \
    get_gripper_link, get_arm_joints, get_base_arm_joints
from ...utils import multiply, get_link_pose, link_from_name, get_joint_positions, \
    joint_from_name, invert, all_between, sub_inverse_kinematics, set_joint_positions, \
    inverse_kinematics, get_joint_positions, pairwise_collision, \
    get_custom_limits, get_custom_limits_with_base
from ...ikfast.utils import IKFastInfo

BASE_FRAME = 'base_footprint'
TORSO_JOINT = 'torso_lift_joint'
ROTATION_JOINT = 'joint_rz'
LIFT_JOINT = 'arm_lift_joint'
IK_FRAME = {'arm': 'hand_palm_link'}

#####################################

HSRB_INFOS = {arm: IKFastInfo(
                module_name='hsrb.ikArm',
                base_link=BASE_FRAME,
                ee_link=IK_FRAME[arm],
                free_joints=[TORSO_JOINT]
            ) for arm in IK_FRAME}

def get_if_info(arm):
    return HSRB_INFOS[arm]

#####################################

def get_tool_pose(robot, arm):
    from .ikArm import armFK
    arm_fk = {'arm': armFK}
    ik_joints = get_base_arm_joints(robot, arm)
    conf = get_joint_positions(robot, ik_joints)
    assert len(conf) == 8
    base_from_tool = compute_forward_kinematics(arm_fk[arm], conf)
    world_from_base = get_link_pose(robot, link_from_name(robot, BASE_FRAME))
    return multiply(world_from_base, base_from_tool)

#####################################

def is_ik_compiled():
    try:
        from .ikArm import armIK
        return True

    except ImportError:
        return False

def get_ikfast_generator(robot, arm, ik_pose, rotation_limits=USE_ALL, lift_limits=USE_ALL, custom_limits={}):
    from .ikArm import armIK

    arm_ik = {'arm': armIK}
    world_from_base = get_link_pose(robot, link_from_name(robot, BASE_FRAME))
    base_from_ik = multiply(invert(world_from_base), ik_pose)
    sampled_joints = [joint_from_name(robot, name) for name in [ROTATION_JOINT, LIFT_JOINT]]
    sampled_limits = [get_ik_limits(robot, joint, limits) for joint, limits in zip(sampled_joints, [rotation_limits, lift_limits])]
    arm_joints = get_arm_joints(robot, arm)
    base_joints = get_base_joints(robot, arm)
    min_limits, max_limits = get_custom_limits_with_base(robot, arm_joints, base_joints, custom_limits)

    # arm_rot = R.from_quat(ik_pose[1]).as_euler('xyz')[0]
    arm_rot = np.pi # TODO: modify
    sampled_limits = [(arm_rot-np.pi, arm_rot-np.pi), (0.0, 0.34)]
    while True:
        sampled_values = [random.uniform(*limits) for limits in sampled_limits]
        confs = compute_inverse_kinematics(arm_ik[arm], base_from_ik, sampled_values)
        solutions = [q for q in confs if all_between(min_limits, q, max_limits)]
        yield solutions
        if all(lower == upper for lower, upper in sampled_limits):
            break

def get_pybullet_generator(robot, arm, ik_pose, torso_limits=USE_ALL, upper_limits=USE_ALL, custom_limits={}):
    world_from_base = get_link_pose(robot, link_from_name(robot, BASE_FRAME))
    base_from_ik = multiply(invert(world_from_base), ik_pose)
    sampled_joints = [joint_from_name(robot, name) for name in [TORSO_JOINT]]
    sampled_limits = [get_ik_limits(robot, joint, limits) for joint, limits in zip(sampled_joints, [torso_limits])]
    arm_joints = get_torso_arm_joints(robot, arm)
    min_limits, max_limits = get_custom_limits(robot, arm_joints, custom_limits)

    while True:
        sampled_values = [random.uniform(*limits) for limits in sampled_limits]
        confs = inverse_kinematics(robot, 35, base_from_ik)
        confs = ((confs[3], confs[6], confs[7], confs[8], confs[9], 1.0),)
        solutions = [q for q in confs if all_between(min_limits, q, max_limits)]
        yield solutions
        if all(lower == upper for lower, upper in sampled_limits):
            break

def get_tool_from_ik(robot, arm):
    world_from_tool = get_link_pose(robot, link_from_name(robot, HSR_TOOL_FRAMES[arm]))
    world_from_ik = get_link_pose(robot, link_from_name(robot, IK_FRAME[arm]))
    return multiply(invert(world_from_tool), world_from_ik)

def sample_tool_ik(robot, arm, tool_pose, nearby_conf=USE_CURRENT, max_attempts=100, solver='ikfast', **kwargs):
    ik_pose = multiply(tool_pose, get_tool_from_ik(robot, arm))

    if solver == 'ikfast':
        generator = get_ikfast_generator(robot, arm, ik_pose, **kwargs)
    elif solver == 'pybullet':
        generator = get_pybullet_generator(robot, arm, ik_pose, **kwargs)
    else:
        pass # generator = get_pinocchio_generator(robot, arm, ik_pose, **kwargs)

    base_arm_joints = get_base_arm_joints(robot, arm)

    for _ in range(max_attempts):
        try:
            solutions = next(generator)
            if solutions:
                return select_solution(robot, base_arm_joints, solutions, nearby_conf=nearby_conf)
        except StopIteration:
            break

    return None

def hsr_inverse_kinematics(robot, arm, gripper_pose, obstacles=[], custom_limits={}, solver='ikfast', set_pose=True, **kwargs):
    arm_link = get_gripper_link(robot, arm)
    arm_joints = get_arm_joints(robot, arm)
    base_arm_joints = get_base_arm_joints(robot, arm)

    if is_ik_compiled():
        ik_joints = get_base_arm_joints(robot, arm)
        base_arm_conf = sample_tool_ik(robot,
                                       arm,
                                       gripper_pose,
                                       custom_limits=custom_limits,
                                       solver=solver,
                                       **kwargs)
        if base_arm_conf is None:
            return None
        if set_pose:
            set_joint_positions(robot, ik_joints, base_arm_conf)

    else:
        arm_conf = sub_inverse_kinematics(robot, arm_joints[0], arm_link, gripper_pose, custom_limits=custom_limits)
        if arm_conf is None:
            return None

    if any(pairwise_collision(robot, b) for b in obstacles):
        return None

    return get_joint_positions(robot, base_arm_joints)
