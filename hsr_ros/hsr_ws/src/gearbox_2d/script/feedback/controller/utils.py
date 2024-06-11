import random
import numpy as np
import pybullet as p
from collections import namedtuple
from scipy.spatial.transform import Rotation as R

from .hsrb_utils import get_joint_limits, get_joint_position, get_joint_positions, get_distance

IKFastInfo = namedtuple('IKFastInfo', ['module_name', 'base_link', 'ee_link', 'free_joints'])

USE_ALL = False
USE_CURRENT = None

############ Mathematics

def invert(pose):
    point, quat = pose
    return p.invertTransform(point, quat) # TODO: modify

def multiply(*poses):
    pose = poses[0]
    for next_pose in poses[1:]:
        pose = p.multiplyTransforms(pose[0], pose[1], *next_pose) # TODO: modify
    return pose

##############

def all_between(lower_limits, values, upper_limits):
    assert len(lower_limits) == len(values)
    assert len(values) == len(upper_limits)
    return np.less_equal(lower_limits, values).all() and \
           np.less_equal(values, upper_limits).all()

def compute_forward_kinematics(fk_fn, conf):
    pose = fk_fn(list(conf))
    pos, rot = pose
    quat = R.from_matrix(rot).as_quat()

    return pos, quat

def compute_inverse_kinematics(ik_fn, pose, sampled=[]):
    pos, quat = pose[0], pose[1]
    rot = R.from_quat(quat).as_matrix().tolist()

    if len(sampled) == 0:
        solutions = ik_fn(list(rot), list(pos))
    else:
        solutions = ik_fn(list(rot), list(pos), list(sampled))

    if solutions is None:
        return []

    return solutions

def get_ik_limits(joint, limits=USE_ALL):
    if limits is USE_ALL:
        return get_joint_limits(joint)

    elif limits is USE_CURRENT:
        value = get_joint_position(joint)
        return value, value

    return limits

def select_solution(joints, solutions, nearby_conf=USE_ALL, **kwargs):
    if not solutions:
        return None

    if nearby_conf is USE_ALL:
        return random.choice(solutions)

    if nearby_conf is USE_CURRENT:
        nearby_conf = get_joint_positions(joints)

    return min(solutions, key=lambda conf: get_distance(nearby_conf, conf, **kwargs))