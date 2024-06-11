from __future__ import print_function
import os
import sys
import glob
import random
import numpy as np
from scipy.spatial.transform import Rotation as R
from .robot_utils import multiply, all_between, select_solution, \
    compute_inverse_kinematics, compute_forward_kinematics

BASE_FRAME = 'base_footprint'
TORSO_JOINT = 'torso_lift_joint'
ROTATION_JOINT = 'joint_rz'
LIFT_JOINT = 'arm_lift_joint'
IK_FRAME = {'arm': 'hand_palm_link'}

def get_ik_lib():
    lib_path = os.environ['PYTHONPATH'].split(':')[1] # TODO: modify
    ik_lib_path = glob.glob(os.path.join(lib_path, '**/hsrb'), recursive=True)
    return ik_lib_path[1]

class HSRIKSolver(object):
    def __init__(
        self,
        move_group: str = "arm",
    ) -> None:
        self.arm = move_group
        self.ik_joints = ['world_joint', 'arm_lift_joint', 'arm_flex_joint',
                 'arm_roll_joint', 'wrist_flex_joint', 'wrist_roll_joint']

    def solve_fk_hsr(self, base_pos, curr_joints):
        sys.path.append(get_ik_lib())
        from ikArm import armFK
        arm_fk = {'arm': armFK}

        assert len(curr_joints) == 8

        base_from_tool = compute_forward_kinematics(arm_fk[self.arm], curr_joints)
        return multiply(base_pos, base_from_tool)
    
    def solve_ik_pos_hsr(self, des_pos, des_quat, curr_joints=None, n_trials=7, dt=0.1, pos_threshold=0.05, angle_threshold=15.*np.pi/180, verbose=True):
        def get_ik_generator(ik_pose, custom_limits={}):
            sys.path.append(get_ik_lib())
            from ikArm import armIK
            arm_ik = {'arm': armIK}

            min_limits = [-10.0, -10.0, -10.0, 0.0, -2.617, -1.919, -1.919, -1.919]
            max_limits = [10.0, 10.0, 10.0, 0.69, 0.0, 3.665, 1.221, 3.665]

            arm_rot = R.from_quat(des_quat).as_euler('xyz')[0]
            sampled_limits = [(arm_rot-np.pi, arm_rot-np.pi), (0.0, 0.34)]
            while True:
                sampled_values = [random.uniform(*limits) for limits in sampled_limits]
                confs = compute_inverse_kinematics(arm_ik[self.arm], ik_pose, sampled_values)
                solutions = [q for q in confs if all_between(min_limits, q, max_limits)]
                yield solutions
                if all(lower == upper for lower, upper in sampled_limits):
                    break

        ik_pose = (des_pos, des_quat)
        generator = get_ik_generator(ik_pose)
        for _ in range(n_trials):
            try:
                solutions = next(generator)
                if solutions:
                    return select_solution(self.ik_joints, solutions, curr_joints, nearby_conf=False)
            except StopIteration:
                break


if __name__ == '__main__':
    hsr_ik_solver = HSRIKSolver()
    best_q = hsr_ik_solver.solve_ik_pos_hsr(np.array([0.5, 0.1, 0.3]), np.array([0., 0., 0., 1.0]))
    print('best_q: ', best_q)