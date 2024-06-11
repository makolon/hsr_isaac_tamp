#!/usr/bin/env/python3
import os
import sys
import yaml
import rospy
import torch
import numpy as np
from scipy.spatial.transform import Rotation as R

# TODO: fix
sys.path.append('/root/tamp-hsr/hsr_tamp/experiments/gearbox_3d/')
sys.path.append('..')
from base_controller.controller import BaseController
from rl_agent import ResidualRL

sys.path.append('/root/tamp-hsr/')
from hsr_rl.utils.hydra_cfg.reformat import omegaconf_to_dict
from hsr_rl.tasks.utils.pinoc_utils import HSRIKSolver
from hsr_rl.tasks.utils.ik_utils import DifferentialInverseKinematicsCfg, DifferentialInverseKinematics


def load_config(policy_name: str = 'pick'):
    file_name = os.path.join('.', 'config', policy_name + '_config.yaml')
    with open(file_name, 'r') as f:
        config = yaml.load(f, Loader=yaml.SafeLoader)

    return config

def norm_diff_pos(p1: torch.Tensor, p2: torch.Tensor) -> torch.Tensor:
    # Calculate norm
    diff_norm = torch.norm(p1 - p2, p=2, dim=-1)

    return diff_norm

def norm_diff_xy(p1: torch.Tensor, p2: torch.Tensor) -> torch.Tensor:
    # Calculate norm
    diff_norm = torch.norm(p1[:2] - p2[:2], p=2, dim=-1)

    return diff_norm

def norm_diff_rot(q1: torch.Tensor, q2: torch.Tensor) -> torch.Tensor:
    # Calculate norm
    diff_norm1 = torch.norm(q1 - q2, p=2, dim=-1)
    diff_norm2 = torch.norm(q2 - q1, p=2, dim=-1)

    diff_norm = torch.min(diff_norm1, diff_norm2)

    return diff_norm

def normalize(x, eps: float = 1e-9):
    return x / x.norm(p=2, dim=-1).clamp(min=eps, max=None).unsqueeze(-1)

def quaternion_multiply(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    aw, ax, ay, az = torch.unbind(a, -1)
    bw, bx, by, bz = torch.unbind(b, -1)
    ow = aw * bw - ax * bx - ay * by - az * bz
    ox = aw * bx + ax * bw + ay * bz - az * by
    oy = aw * by - ax * bz + ay * bw + az * bx
    oz = aw * bz + ax * by - ay * bx + az * bw
    return torch.stack((ow, ox, oy, oz), -1)

def calc_diff_pos(p1, p2):
    return p1 - p2

def calc_diff_rot(q1, q2):
    q1 = normalize(q1)
    q2 = normalize(q2)

    scaling = torch.tensor([1, -1, -1, -1], device=q1.device)
    q1_inv = q1 * scaling
    q_diff = quaternion_multiply(q2, q1_inv)

    return q_diff

def modify_target_pose(target_obj_pose, gearbox_base_pose, object_name):
    # Calculate position difference from motion capture
    #   difference from left_hole: [ 0.16399157 -0.09748133 -0.03273092]
    #   difference from right_hole: [ 0.16733813  0.09592897 -0.04950504]
    #   difference from middle_shaft: [ 0.2633127  -0.00121711 -0.11001092]
    gearbox_base_pos, _ = gearbox_base_pose
    left_offset = (-0.164, 0.0975, 0.0)
    right_offset = (-0.167, -0.0960, 0.0)
    middle_offset = (-0.263, 0.0, 0.0)
    if object_name == 'gear1':
        return ((gearbox_base_pos[0]+left_offset[0],
                 gearbox_base_pos[1]+left_offset[1],
                 target_obj_pose[0][2]),
                 target_obj_pose[1])
    elif object_name == 'gear2':
        return ((gearbox_base_pos[0]+right_offset[0],
                 gearbox_base_pos[1]+right_offset[1],
                 target_obj_pose[0][2]),
                 target_obj_pose[1])
    elif object_name == 'gear3':
        return ((gearbox_base_pos[0]+middle_offset[0],
                 gearbox_base_pos[1]+middle_offset[1],
                 target_obj_pose[0][2]),
                 target_obj_pose[1])
    elif object_name == 'shaft1':
        return ((gearbox_base_pos[0]+left_offset[0],
                 gearbox_base_pos[1]+left_offset[1],
                 target_obj_pose[0][2]),
                 target_obj_pose[1])
    elif object_name == 'shaft2':
        return ((gearbox_base_pos[0]+right_offset[0],
                 gearbox_base_pos[1]+right_offset[1],
                 target_obj_pose[0][2]),
                 target_obj_pose[1])


class ExecutePlan(BaseController):
    def __init__(self, standalone=False):
        super(ExecutePlan, self).__init__()

        self._device = 'cuda'
        self._dt = torch.tensor(1.0/self.control_freq, device=self._device)

    def augment_plan(self, plan):
        # Replay_trajectory
        return self.tamp_planner.execute(plan)

    def check_success(self, target_pos, object_pos):
        object_pos = torch.tensor(object_pos)
        target_pos = torch.tensor(target_pos)

        # Calculate norm distance
        pos_dist = norm_diff_xy(object_pos, target_pos)
        print('insert distance:', pos_dist)

        insert_success = torch.where(
            pos_dist < torch.tensor([0.003]),
            torch.ones((1,)),
            torch.zeros((1,))
        )
        return insert_success

    def execute(self):
        plan = self.plan()

        if plan is None:
            return None
        
        ee_pos, ee_rot = self.mocap_interface.get_pose('end_effector')
        print('ee_pos:', ee_pos)

        # Augment plan
        _, _, _, insert_metadata = self.augment_plan(plan)

        object_names = []
        for action_name, args in plan:
            # Post process TAMP commands to hsr executable actions
            _, object_name, _ = self.process(action_name, args)
            if action_name == 'insert':
                object_names.append(object_name)

        np.set_printoptions(precision=6)
        for i, inserted_pose in enumerate(insert_metadata['target_object_pose']):
            print('target object:', object_names[i])
            input('wait_for_user')

            finish = False
            while not finish:
                obj_pos, _ = self.mocap_interface.get_pose(object_names[i])
                inserted_pos, _ = inserted_pose

                print('object_pos:', obj_pos)
                print('inserted_pos:', inserted_pos)
                finish = self.check_success(inserted_pos, obj_pos)

                self.rate.sleep()


if __name__ == '__main__':
    exec_plan = ExecutePlan()
    exec_plan.execute()