#!/usr/bin/env/python3
import sys
import rospy
import torch
import numpy as np
from scipy.spatial.transform import Rotation as R

# TODO: fix
sys.path.append('/root/tamp-hsr/hsr_tamp/experiments/gearbox_3d/')
sys.path.append('/root/tamp-hsr/')
sys.path.append('../..')
from base_controller.controller import BaseController
from hsr_rl.tasks.utils.pinoc_utils import HSRIKSolver
from hsr_rl.tasks.utils.ik_utils import DifferentialInverseKinematicsCfg, DifferentialInverseKinematics


class ExecutePlan(BaseController):
    def __init__(self, standalone=False):
        super(ExecutePlan, self).__init__()

        self._device = 'cuda'
        self._command_type = 'position_rel'
        self._dt = torch.tensor(1.0/self.control_freq, device=self._device)
        self._gamma = torch.tensor(0.3, device=self._device)
        self._clip_obs = torch.tensor(5.0, device=self._device)

        # Core module
        self.hsr_ik_utils = HSRIKSolver()

        # Set ik controller
        self.ik_controller = self.set_ik_controller()

    def set_ik_controller(self):
        ik_control_cfg = DifferentialInverseKinematicsCfg(
            command_type=self._command_type,
            ik_method="pinv",
            position_offset=(0.0, 0.0, 0.0),
            rotation_offset=(1.0, 0.0, 0.0, 0.0),
        )
        return DifferentialInverseKinematics(ik_control_cfg, 1, self._device)

    def execute(self):
        while not rospy.is_shutdown():
            # Calculate robot jacobian
            ee_pos, ee_rot = self.mocap_interface.get_pose('end_effector')
            joint_positions = np.array(self.hsr_interface.get_joint_positions())
            robot_jacobian = self.hsr_ik_utils.get_jacobian(joint_positions)

            # To tensor and device
            target_pos = torch.tensor([0.8, -0.5, 0.2], device=self._device).view(1, 3)
            target_rot = torch.tensor([0.70710678, 0.0, 0.70710678, 0.0], device=self._device).view(1, 4)
            ee_pos = torch.tensor(ee_pos, dtype=torch.float32, device=self._device).view(1, 3)
            ee_rot = torch.tensor(ee_rot, dtype=torch.float32, device=self._device).view(1, 4)
            robot_jacobian = torch.tensor(robot_jacobian, dtype=torch.float32, device=self._device).view(1, 6, 8)

            # Calculate diff_pose divide with ratio
            diff_pos = target_pos - ee_pos
            diff_rot = target_rot - ee_rot
            diff_rot = torch.tensor(R.from_quat(diff_rot.to('cpu').detach().numpy().copy()).as_euler('ZYX'), device=self._device)
            diff_rot = torch.tensor([0.0, 0.0, 0.0], device=self._device).view(1, 3)
            diff_pose = torch.cat((diff_pos, diff_rot), dim=1)

            # Multiply target 6D pose and residual 6D pose
            self.ik_controller.set_command(diff_pose)

            # Calcurate delta pose
            joint_positions = torch.tensor(joint_positions, device=self._device).view(1, 8)
            delta_pose = self.ik_controller.compute_delta(ee_pos, ee_rot, robot_jacobian, real=True)
            target_pose = joint_positions + delta_pose
            target_pose = torch.squeeze(target_pose, dim=0)
            target_pose = target_pose.to('cpu').detach().numpy().copy() # 8 dim

            # Add residual control
            target_base_pose = target_pose[:3]
            target_arm_pose = target_pose[3:]

            print('target_pose:', target_pose)

            # Set target pose
            base_traj = self.set_base_pose(target_base_pose)
            arm_traj = self.set_arm_pose(target_arm_pose)

            # Publish command
            self.base_pub.publish(base_traj)
            self.arm_pub.publish(arm_traj)
            self.rate.sleep()


if __name__ == '__main__':
    exec_plan = ExecutePlan()
    exec_plan.execute()