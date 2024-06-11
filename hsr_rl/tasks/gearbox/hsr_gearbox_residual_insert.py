# Copyright (c) 2018-2022, NVIDIA Corporation
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
#    list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
#    this list of conditions and the following disclaimer in the documentation
#    and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its
#    contributors may be used to endorse or promote products derived from
#    this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

import carb
import hydra
import torch
import numpy as np
from scipy.spatial.transform import Rotation as R
from hsr_rl.utils.dataset_utils import load_skill_dataset
from hsr_rl.tasks.utils.gp_utils import ResidualGPModel
from hsr_rl.tasks.utils.mlp_utils import ResidualDeepModel
from hsr_rl.tasks.gearbox.base.hsr_gearbox_base import HSRGearboxBaseTask
from hsr_rl.tasks.gearbox.base.hsr_gearbox_base import norm_diff_pos, norm_diff_rot, calc_diff_pos, calc_diff_rot

from omni.isaac.core.simulation_context import SimulationContext
from omni.isaac.core.utils.stage import get_current_stage
from omni.isaac.core.utils.torch.maths import torch_rand_float, tensor_clamp
from omni.isaac.core.utils.torch.rotations import quat_diff_rad, quat_mul, normalize
from pxr import Gf, Sdf, UsdGeom, PhysxSchema, UsdPhysics


class HSRGearboxResidualInsertTask(HSRGearboxBaseTask):
    def __init__(
        self,
        name,
        sim_config,
        env
    ) -> None:
        HSRGearboxBaseTask.__init__(self, name, sim_config, env)

        # Load ik controller
        self.ik_controller = self.set_ik_controller()

        if self._residual_dynamics:
            # Load residual dynamics models
            self.load_mlp_model()
        if self._residual_observation:
            # Load rediaul observation models
            self.load_gp_model()
        return

    def add_anchor(self):
        # The (x, y, z) axes in the global coordinate system are transformed to (z, y, x) in the local coordinate system
        self._stage = get_current_stage()
        pos_noise = self.pos_noise.to('cpu').detach().numpy().copy()
        rot_noise = self.rot_noise.to('cpu').detach().numpy().copy()
        for i in range(self._num_envs):
            object_prim_path = f"/World/envs/env_{i}/{self._dynamic_obj_names[i]}/{self._dynamic_obj_names[i]}"
            fixed_joint_path = f"/World/envs/env_{i}/hsrb/hand_palm_link/fixed_joint"

            # Fix the object to left hand
            if 'shaft' in self._dynamic_obj_names[i]:
                # Position randomize
                fixed_pos = np.array([0.0, 0.0, 0.06])
                idx = 2 if self.randomize_idx[i] == 0 else 0
                fixed_pos[idx] += pos_noise[i]
                anchor_pos = Gf.Vec3d(fixed_pos.tolist())

                # Rotation randomize
                fixed_rot = np.array([-0.70710678, 0.0, 0.70710678, 0.0])
                fixed_rot = R.from_quat(fixed_rot).as_euler('xyz')
                fixed_rot = np.array([fixed_rot[0], fixed_rot[1]+rot_noise[i], fixed_rot[2]])
                fixed_rot = R.from_euler('xyz', fixed_rot).as_quat()
                anchor_rot = Gf.Quatf(fixed_rot[3], Gf.Vec3f(fixed_rot[0], fixed_rot[1], fixed_rot[2]))
            elif 'gear' in self._dynamic_obj_names[i]:
                # Position randomize
                fixed_pos = np.array([0.0, 0.0, 0.12])
                idx = 2 if self.randomize_idx[i] == 0 else 1
                fixed_pos[idx] += pos_noise[i]
                anchor_pos = Gf.Vec3d(fixed_pos.tolist())

                # Rotation randomize
                fixed_rot = np.array([-0.5, 0.5, 0.5, 0.5])
                fixed_rot = R.from_quat(fixed_rot).as_euler('xyz')
                fixed_rot = np.array([fixed_rot[0]+rot_noise[i], fixed_rot[1], fixed_rot[2]])
                fixed_rot = R.from_euler('xyz', fixed_rot).as_quat()
                anchor_rot = Gf.Quatf(fixed_rot[3], Gf.Vec3f(fixed_rot[0], fixed_rot[1], fixed_rot[2]))

            self.fix_to_hand(self._stage, fixed_joint_path, object_prim_path, anchor_pos, anchor_rot, i, 'hand_palm_link')

    def fix_to_hand(self, stage, joint_path, prim_path, anchor_pos, anchor_rot, env_id, hand_name):
        # Calculate anchor rotation
        anchor1_rot = anchor_rot
        anchor2_rot = Gf.Quatf(1.0, Gf.Vec3f(0.0, 0.0, 0.0))

        # D6 fixed joint
        FixedJoint = UsdPhysics.FixedJoint.Define(stage, joint_path)
        FixedJoint.CreateBody0Rel().SetTargets([f"/World/envs/env_{env_id}/hsrb/{hand_name}"])
        FixedJoint.CreateBody1Rel().SetTargets([prim_path])
        FixedJoint.CreateLocalPos0Attr().Set(Gf.Vec3f(anchor_pos))
        FixedJoint.CreateLocalRot0Attr().Set(anchor1_rot)
        FixedJoint.CreateLocalPos1Attr().Set(Gf.Vec3f(0, 0, 0))
        FixedJoint.CreateLocalRot1Attr().Set(anchor2_rot)

    def get_observations(self):
        # Get end effector positions and orientations
        end_effector_positions, end_effector_orientations = self._robots._fingertip_centered.get_world_poses(clone=False)
        end_effector_positions = end_effector_positions[:, 0:3] - self._env_pos
        end_effector_orientations = end_effector_orientations[:, [3, 0, 1, 2]]

        target_obj_positions, target_obj_orientations = [], []
        for i in range(self._num_envs):
            parts_pos, parts_rot = self._parts[self._dynamic_obj_names[i]].get_world_poses()
            parts_pos -= self._env_pos
            if 'shaft' in self._dynamic_obj_names[i]:
                parts_rot[i, :] = parts_rot[i, [3, 0, 1, 2]]
            target_obj_positions.append(parts_pos[i, :])
            target_obj_orientations.append(parts_rot[i, :])

        target_obj_positions = torch.stack(target_obj_positions)
        target_obj_orientations = torch.stack(target_obj_orientations)

        # Calculate difference
        diff_ee_pos = calc_diff_pos(end_effector_positions, target_obj_positions)
        diff_ee_rot = calc_diff_rot(end_effector_orientations, target_obj_orientations)

        # Calculate difference between target object pose and current object pose
        diff_obj_pos = calc_diff_pos(target_obj_positions, self.target_parts_pos)
        diff_obj_rot = calc_diff_rot(target_obj_orientations, self.target_parts_rot)

        # Get dof positions / velocities
        dof_pos = self._robots.get_joint_positions(clone=False)

        self.obs_buf[..., 0:5] = dof_pos[..., self.arm_dof_idxs]
        self.obs_buf[..., 5:8] = diff_ee_pos
        self.obs_buf[..., 8:12] = diff_ee_rot
        self.obs_buf[..., 12:15] = diff_obj_pos
        self.obs_buf[..., 15:19] = diff_obj_rot

        # Observation noise
        if self._residual_observation:
            residual_obs = torch.cat((dof_pos[..., self.movable_dof_indices], end_effector_positions, end_effector_orientations), dim=1).view(self._num_envs, 1, -1)

            # Calculate joint positions
            robot_jacobian = self._robots.get_jacobians(clone=False)[:, self._robots._body_indices['hand_palm_link']-1, :, self.movable_dof_indices]

            ee_pose_noise = []
            for idx in range(len(self.state_noise_models)):
                ee_noise, _, _ = self.state_noise_models[idx].predict(residual_obs)
                ee_pose_noise.append(ee_noise)
            ee_pose_noise = torch.stack(ee_pose_noise).view(self._num_envs, -1)
            self.ik_controller.set_command(ee_pose_noise)
            delta_noise = self.ik_controller.compute_delta(end_effector_positions, end_effector_orientations, robot_jacobian)
        else:
            dof_pos_noise = torch_rand_float(-0.001, 0.001, (self._num_envs, len(self.arm_dof_idxs)), device=self._device)
            diff_pos_noise = torch_rand_float(-0.001, 0.001, (self._num_envs, 3), device=self._device)
            diff_rot_noise = torch_rand_float(-0.001, 0.001, (self._num_envs, 4), device=self._device)
            target_pos_noise = torch_rand_float(-0.001, 0.001, (self._num_envs, 3), device=self._device)
            target_rot_noise = torch_rand_float(-0.001, 0.001, (self._num_envs, 4), device=self._device)

        # Add observation noise
        self.obs_buf[..., 0:5] += dof_pos_noise
        self.obs_buf[..., 5:8] += diff_pos_noise
        self.obs_buf[..., 8:12] += diff_rot_noise
        self.obs_buf[..., 12:15] += target_pos_noise
        self.obs_buf[..., 15:19] += target_rot_noise

        observations = {
            self._robots.name: {
                "obs_buf": self.obs_buf
            }
        }
        return observations

    def pre_physics_step(self, actions) -> None:
        if not self._env._world.is_playing():
            return

        reset_env_ids = self.reset_buf.nonzero(as_tuple=False).squeeze(-1)
        if len(reset_env_ids) > 0:
            for _ in range(3):
                self.reset_idx(reset_env_ids)
                SimulationContext.step(self._env._world, render=True)

        # Update position targets from actions
        replay_indices = self.replay_count.clone().detach()
        jt_actions = self._dt * self._action_speed_scale * actions.to(self.device)

        # Calculate target joint positions
        dof_pos = self._robots.get_joint_positions()
        moveble_dof_pos = dof_pos[:, self.movable_dof_indices]

        # Action noise
        if self._residual_dynamics:
            end_effector_positions, end_effector_orientations = self._robots._fingertip_centered.get_world_poses(clone=False)
            end_effector_positions = end_effector_positions[:, 0:3] - self._env_pos
            end_effector_orientations = end_effector_orientations[:, [3, 0, 1, 2]]
            residual_obs = torch.cat((moveble_dof_pos, end_effector_positions, end_effector_orientations), dim=1)
            with torch.no_grad():
                action_noise = self._control_frequency * self.action_noise_model(residual_obs).to(self._device)
        else:
            action_noise = torch_rand_float(-0.001, 0.001, (self._num_envs, len(self.movable_dof_indices)), device=self._device)

        self.dof_position_targets[..., self.movable_dof_indices] = moveble_dof_pos
        self.dof_position_targets[..., self.movable_dof_indices] += self.calculate_command(moveble_dof_pos, replay_indices)
        self.dof_position_targets[..., self.movable_dof_indices] += jt_actions
        self.dof_position_targets[..., self.movable_dof_indices] += action_noise
        self.dof_position_targets[:] = tensor_clamp(
            self.dof_position_targets, self.robot_dof_lower_limits, self.robot_dof_upper_limits
        )

        # Modify torso joint positions
        arm_pos = dof_pos[:, self.arm_dof_idxs]
        scaled_arm_lift_pos = arm_pos[:, 0] / self.arm_dof_upper[0]
        scaled_torso_lift_pos = scaled_arm_lift_pos * self.torso_dof_upper[0]
        self.dof_position_targets[:, self.torso_dof_idx] = scaled_torso_lift_pos.unsqueeze(dim=1)

        # Reset position targets for reset envs
        self.dof_position_targets[reset_env_ids] = self.initial_dof_positions[reset_env_ids]
        self._robots.set_joint_position_targets(self.dof_position_targets)

    def reset_idx(self, env_ids):
        env_ids_32 = env_ids.type(torch.int32)
        env_ids_64 = env_ids.type(torch.int64)

        # Reset root state for robots in selected envs
        self._robots.set_world_poses(self.initial_robot_pos[env_ids_64], self.initial_robot_rot[env_ids_64], indices=env_ids_32)

        # Reset DOF states for robots in selected envs
        self._robots.set_joint_position_targets(self.initial_dof_positions, indices=env_ids_32)
        self._robots.set_joint_positions(self.initial_dof_positions, indices=env_ids_32)
        self._robots.set_joint_velocities(self.initial_dof_velocities, indices=env_ids_32)

        # Reset DOF effort
        gripper_dof_effort = torch.tensor([0., 0.], device=self._device)
        self._robots.set_joint_efforts(gripper_dof_effort, indices=env_ids_32, joint_indices=self.gripper_proximal_dof_idxs)

        # Shaft1 pos / rot, velocities
        self.shaft1_pos = self.initial_shaft1_pos.clone()
        self.shaft1_rot = self.initial_shaft1_rot.clone()

        # Shaft2 pos / rot, velocities
        self.shaft2_pos = self.initial_shaft2_pos.clone()
        self.shaft2_rot = self.initial_shaft2_rot.clone()

        # Gear1 pos / rot, velocities
        self.gear1_pos = self.initial_gear1_pos.clone()
        self.gear1_rot = self.initial_gear1_rot.clone()

        # Gear2 pos / rot, velocities
        self.gear2_pos = self.initial_gear2_pos.clone()
        self.gear2_rot = self.initial_gear2_rot.clone()

        # Gear3 pos / rot, velocities
        self.gear3_pos = self.initial_gear3_pos.clone()
        self.gear3_rot = self.initial_gear3_rot.clone()

        # For randomizae (x, y) position
        self.pos_noise = torch_rand_float(-0.01, 0.01, (len(env_ids), 1), device=self._device).view(-1)
        self.rot_noise = torch_rand_float(-0.1, 0.1, (len(env_ids), 1), device=self._device).view(-1)

        # Add noise to parts position (x, y, z)
        # Select one direction to add noise
        for i in range(self._num_envs):
            if self._dynamic_obj_names[i] == 'shaft1':
                idx = np.random.choice([0, 2])
                self.shaft1_pos[i, idx] += self.pos_noise[i]
                shaft1_rot = R.from_quat(self.shaft1_rot[i, [1, 2, 3, 0]].to('cpu').detach().numpy().copy()).as_euler('xyz')
                shaft1_rot[2] += self.rot_noise[i]
                shaft1_rot = torch.tensor(R.from_euler('xyz', shaft1_rot).as_quat(), device=self._device)
                self.shaft1_rot[i, :] = shaft1_rot[[3, 0, 1, 2]]
            elif self._dynamic_obj_names[i] == 'shaft2':
                idx = np.random.choice([0, 2])
                self.shaft2_pos[i, idx] += self.pos_noise[i]
                shaft2_rot = R.from_quat(self.shaft2_rot[i, [1, 2, 3, 0]].to('cpu').detach().numpy().copy()).as_euler('xyz')
                shaft2_rot[2] += self.rot_noise[i]
                shaft2_rot = torch.tensor(R.from_euler('xyz', shaft2_rot).as_quat(), device=self._device)
                self.shaft2_rot[i, :] = shaft2_rot[[3, 0, 1, 2]]
            elif self._dynamic_obj_names[i] == 'gear1':
                idx = np.random.choice([0, 1])
                self.gear1_pos[i, idx] += self.pos_noise[i]
                gear1_rot = R.from_quat(self.gear1_rot[i, [1, 2, 3, 0]].to('cpu').detach().numpy().copy()).as_euler('xyz')
                gear1_rot[2] += self.rot_noise[i]
                gear1_rot = torch.tensor(R.from_euler('xyz', gear1_rot).as_quat(), device=self._device)
                self.gear1_rot[i, :] = gear1_rot[[3, 0, 1, 2]]
            elif self._dynamic_obj_names[i] == 'gear2':
                idx = np.random.choice([0, 1])
                self.gear2_pos[i, idx] += self.pos_noise[i]
                gear2_rot = R.from_quat(self.gear2_rot[i, [1, 2, 3, 0]].to('cpu').detach().numpy().copy()).as_euler('xyz')
                gear2_rot[2] += self.rot_noise[i]
                gear2_rot = torch.tensor(R.from_euler('xyz', gear2_rot).as_quat(), device=self._device)
                self.gear2_rot[i, :] = gear2_rot[[3, 0, 1, 2]]
            elif self._dynamic_obj_names[i] == 'gear3':
                idx = np.random.choice([0, 1])
                self.gear3_pos[i, idx] += self.pos_noise[i]
                gear3_rot = R.from_quat(self.gear3_rot[i, [1, 2, 3, 0]].to('cpu').detach().numpy().copy()).as_euler('xyz')
                gear3_rot[2] += self.rot_noise[i]
                gear3_rot = torch.tensor(R.from_euler('xyz', gear3_rot).as_quat(), device=self._device)
                self.gear3_rot[i, :] = gear3_rot[[3, 0, 1, 2]]

        # Reset parts pose in selected envs
        self._shaft1.set_world_poses(self.shaft1_pos[env_ids_64], self.shaft1_rot[env_ids_64], indices=env_ids_32)
        self._shaft2.set_world_poses(self.shaft2_pos[env_ids_64], self.shaft2_rot[env_ids_64], indices=env_ids_32)
        self._gear1.set_world_poses(self.gear1_pos[env_ids_64], self.gear1_rot[env_ids_64], indices=env_ids_32)
        self._gear2.set_world_poses(self.gear2_pos[env_ids_64], self.gear2_rot[env_ids_64], indices=env_ids_32)
        self._gear3.set_world_poses(self.gear3_pos[env_ids_64], self.gear3_rot[env_ids_64], indices=env_ids_32)

        # Reset parts velocity in selected envs
        self._shaft1.set_velocities(self.initial_shaft1_velocities[env_ids_64], indices=env_ids_32)
        self._shaft2.set_velocities(self.initial_shaft2_velocities[env_ids_64], indices=env_ids_32)
        self._gear1.set_velocities(self.initial_gear1_velocities[env_ids_64], indices=env_ids_32)
        self._gear2.set_velocities(self.initial_gear2_velocities[env_ids_64], indices=env_ids_32)
        self._gear3.set_velocities(self.initial_gear3_velocities[env_ids_64], indices=env_ids_32)

        # Bookkeeping
        self.gripper_close[env_ids] = False
        self.gripper_open[env_ids] = False
        self.gripper_hold[env_ids] = False
        self.extras[env_ids] = 0
        self.reset_buf[env_ids] = 0
        self.replay_count[env_ids] = 0
        self.progress_buf[env_ids] = 0
        self.is_collided[env_ids] = 0
        self.insert_success[env_ids] = 0

    def post_reset(self):
        if self._task_cfg["sim"]["disable_gravity"]:
            self.disable_gravity()

        self.set_dof_idxs()
        self.set_dof_limits()
        self.set_default_state()

        # Reset robot pos / rot, and velocities
        self.initial_robot_pos, self.initial_robot_rot = self._robots.get_world_poses()
        self.initial_robot_velocities = self._robots.get_velocities()

        # Reset shaft1 pos / rot, and velocities
        self.initial_shaft1_rot[:] = torch.tensor([0.0, 0.0, 1.0, 0.0], device=self._device)
        self.initial_shaft1_pos[:, :2] += self._env_pos[:, :2]
        self.initial_shaft1_velocities = self._shaft1.get_velocities()

        # Reset shaft2 pos / rot, and velocities
        self.initial_shaft2_rot[:] = torch.tensor([0.0, 0.0, 1.0, 0.0], device=self._device)
        self.initial_shaft2_pos[:, :2] += self._env_pos[:, :2]
        self.initial_shaft2_velocities = self._shaft2.get_velocities()

        # Reset gear1 pos / rot, and velocities
        _, self.initial_gear1_rot = self._gear1.get_world_poses()
        self.initial_gear1_pos[:, :2] += self._env_pos[:, :2]
        self.initial_gear1_velocities = self._gear1.get_velocities()

        # Reset gear2 pos / rot, and velocities
        _, self.initial_gear2_rot = self._gear2.get_world_poses()
        self.initial_gear2_pos[:, :2] += self._env_pos[:, :2]
        self.initial_gear2_velocities = self._gear2.get_velocities()

        # Reset gear3 pos / rot, and velocities
        _, self.initial_gear3_rot = self._gear3.get_world_poses()
        self.initial_gear3_pos[:, :2] += self._env_pos[:, :2]
        self.initial_gear3_velocities = self._gear3.get_velocities()

        # Remove rigid body physics from static objects
        self._gear1.disable_rigid_body_physics(indices=self.static_gear1_env_ids)
        self._gear2.disable_rigid_body_physics(indices=self.static_gear2_env_ids)
        self._gear3.disable_rigid_body_physics(indices=self.static_gear3_env_ids)
        self._shaft1.disable_rigid_body_physics(indices=self.static_shaft1_env_ids)
        self._shaft2.disable_rigid_body_physics(indices=self.static_shaft2_env_ids)

        # Add virtual blocks of contact for shafts
        self.add_block()

        # Reset all environment
        indices = torch.arange(self._num_envs, dtype=torch.int64, device=self._device)
        self.reset_idx(indices)

    def calculate_metrics(self) -> None:
        self.rew_buf[:] = torch.zeros(self._num_envs, device=self._device, dtype=torch.float)

        # Get current parts positions and orientations
        curr_parts_positions, curr_parts_orientations = [], []
        for i in range(self._num_envs):
            parts_pos, parts_rot = self._parts[self._dynamic_obj_names[i]].get_world_poses()
            parts_pos -= self._env_pos
            if 'shaft' in self._dynamic_obj_names[i]:
                parts_rot[i, :] = parts_rot[i, [3, 0, 1, 2]]
            curr_parts_positions.append(parts_pos[i, :])
            curr_parts_orientations.append(parts_rot[i, :])

        curr_parts_positions = torch.stack(curr_parts_positions)
        curr_parts_orientations = torch.stack(curr_parts_orientations)

        # Calculate difference between target object pose and final object pose
        target_pos_dist = norm_diff_pos(curr_parts_positions, self.target_parts_pos)
        target_rot_dist = norm_diff_rot(curr_parts_orientations, self.target_parts_rot)

        target_pos_dist_reward = 1.0 / (1.0 + target_pos_dist ** 2)
        target_pos_dist_reward *= target_pos_dist_reward
        target_pos_dist_reward = torch.where(target_pos_dist <= 0.04, target_pos_dist_reward * 2, target_pos_dist_reward)

        target_rot_dist_reward = 1.0 / (1.0 + target_rot_dist ** 2)
        target_rot_dist_reward *= target_rot_dist_reward
        target_rot_dist_reward = torch.where(target_rot_dist <= 0.1, target_rot_dist_reward * 2, target_rot_dist_reward)

        self.rew_buf[:] += target_pos_dist_reward * self._task_cfg['rl']['target_position_distance_scale']
        self.rew_buf[:] += target_rot_dist_reward * self._task_cfg['rl']['target_rotation_distance_scale']

        # In this policy, episode length is constant across all envs
        is_last_step = (self.progress_buf[0] == self._max_episode_length - 1)
        if is_last_step:
            # Check if block is picked up and above table
            insert_success = self._check_insert_success()
            self.rew_buf[:] += insert_success * self._task_cfg['rl']['insert_success_bonus']
            self.extras['insert_successes'] = torch.mean(insert_success.float())
            self.insert_success = torch.where(
                insert_success[:] == 1,
                torch.ones_like(insert_success),
                -torch.ones_like(insert_success)
            )

    def _check_insert_success(self):
        parts_positions, parts_orientations = [], []
        for i in range(self._num_envs):
            prop_pos, prop_rot = self._parts[self._target_obj_names[i]].get_world_poses()
            prop_pos -= self._env_pos
            if 'shaft' in self._target_obj_names[i]:
                prop_rot[i, :] = prop_rot[i, [3, 0, 1, 2]]
            parts_positions.append(prop_pos[i, :])
            parts_orientations.append(prop_rot[i, :])

        parts_positions = torch.stack(parts_positions)
        parts_orientations = torch.stack(parts_orientations)

        # Check difference between target pose and current pose
        target_pos_dist = norm_diff_pos(parts_positions, self.target_parts_pos)
        target_rot_dist = norm_diff_rot(parts_orientations, self.target_parts_rot)

        print('target_pos_dist:', target_pos_dist)
        print('target_rot_dist:', target_rot_dist)

        insert_success = torch.where(
            target_pos_dist < torch.tensor([0.02], device=self._device),
            torch.ones((self._num_envs,), device=self._device),
            torch.zeros((self._num_envs,), device=self._device)
        )
        insert_success = torch.where(
            target_rot_dist < torch.tensor([0.08], device=self._device),
            insert_success,
            torch.zeros((self._num_envs,), device=self._device)
        )
        print('insert_success:', insert_success)
        print('success_rate:', torch.mean(insert_success.float()))

        return insert_success

    def post_physics_step(self):
        """Step buffers. Refresh tensors. Compute observations and reward. Reset environments."""
        self.progress_buf[:] += 1
        self.replay_count[:] = torch.where(
            self.progress_buf % self.control_frequency_inv == 0,
            self.replay_count + 1,
            self.replay_count
        )
        if self._env._world.is_playing():
            # In this policy, episode length is constant
            env_ids = torch.arange(0, self._num_envs, device=self._device)
            self._hold_gripper(env_ids)

            self.get_observations()
            self.get_states()
            self.calculate_metrics()
            self.is_done()
            self.get_extras()

        return self.obs_buf, self.rew_buf, self.reset_buf, self.extras

    def load_exp_dataset(self):
        exp_actions, metadata = load_skill_dataset('gearbox_problem', 'insert')

        # Get target object name
        self._target_obj_names = metadata['target_object_name']

        # Padding
        exp_traj = []
        max_length = self._max_episode_length
        for action_name, traj in exp_actions:
            traj_np = np.array(traj)
            padding = np.zeros((max_length - traj_np.shape[0], 10))
            padding[:] = traj[-1]
            seq = np.concatenate([traj_np, padding])
            exp_traj.append(seq)

        # Parallelize
        exp_seq = []
        for i in range(self._num_envs):
            exp_seq.append(exp_traj[i])

        # To device
        exp_seq = np.array(exp_seq)
        exp_seq = torch.tensor(exp_seq, device=self._device, dtype=torch.float)

        # Parallelize
        init_rp = []
        for i in range(self._num_envs):
            rp = metadata['initial_robot_pose'][i]
            if 'shaft' in self._target_obj_names[i]:
                rp[-4:-2] = [0.25, 0.25]
            elif 'gear' in self._target_obj_names[i]:
                rp[-4:-2] = [0.15, 0.15]
            init_rp.append(rp) # 15 dim

        # List to numpy.ndarray
        init_rp = np.array(init_rp)

        goal_ee_pos, goal_ee_orn = [], []
        for i in range(self._num_envs):
            ee_pos, ee_rot = metadata['target_robot_pose'][i]
            goal_ee_pos.append(ee_pos) # 3 dim
            goal_ee_orn.append(ee_rot) # 4 dim

        # Parallelize
        goal_parts_pos, goal_parts_rot = [], []
        for i in range(self._num_envs):
            goal_parts_pos.append(
                metadata['goal_object_pose'][i][self._target_obj_names[i]][0])
            goal_parts_rot.append(
                metadata['goal_object_pose'][i][self._target_obj_names[i]][1])

        # List to numpy.ndarray
        goal_parts_pos = np.array(goal_parts_pos)
        goal_parts_rot = np.array(goal_parts_rot)

        self.target_parts_pos = torch.tensor(goal_parts_pos, device=self._device)
        self.target_parts_rot = torch.tensor(goal_parts_rot, device=self._device)

        # To device
        self.initial_dof_positions = torch.tensor(init_rp, device=self._device, dtype=torch.float)
        self.initial_dof_velocities = torch.zeros_like(self.initial_dof_positions, device=self._device, dtype=torch.float)
        self.target_ee_positions = torch.tensor(goal_ee_pos, device=self._device, dtype=torch.float)
        self.target_ee_orientations = torch.tensor(goal_ee_orn, device=self._device, dtype=torch.float)

        init_shaft1_pos, init_shaft1_rot = [], []
        init_shaft2_pos, init_shaft2_rot = [], []
        init_gear1_pos, init_gear1_rot = [], []
        init_gear2_pos, init_gear2_rot = [], []
        init_gear3_pos, init_gear3_rot = [], []
        for i in range(self._num_envs):
            init_shaft1_pos.append(metadata['initial_object_pose'][i]['shaft1'][0])
            init_shaft1_rot.append(metadata['initial_object_pose'][i]['shaft1'][1])
            init_shaft2_pos.append(metadata['initial_object_pose'][i]['shaft2'][0])
            init_shaft2_rot.append(metadata['initial_object_pose'][i]['shaft2'][1])
            init_gear1_pos.append(metadata['initial_object_pose'][i]['gear1'][0])
            init_gear1_rot.append(metadata['initial_object_pose'][i]['gear1'][1])
            init_gear2_pos.append(metadata['initial_object_pose'][i]['gear2'][0])
            init_gear2_rot.append(metadata['initial_object_pose'][i]['gear2'][1])
            init_gear3_pos.append(metadata['initial_object_pose'][i]['gear3'][0])
            init_gear3_rot.append(metadata['initial_object_pose'][i]['gear3'][1])

        self.initial_shaft1_pos = torch.tensor(init_shaft1_pos, device=self._device)
        self.initial_shaft1_rot = torch.tensor(init_shaft1_rot, device=self._device)
        self.initial_shaft2_pos = torch.tensor(init_shaft2_pos, device=self._device)
        self.initial_shaft2_rot = torch.tensor(init_shaft2_rot, device=self._device)
        self.initial_gear1_pos = torch.tensor(init_gear1_pos, device=self._device)
        self.initial_gear1_rot = torch.tensor(init_gear1_rot, device=self._device)
        self.initial_gear2_pos = torch.tensor(init_gear2_pos, device=self._device)
        self.initial_gear2_rot = torch.tensor(init_gear2_rot, device=self._device)
        self.initial_gear3_pos = torch.tensor(init_gear3_pos, device=self._device)
        self.initial_gear3_rot = torch.tensor(init_gear3_rot, device=self._device)

        return exp_seq, metadata # (dataset_length, num_actions)
    
    def load_gp_model(self):
        hdf5_position_x = [f'/root/tamp-hsr/hsr_ros/hsr_ws/src/gearbox_3d/script/rl_controller/residual_dataset/insert_dataset_position_x_1.h5']
        hdf5_position_y = [f'/root/tamp-hsr/hsr_ros/hsr_ws/src/gearbox_3d/script/rl_controller/residual_dataset/insert_dataset_position_y_1.h5']
        hdf5_position_z = [f'/root/tamp-hsr/hsr_ros/hsr_ws/src/gearbox_3d/script/rl_controller/residual_dataset/insert_dataset_position_z_1.h5']
        hdf5_rotation_x = [f'/root/tamp-hsr/hsr_ros/hsr_ws/src/gearbox_3d/script/rl_controller/residual_dataset/insert_dataset_rotation_x_1.h5']
        hdf5_rotation_y = [f'/root/tamp-hsr/hsr_ros/hsr_ws/src/gearbox_3d/script/rl_controller/residual_dataset/insert_dataset_rotation_y_1.h5']
        hdf5_rotation_z = [f'/root/tamp-hsr/hsr_ros/hsr_ws/src/gearbox_3d/script/rl_controller/residual_dataset/insert_dataset_rotation_z_1.h5']
        hdf5_insert_paths = [hdf5_position_x, hdf5_position_y, hdf5_position_z,
                           hdf5_rotation_x, hdf5_rotation_y, hdf5_rotation_z]

        self.state_noise_models = []
        for hdf5_paths in hdf5_insert_paths:
            gp_regression = ResidualGPModel()
            gp_regression.set_up_model(hdf5_paths=hdf5_paths)

            # Load model
            gp_regression.load_model('insert')
            self.state_noise_models.append(gp_regression)

    def load_mlp_model(self):
        model_path = '/root/tamp-hsr/hsr_rl/tasks/utils/mlp_models/insert_regression_model.pth'
        self.action_noise_model = ResidualDeepModel()
        self.action_noise_model.load_state_dict(torch.load(model_path))
        self.action_noise_model.to(self._device)
        self.action_noise_model.eval()