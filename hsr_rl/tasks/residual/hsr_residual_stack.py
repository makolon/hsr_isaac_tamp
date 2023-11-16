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

from hsr_rl.tasks.base.rl_task import RLTask
from hsr_rl.robots.articulations.hsr import HSR
from hsr_rl.robots.articulations.views.hsr_view import HSRView
from hsr_rl.utils.dataset_utils import load_dataset

from omni.isaac.core.utils.prims import get_prim_at_path
from omni.isaac.core.utils.stage import get_current_stage
from omni.isaac.core.prims.rigid_prim_view import RigidPrimView
from omni.isaac.core.prims.geometry_prim_view import GeometryPrimView
from omni.isaac.core.articulations.articulation_view import ArticulationView
from omni.isaac.core.utils.torch.transformations import *
from omni.isaac.core.utils.torch.rotations import *
from omni.isaac.core.utils.stage import print_stage_prim_paths
from omni.isaac.core.simulation_context import SimulationContext
from omni.isaac.core.objects import DynamicCuboid, FixedCuboid
from omni.isaac.core.materials import PhysicsMaterial
from omni.isaac.sensor import _sensor
from omni.isaac.cloner import Cloner
from omni.physx.scripts import utils, physicsUtils

import re
import math
import torch
import numpy as np
from pxr import Usd, UsdGeom, PhysxSchema, UsdPhysics, Gf


class HSRResidualStackTask(RLTask):
    def __init__(
        self,
        name,
        sim_config,
        env,
    ) -> None:
        self._sim_config = sim_config
        self._cfg = sim_config.config
        self._task_cfg = sim_config.task_config

        self._device = self._cfg["sim_device"]
        self._num_envs = self._task_cfg["env"]["numEnvs"]
        self._env_spacing = self._task_cfg["env"]["envSpacing"]

        # Get dt for integrating velocity commands and checking limit violations
        self._dt = torch.tensor(self._task_cfg["sim"]["dt"] * self._task_cfg["env"]["controlFrequencyInv"], device=self._device)

        self._num_observations = self._task_cfg["env"]["num_observations"]
        self._num_actions = self._task_cfg["env"]["num_actions"]
        self._num_props = self._task_cfg["env"]["numProps"]

        # Table and prop settings
        self._table_height = 0.26
        self._table_width = 0.5
        self._table_depth = 0.5
        self._table_size = 1.0
        self._prop_size = self._sim_config.task_config["sim"]["parts"]["size"]
        self._prop_density = self._sim_config.task_config["sim"]["parts"]["density"]
        self._prop_static_friction = self._sim_config.task_config["sim"]["parts"]["static_friction"]
        self._prop_dynamic_friction = self._sim_config.task_config["sim"]["parts"]["dynamic_friction"]
        self._prop_restitution = self._sim_config.task_config["sim"]["parts"]["restitution"]

        self._gripper_mass = self._sim_config.task_config["sim"]["gripper"]["mass"]
        self._gripper_density = self._sim_config.task_config["sim"]["gripper"]["density"]
        self._gripper_static_friction = self._sim_config.task_config["sim"]["gripper"]["static_friction"]
        self._gripper_dynamic_friction = self._sim_config.task_config["sim"]["gripper"]["dynamic_friction"]
        self._gripper_restitution = self._sim_config.task_config["sim"]["gripper"]["restitution"]

        self._pick_success = self._table_height + 0.05
        self._place_success = torch.tensor([0.2, 0.2], device=self._device)

        self._hsr_position = torch.tensor([0.0, 0.0, 0.03], device=self._device)
        self._hsr_rotation = torch.tensor([1.0, 0.0, 0.0, 0.0], device=self._device)
        self._table_position = torch.tensor([1.625, 0.0, self._table_height/2], device=self._device)
        self._table_rotation = torch.tensor([1.0, 0.0, 0.0, 0.0], device=self._device)
        self._prop_position = torch.tensor([1.48, 0.455, self._table_height+self._prop_size/2], device=self._device)
        self._prop_rotation = torch.tensor([1.0, 0.0, 0.0, 0.0], device=self._device)

        self._action_speed_scale = self._task_cfg["env"]["actionSpeedScale"]
        self._max_episode_length = self._task_cfg["env"]["maxEpisodeLength"]

        # Start at 'home' positions
        self.torso_start = torch.tensor([0.1], device=self._device)
        self.base_start = torch.tensor([0.0, 0.0, 0.0], device=self._device)
        self.arm_start = torch.tensor([0.1, -1.570796, 0.0, -0.392699, 0.0], device=self._device)
        self.gripper_proximal_start = torch.tensor([0.75, 0.75], device=self._device)

        self.initial_dof_positions = torch.tensor([0.0, 0.0, 0.0, 0.1, 0.1, -1.570796, 0.0, 0.0, 0.0, 0.0, 0.0, 0.75, 0.75, 0.0, 0.0], device=self._device)

        # Dof joint gains
        self.joint_kps = torch.tensor([1e9, 1e9, 5.7296e10, 1e9, 1e9, 5.7296e10,
            5.7296e10, 5.7296e10, 5.7296e10, 5.7296e10, 5.7296e10, 2.8648e4,
            2.8648e4, 5.7296e10, 5.7296e10], device=self._device)
        self.joint_kds = torch.tensor([1.4, 1.4, 80.2141, 1.4, 0.0, 80.2141, 0.0, 80.2141,
            0.0, 80.2141, 80.2141, 17.1887, 17.1887, 17.1887, 17.1887], device=self._device)

        # Dof joint friction coefficients
        self.joint_friction_coefficients = torch.tensor([1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0], device=self._device)

        # Joint & body names
        self._torso_joint_name = ["torso_lift_joint"]
        self._base_joint_names = ["joint_x", "joint_y", "joint_rz"]
        self._arm_names = ["arm_lift_joint", "arm_flex_joint", "arm_roll_joint", "wrist_flex_joint", "wrist_roll_joint"]
        self._gripper_proximal_names = ["hand_l_proximal_joint", "hand_r_proximal_joint"]

        # Values are set in post_reset after model is loaded
        self.torso_dof_idx = []
        self.base_dof_idxs = []
        self.arm_dof_idxs = []
        self.gripper_proximal_dof_idxs = []

        # Dof joint position limits
        self.torso_dof_lower = []
        self.torso_dof_upper = []
        self.base_dof_lower = []
        self.base_dof_upper = []
        self.arm_dof_lower = []
        self.arm_dof_upper = []
        self.gripper_p_dof_lower = []
        self.gripper_p_dof_upper = []

        # Add contact sensor
        self._contact_sensor_interface = _sensor.acquire_contact_sensor_interface()

        self.replay_count = torch.zeros(self._num_envs, device=self._device, dtype=torch.long)
        self.is_collided = torch.zeros(self._num_envs, device=self._device, dtype=torch.long)
        self.lift_success = torch.zeros(self._num_envs, device=self._device, dtype=torch.long)
        self.place_success = torch.zeros(self._num_envs, device=self._device, dtype=torch.long)
        self.gripper_close = torch.zeros(self._num_envs, device=self._device, dtype=torch.bool)
        self.gripper_open = torch.zeros(self._num_envs, device=self._device, dtype=torch.bool)
        self.gripper_hold = torch.zeros(self._num_envs, device=self._device, dtype=torch.bool)

        self.exp_actions = self.load_exp_dataset()

        RLTask.__init__(self, name, env)
        return

    def set_up_scene(self, scene) -> None:
        self.create_prop_material()
        self.create_gripper_material()

        self.add_hsr()
        self.add_prop()
        self.add_table1()
        self.add_table2()

        # Set up scene
        super().set_up_scene(scene)

        # Add robot to scene
        self._robots = HSRView(prim_paths_expr="/World/envs/.*/hsrb", name="hsrb_view")
        scene.add(self._robots)
        scene.add(self._robots._hands)
        scene.add(self._robots._lfingers)
        scene.add(self._robots._rfingers)
        scene.add(self._robots._fingertip_centered)

        # Add prop to scene
        self._props = RigidPrimView(prim_paths_expr="/World/envs/.*/prop", name="prop_view", reset_xform_properties=False)
        scene.add(self._props)

    def create_prop_material(self):
        self._stage = get_current_stage()
        self.propPhysicsMaterialPath = "/World/Physics_Materials/PropMaterial"

        utils.addRigidBodyMaterial(
            self._stage,
            self.propPhysicsMaterialPath,
            density=self._prop_density,
            staticFriction=self._prop_static_friction,
            dynamicFriction=self._prop_dynamic_friction,
        )

    def create_gripper_material(self):
        self._stage = get_current_stage()
        self.gripperPhysicsMaterialPath = "/World/Physics_Materials/GripperMaterial"

        utils.addRigidBodyMaterial(
            self._stage,
            self.gripperPhysicsMaterialPath,
            density=self._gripper_density,
            staticFriction=self._gripper_static_friction,
            dynamicFriction=self._gripper_dynamic_friction,
            restitution=self._gripper_restitution
        )

    def add_hsr(self):
        hsrb = HSR(prim_path=self.default_zero_env_path + "/hsrb",
                   name="hsrb",
                   translation=self._hsr_position,
                   orientation=self._hsr_rotation)
        self._sim_config.apply_articulation_settings("hsrb", get_prim_at_path(hsrb.prim_path), self._sim_config.parse_actor_config("hsrb"))

    def add_prop(self):
        prop = DynamicCuboid(prim_path=self.default_zero_env_path + "/prop",
                             name="prop",
                             translation=self._prop_position,
                             orientation=self._prop_rotation,
                             size=self._prop_size,
                             density=self._prop_density,
                             color=torch.tensor([0.2, 0.4, 0.6]))
        self._sim_config.apply_articulation_settings("prop", get_prim_at_path(prop.prim_path), self._sim_config.parse_actor_config("prop"))

        physicsUtils.add_physics_material_to_prim(
            self._stage,
            self._stage.GetPrimAtPath(self.default_zero_env_path + "/prop"),
            self.propPhysicsMaterialPath
        )

    def add_table1(self):
        table_pos = torch.tensor([1.5, -0.5, self._table_height/2], device=self._device)
        table_orn = torch.tensor([1.0, 0.0, 0.0, 0.0], device=self._device)

        table = FixedCuboid(prim_path=self.default_zero_env_path + "/table1",
                            name="table1",
                            translation=table_pos,
                            orientation=table_orn,
                            color=torch.tensor([0.75, 0.75, 0.75]),
                            scale=torch.tensor([self._table_width, self._table_depth, self._table_height]))
        self._sim_config.apply_articulation_settings("table1", get_prim_at_path(table.prim_path), self._sim_config.parse_actor_config("table"))

    def add_table2(self):
        table_pos = torch.tensor([1.5, 0.5, self._table_height/2], device=self._device)
        table_orn = torch.tensor([1.0, 0.0, 0.0, 0.0], device=self._device)

        table = FixedCuboid(prim_path=self.default_zero_env_path + "/table2",
                            name="table2",
                            translation=table_pos,
                            orientation=table_orn,
                            color=torch.tensor([0.75, 0.75, 0.75]),
                            scale=torch.tensor([self._table_width, self._table_depth, self._table_height]))
        self._sim_config.apply_articulation_settings("table2", get_prim_at_path(table.prim_path), self._sim_config.parse_actor_config("table"))

    def get_observations(self):
        # Get prop positions and orientations
        prop_positions, prop_orientations = self._props.get_world_poses(clone=False)
        prop_positions = prop_positions[:, 0:3] - self._env_pos

        # Get prop velocities
        prop_velocities = self._props.get_velocities(clone=False)
        prop_linvels = prop_velocities[:, 0:3]
        prop_angvels = prop_velocities[:, 3:6]

        # Get end effector positions and orientations
        end_effector_positions, end_effector_orientations = self._robots._fingertip_centered.get_world_poses(clone=False)
        end_effector_positions = end_effector_positions[:, 0:3] - self._env_pos

        # Get end effector velocities
        end_effector_velocities = self._robots._fingertip_centered.get_velocities(clone=False)
        end_effector_linvels = end_effector_velocities[:, 0:3]
        end_effector_angvels = end_effector_velocities[:, 3:6]

        self.prop_positions = prop_positions
        self.prop_linvels = prop_linvels

        dof_pos = self._robots.get_joint_positions(clone=False)
        dof_vel = self._robots.get_joint_velocities(clone=False)

        self.obs_buf[..., 0:10] = dof_pos[..., self.actuated_dof_indices]
        self.obs_buf[..., 10:20] = dof_vel[..., self.actuated_dof_indices]
        self.obs_buf[..., 20:23] = end_effector_positions
        self.obs_buf[..., 23:27] = end_effector_orientations
        self.obs_buf[..., 27:30] = end_effector_linvels
        self.obs_buf[..., 30:33] = end_effector_angvels
        self.obs_buf[..., 33:36] = prop_positions
        self.obs_buf[..., 38:42] = prop_orientations

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
            self.reset_idx(reset_env_ids)

        # Update position targets from actions
        replay_indices = torch.tensor(self.replay_count)
        self.dof_position_targets[..., self.movable_dof_indices] = self._robots.get_joint_positions(joint_indices=self.movable_dof_indices)
        self.dof_position_targets[..., self.movable_dof_indices] += self.exp_actions[replay_indices, :8] # replay tamp trajectory
        self.dof_position_targets[..., self.movable_dof_indices] += self._dt * self._action_speed_scale * actions.to(self.device) # add residual rl action
        self.dof_position_targets[:] = tensor_clamp(
            self.dof_position_targets, self.robot_dof_lower_limits, self.robot_dof_upper_limits
        )

        # Modify torso joint positions
        dof_pos = self._robots.get_joint_positions()
        arm_pos = dof_pos[:, self.arm_dof_idxs]
        scaled_arm_lift_pos = arm_pos[:, 0] / self.arm_dof_upper[0]
        scaled_torso_lift_pos = scaled_arm_lift_pos * self.torso_dof_upper[0]
        self.dof_position_targets[:, self.torso_dof_idx] = scaled_torso_lift_pos.unsqueeze(dim=1)

        # reset position targets for reset envs
        self.dof_position_targets[reset_env_ids] = self.initial_dof_positions
        self._robots.set_joint_position_targets(self.dof_position_targets)

    def reset_idx(self, env_ids):
        num_resets = len(env_ids)

        env_ids_32 = env_ids.type(torch.int32)
        env_ids_64 = env_ids.type(torch.int64)

        min_d = 0.0 # min horizontal dist from origin
        max_d = 0.0 # max horizontal dist from origin
        min_height = 0.0 # min vertical dist from origin
        max_height = 0.0 # min vertical dist from origin

        dists = torch_rand_float(min_d, max_d, (num_resets, 1), self._device)
        dirs = torch_random_dir_2((num_resets, 1), self._device)
        hpos = dists * dirs

        # Prop pos / rot, velocities
        self.prop_pos = self.initial_prop_pos.clone()
        self.prop_rot = self.initial_prop_rot.clone()
        # position
        self.prop_pos[env_ids_64, 0:2] += hpos[..., 0:2]
        self.prop_pos[env_ids_64, 2] += torch_rand_float(min_height, max_height, (num_resets, 1), self._device).squeeze()
        # rotation
        self.prop_rot[env_ids_64, 0] = 1
        self.prop_rot[env_ids_64, 1:] = 0

        # reset root state for props in selected envs
        self._props.set_world_poses(self.prop_pos[env_ids_64], self.prop_rot[env_ids_64], indices=env_ids_32)

        # reset root state for robots in selected envs
        self._robots.set_world_poses(self.initial_robot_pos[env_ids_64], self.initial_robot_rot[env_ids_64], indices=env_ids_32)

        # reset DOF states for robots in selected envs
        self._robots.set_joint_position_targets(self.initial_dof_positions, indices=env_ids_32)

        gripper_dof_effort = torch.tensor([0., 0.], device=self._device)
        self._robots.set_joint_efforts(gripper_dof_effort, indices=env_ids_32, joint_indices=self.gripper_proximal_dof_idxs)

        # bookkeeping
        self.gripper_close[env_ids] = False
        self.gripper_open[env_ids] = False
        self.gripper_hold[env_ids] = False
        self.replay_count[env_ids] = 0
        self.reset_buf[env_ids] = 0
        self.progress_buf[env_ids] = 0
        self.extras[env_ids] = 0
        self.is_collided[env_ids] = 0
        self.lift_success[env_ids] = 0
        self.place_success[env_ids] = 0

    def post_reset(self):
        self.set_dof_idxs()
        self.set_dof_limits()
        self.set_default_state()
        self.set_joint_gains()
        self.set_joint_frictions()

        # reset prop pos / rot, and velocities
        self.initial_robot_pos, self.initial_robot_rot = self._robots.get_world_poses()
        self.initial_robot_velocities = self._robots.get_velocities()

        # reset prop pos / rot, and velocities
        self.initial_prop_pos, self.initial_prop_rot = self._props.get_world_poses()
        self.initial_prop_velocities = self._props.get_velocities()

    def calculate_metrics(self) -> None:
        # Distance from hand to the ball
        dist = torch.norm(self.obs_buf[..., 33:36] - self.obs_buf[..., 20:23], p=2, dim=-1)
        dist_reward = 1.0 / (1.0 + dist ** 2)
        dist_reward *= dist_reward
        dist_reward = torch.where(dist <= 0.02, dist_reward * 2, dist_reward)

        self.rew_buf[:] = dist_reward * self._task_cfg['rl']['distance_scale']

        # In this policy, episode length is constant across all envs
        is_mid_step = (self.progress_buf[0] == int(self.exp_actions.size()[0] / 2))
        is_last_step = (self.progress_buf[0] == int(self.exp_actions.size()[0] - 1))
        if is_mid_step:
            # Check if block is picked up and above table
            lift_success = self._check_lift_success(height_threashold=self._pick_success)
            self.rew_buf[:] += lift_success * self._task_cfg['rl']['pick_success_bonus']
            self.extras['lift_successes'] = torch.mean(lift_success.float())
            self.lift_success = torch.where(
                lift_success[:] == 1,
                torch.ones_like(lift_success),
                -torch.ones_like(lift_success)
            )

        if is_last_step:
            # Check if block is in a target region
            place_success = self._check_place_success(place_threashold=self._place_success)
            self.rew_buf[:] += place_success * self._task_cfg['rl']['place_success_bonus']
            self.extras['place_successes'] = torch.mean(place_success.float())
            self.place_success = torch.where(
                place_success[:] == 1,
                torch.ones_like(place_success),
                -torch.ones_like(place_success)
            )

    def is_done(self) -> None:
        self.reset_buf = torch.where(
            self.progress_buf == self.exp_actions.size()[0] - 1,
            torch.ones_like(self.reset_buf),
            self.reset_buf
        )

    def post_physics_step(self):
        """Step buffers. Refresh tensors. Compute observations and reward. Reset environments."""
        self.replay_count[:] += 1
        self.progress_buf[:] += 1
        if self._env._world.is_playing():
            # In this policy, episode length is constant
            replay_indices = torch.tensor(self.replay_count)
            self.gripper_close = self.exp_actions[replay_indices, -1] < -0.01
            self.gripper_open = self.exp_actions[replay_indices, -1] > 0.01
            if torch.any(self.gripper_close):
                close_indices = torch.where(self.gripper_close)[0]
                self._close_gripper(close_indices, sim_steps=self._task_cfg['env']['num_gripper_close_sim_steps'])
            elif torch.any(self.gripper_open):
                open_indices = torch.where(self.gripper_open)[0]
                self._open_gripper(open_indices, sim_steps=self._task_cfg['env']['num_gripper_open_sim_steps'])
            elif torch.any(self.gripper_hold):
                hold_indices = torch.where(self.gripper_hold)[0]
                self._hold_gripper(hold_indices)

            self.get_observations()
            self.get_states()
            self.calculate_metrics()
            self.is_done()
            self.get_extras()

        return self.obs_buf, self.rew_buf, self.reset_buf, self.extras

    def set_dof_idxs(self):
        [self.torso_dof_idx.append(self._robots.get_dof_index(name)) for name in self._torso_joint_name]
        [self.base_dof_idxs.append(self._robots.get_dof_index(name)) for name in self._base_joint_names]
        [self.arm_dof_idxs.append(self._robots.get_dof_index(name)) for name in self._arm_names]
        [self.gripper_proximal_dof_idxs.append(self._robots.get_dof_index(name)) for name in self._gripper_proximal_names]

        # Movable joints
        self.actuated_dof_indices = torch.LongTensor(self.base_dof_idxs+self.arm_dof_idxs+self.gripper_proximal_dof_idxs).to(self._device) # torch.LongTensor([0, 1, 2, 3, 5, 7, 9, 10, 11, 12]).to(self._device)
        self.movable_dof_indices = torch.LongTensor(self.base_dof_idxs+self.arm_dof_idxs).to(self._device) # torch.LongTensor([0, 1, 2, 3, 5, 7, 9, 10]).to(self._device)

    def set_dof_limits(self): # dof position limits
        # (num_envs, num_dofs, 2)
        dof_limits = self._robots.get_dof_limits()
        dof_limits_lower = dof_limits[0, :, 0].to(self._device)
        dof_limits_upper = dof_limits[0, :, 1].to(self._device)

        # Set relevant joint position limit values
        self.torso_dof_lower = dof_limits_lower[self.torso_dof_idx]
        self.torso_dof_upper = dof_limits_upper[self.torso_dof_idx]
        self.base_dof_lower = dof_limits_lower[self.base_dof_idxs]
        self.base_dof_upper = dof_limits_upper[self.base_dof_idxs]
        self.arm_dof_lower = dof_limits_lower[self.arm_dof_idxs]
        self.arm_dof_upper = dof_limits_upper[self.arm_dof_idxs]
        self.gripper_p_dof_lower = dof_limits_lower[self.gripper_proximal_dof_idxs]
        self.gripper_p_dof_upper = dof_limits_upper[self.gripper_proximal_dof_idxs]

        self.robot_dof_lower_limits, self.robot_dof_upper_limits = torch.t(dof_limits[0].to(device=self._device))

    def set_default_state(self):
        # Set default joint state
        joint_states = self._robots.get_joints_default_state()
        jt_pos = joint_states.positions
        jt_pos[:, self.torso_dof_idx] = self.torso_start
        jt_pos[:, self.base_dof_idxs] = self.base_start
        jt_pos[:, self.arm_dof_idxs] = self.arm_start
        jt_pos[:, self.gripper_proximal_dof_idxs] = self.gripper_proximal_start

        jt_vel = joint_states.velocities
        jt_vel[:, self.torso_dof_idx] = torch.zeros_like(self.torso_start, device=self._device)
        jt_vel[:, self.base_dof_idxs] = torch.zeros_like(self.base_start, device=self._device)
        jt_vel[:, self.arm_dof_idxs] = torch.zeros_like(self.arm_start, device=self._device)
        jt_vel[:, self.gripper_proximal_dof_idxs] = torch.zeros_like(self.gripper_proximal_start, device=self._device)

        self._robots.set_joints_default_state(positions=jt_pos, velocities=jt_vel)

        # Initialize target positions
        self.dof_position_targets = jt_pos

    def set_joint_gains(self):
        self._robots.set_gains(kps=self.joint_kps, kds=self.joint_kds)

    def set_joint_frictions(self):
        self._robots.set_friction_coefficients(self.joint_friction_coefficients)

    def _close_gripper(self, env_ids, sim_steps=10):
        env_ids_32 = env_ids.type(torch.int32)
        env_ids_64 = env_ids.type(torch.int64)

        gripper_dof_pos = torch.tensor([-30., -30.], device=self._device)

        # Step sim
        for _ in range(sim_steps):
            self._robots.set_joint_efforts(gripper_dof_pos, indices=env_ids_32, joint_indices=self.gripper_proximal_dof_idxs)
            SimulationContext.step(self._env._world, render=True)

        self.dof_position_targets[env_ids_64[:, None], self.gripper_proximal_dof_idxs] = self._robots.get_joint_positions(indices=env_ids_32, joint_indices=self.gripper_proximal_dof_idxs)
        self.gripper_hold[env_ids_64] = True

    def _open_gripper(self, env_ids, sim_steps=10):
        env_ids_32 = env_ids.type(torch.int32)
        env_ids_64 = env_ids.type(torch.int64)

        gripper_dof_effort = torch.tensor([0., 0.], device=self._device)
        self._robots.set_joint_efforts(gripper_dof_effort, indices=env_ids_32, joint_indices=self.gripper_proximal_dof_idxs)

        gripper_dof_pos = torch.tensor([0.5, 0.5], device=self._device)

        # Step sim
        for _ in range(sim_steps):
            self._robots.set_joint_position_targets(gripper_dof_pos, indices=env_ids_32, joint_indices=self.gripper_proximal_dof_idxs)
            SimulationContext.step(self._env._world, render=True)

        self.dof_position_targets[env_ids_64[:, None], self.gripper_proximal_dof_idxs] = self._robots.get_joint_positions(indices=env_ids_32, joint_indices=self.gripper_proximal_dof_idxs)
        self.gripper_hold[env_ids_64] = False

    def _hold_gripper(self, env_ids):
        env_ids_32 = env_ids.type(torch.int32)
        env_ids_64 = env_ids.type(torch.int64)

        gripper_dof_pos = torch.tensor([-30., -30.], device=self._device)
        self._robots.set_joint_efforts(gripper_dof_pos, indices=env_ids_32, joint_indices=self.gripper_proximal_dof_idxs)
        self.dof_position_targets[env_ids_64[:, None], self.gripper_proximal_dof_idxs] = self._robots.get_joint_positions(indices=env_ids_32, joint_indices=self.gripper_proximal_dof_idxs)

    def _check_lift_success(self, height_threashold):
        prop_pos, _ = self._props.get_world_poses()
        prop_pos -= self._env_pos

        # check z direction range
        lift_success = torch.where(
            prop_pos[:, 2] > height_threashold,
            torch.ones((self.num_envs,), device=self._device),
            torch.zeros((self.num_envs,), device=self._device))

        return lift_success

    def _check_place_success(self, place_threashold):
        table_pos = torch.tensor([1.5, -0.5, self._table_height/2], device=self._device)
        prop_pos, _ = self._props.get_world_poses()
        prop_pos -= self._env_pos

        # check x direction range
        place_success = torch.where(
            prop_pos[:, 0] > table_pos[0]-place_threashold[0],
            torch.ones((self.num_envs,), device=self._device),
            torch.zeros((self.num_envs,), device=self._device)
        )
        place_success = torch.where(
            prop_pos[:, 0] < table_pos[0]+place_threashold[1],
            place_success,
            torch.zeros((self.num_envs,), device=self._device)
        )
        # check y direction range
        place_success = torch.where(
            prop_pos[:, 1] > table_pos[1]-place_threashold[1],
            place_success,
            torch.zeros((self.num_envs,), device=self._device)
        )
        place_success = torch.where(
            prop_pos[:, 1] < table_pos[1]+place_threashold[1],
            place_success,
            torch.zeros((self.num_envs,), device=self._device)
        )

        return place_success

    def _check_robot_collisions(self):
        # Check if the robot collided with an object
        for obst_prim in self._tables._prim_paths:
            match = re.search(r'\d+', obst_prim)
            env_id = int(match.group())
            raw_readings = self._contact_sensor_interface.get_contact_sensor_raw_data(obst_prim + "/Contact_Sensor")
            if raw_readings.shape[0]:
                for reading in raw_readings:
                    if "hsrb" in str(self._contact_sensor_interface.decode_body_name(reading["body1"])):
                        self.is_collided[env_id] = True
                    if "hsrb" in str(self._contact_sensor_interface.decode_body_name(reading["body0"])):
                        self.is_collided[env_id] = True

        collide_penalty = torch.where(
            self.is_collided == True,
            torch.ones((self.num_envs,), device=self._device),
            torch.zeros((self.num_envs,), device=self._device))

        return collide_penalty
    
    def load_exp_dataset(self):
        exp_actions = load_dataset('stacking_problem')
        return torch.tensor(exp_actions, device=self._device) # (dataset_length, num_actions)