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

import torch
from hsr_rl.tasks.base.rl_task import RLTask
from hsr_rl.robots.articulations.hsr import HSR
from hsr_rl.robots.articulations.views.hsr_view import HSRView
from omni.isaac.core.utils.prims import get_prim_at_path
from omni.isaac.core.utils.stage import get_current_stage
from omni.isaac.core.prims.rigid_prim_view import RigidPrimView
from omni.isaac.core.articulations.articulation_view import ArticulationView
from omni.isaac.core.utils.torch.maths import *
from omni.isaac.core.objects import DynamicSphere

# Whole Body example task with holonomic robot base
class HSRExampleReachTask(RLTask):
    def __init__(
        self,
        name,
        sim_config,
        env
    ) -> None:
        self._sim_config = sim_config
        self._cfg = sim_config.config
        self._task_cfg = sim_config.task_config
        self._device = self._cfg["sim_device"]
        self._num_envs = self._task_cfg["env"]["numEnvs"]
        self._env_spacing = self._task_cfg["env"]["envSpacing"]

        self._dt = torch.tensor(self._sim_config.task_config["sim"]["dt"] * self._sim_config.task_config["env"]["controlFrequencyInv"], device=self._device)

        self._num_observations = self._task_cfg["env"]["num_observations"]
        self._num_actions = self._task_cfg["env"]["num_actions"]

        self._hsr_position = torch.tensor([0.0, 0.0, 0.03], device=self._device)

        # ball properties
        self._ball_position = torch.tensor([1.5, 0.0, 0.045], device=self._device)
        self._ball_radius = torch.tensor([0.05], device=self._device)

        self._action_speed_scale = self._task_cfg["env"]["actionSpeedScale"]
        self._max_episode_length = self._task_cfg["env"]["maxEpisodeLength"]

        # Start at 'home' positions
        self.torso_start = torch.tensor([0.1], device=self._device)
        self.base_start = torch.tensor([0.0, 0.0, 0.0], device=self._device)
        self.arm_start = torch.tensor([0.1, -1.570796, 0.0, -0.392699, 0], device=self._device)
        self.gripper_proximal_start = torch.tensor([0.75, 0.75], device=self._device) # Opened gripper by default

        # joint & body names
        self._torso_joint_name = ["torso_lift_joint"]
        self._base_joint_names = ["joint_x", "joint_y", "joint_rz"]
        self._arm_names = ["arm_lift_joint", "arm_flex_joint", "arm_roll_joint", "wrist_flex_joint", "wrist_roll_joint"]
        self._gripper_proximal_names = ["hand_l_proximal_joint", "hand_r_proximal_joint"]

        # values are set in post_reset after model is loaded
        self.torso_dof_idx = []
        self.base_dof_idxs = []
        self.arm_dof_idxs = []
        self.gripper_proximal_dof_idxs = []

        # dof joint position limits
        self.torso_dof_lower = []
        self.torso_dof_upper = []
        self.base_dof_lower = []
        self.base_dof_upper = []
        self.arm_dof_lower = []
        self.arm_dof_upper = []
        self.gripper_p_dof_lower = []
        self.gripper_p_dof_upper = []

        RLTask.__init__(self, name, env)
        return

    def set_up_scene(self, scene) -> None:
        self.add_hsr()
        self.add_ball()

        # Set up scene
        super().set_up_scene(scene)

        # Add robot to scene
        self._robots = HSRView(prim_paths_expr="/World/envs/.*/hsrb", name="hsrb_view") # ArticulationView(prim_paths_expr="/World/envs/.*/hsrb", name="hsrb_view", reset_xform_properties=False)
        scene.add(self._robots)
        scene.add(self._robots._hands)
        scene.add(self._robots._lfingers)
        scene.add(self._robots._rfingers)
        scene.add(self._robots._fingertip_centered)

        # Add ball to scene
        self._balls = RigidPrimView(prim_paths_expr="/World/envs/.*/Ball/ball", name="ball_view", reset_xform_properties=False)
        scene.add(self._balls)
        return

    def add_hsr(self):
        hsrb = HSR(prim_path=self.default_zero_env_path + "/hsrb", name="hsrb", translation=self._hsr_position)
        self._sim_config.apply_articulation_settings("hsrb", get_prim_at_path(hsrb.prim_path), self._sim_config.parse_actor_config("hsrb"))

    def add_ball(self):
        ball = DynamicSphere(
            prim_path=self.default_zero_env_path + "/Ball/ball",
            translation=self._ball_position,
            name="ball_0",
            radius=0.05,
            color=torch.tensor([0.2, 0.4, 0.6]),
        )
        self._sim_config.apply_articulation_settings("ball", get_prim_at_path(ball.prim_path), self._sim_config.parse_actor_config("ball"))

    def get_observations(self):
        # Get ball positions and orientations
        ball_positions, ball_orientations = self._balls.get_world_poses(clone=False)
        ball_positions = ball_positions[:, 0:3] - self._env_pos

        # Get ball velocities
        ball_velocities = self._balls.get_velocities(clone=False)
        ball_linvels = ball_velocities[:, 0:3]
        ball_angvels = ball_velocities[:, 3:6]

        # Get end effector positions and orientations
        end_effector_positions, end_effector_orientations = self._robots._fingertip_centered.get_world_poses(clone=False)
        end_effector_positions = end_effector_positions[:, 0:3] - self._env_pos

        # Get end effector velocities
        end_effector_velocities = self._robots._fingertip_centered.get_velocities(clone=False)
        end_effector_linvels = end_effector_velocities[:, 0:3]
        end_effector_angvels = end_effector_velocities[:, 3:6]

        dof_pos = self._robots.get_joint_positions(clone=False)
        dof_vel = self._robots.get_joint_velocities(clone=False)

        self.obs_buf[..., 0:8] = dof_pos[..., self.actuated_dof_indices]
        self.obs_buf[..., 8:16] = dof_vel[..., self.actuated_dof_indices]
        self.obs_buf[..., 16:19] = end_effector_positions
        self.obs_buf[..., 19:22] = end_effector_linvels
        self.obs_buf[..., 22:25] = ball_positions
        self.obs_buf[..., 25:28] = ball_linvels

        self.ball_positions = ball_positions
        self.ball_linvels = ball_linvels

        observations = {
            "hsr_reach": {
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

        # update position targets from actions
        self.dof_position_targets[..., self.actuated_dof_indices] += self._dt * self._action_speed_scale * actions.to(self.device)
        self.dof_position_targets[:] = tensor_clamp(
            self.dof_position_targets, self.robot_dof_lower_limits, self.robot_dof_upper_limits
        )

        dof_pos = self._robots.get_joint_positions()
        arm_pos = dof_pos[:, self.arm_dof_idxs]
        scaled_arm_lift_pos = arm_pos[:, 0] / self.arm_dof_upper[0]
        scaled_torso_lift_pos = scaled_arm_lift_pos * self.torso_dof_upper[0]
        self.dof_position_targets[:, self.torso_dof_idx] = scaled_torso_lift_pos.unsqueeze(dim=1)

        # reset position targets for reset envs
        self.dof_position_targets[reset_env_ids] = 0

        self._robots.set_joint_positions(self.dof_position_targets)

    def reset_idx(self, env_ids):
        num_resets = len(env_ids)

        env_ids_32 = env_ids.type(torch.int32)
        env_ids_64 = env_ids.type(torch.int64)

        min_d = 0.001 # min horizontal dist from origin
        max_d = 0.4 # max horizontal dist from origin
        min_height = 0.0
        max_height = 0.2
        min_horizontal_speed = 0
        max_horizontal_speed = 2

        dists = torch_rand_float(min_d, max_d, (num_resets, 1), self._device)
        dirs = torch_random_dir_2((num_resets, 1), self._device)
        hpos = dists * dirs

        speedscales = (dists - min_d) / (max_d - min_d)
        hspeeds = torch_rand_float(min_horizontal_speed, max_horizontal_speed, (num_resets, 1), self._device)
        hvels = -speedscales * hspeeds * dirs
        vspeeds = -torch_rand_float(5.0, 5.0, (num_resets, 1), self._device).squeeze()

        ball_pos = self.initial_ball_pos.clone()
        ball_rot = self.initial_ball_rot.clone()
        # position
        ball_pos[env_ids_64, 0:2] += hpos[..., 0:2]
        ball_pos[env_ids_64, 2] += torch_rand_float(min_height, max_height, (num_resets, 1), self._device).squeeze()
        # rotation
        ball_rot[env_ids_64, 0] = 1
        ball_rot[env_ids_64, 1:] = 0
        ball_velocities = self.initial_ball_velocities.clone()
        # linear
        ball_velocities[env_ids_64, 0:2] = hvels[..., 0:2]
        ball_velocities[env_ids_64, 2] = vspeeds
        # angular
        ball_velocities[env_ids_64, 3:6] = 0

        # reset root state for bbots and balls in selected envs
        self._balls.set_world_poses(ball_pos[env_ids_64], ball_rot[env_ids_64], indices=env_ids_32)
        self._balls.set_velocities(ball_velocities[env_ids_64], indices=env_ids_32)

        # reset root pose and velocity
        self._robots.set_world_poses(self.initial_robot_pos[env_ids_64].clone(), self.initial_robot_rot[env_ids_64].clone(), indices=env_ids_32)
        self._robots.set_velocities(self.initial_robot_velocities[env_ids_64].clone(), indices=env_ids_32)

        # reset DOF states for bbots in selected envs
        self._robots.set_joint_positions(self.initial_dof_positions[env_ids_64].clone(), indices=env_ids_32)

        # bookkeeping
        self.reset_buf[env_ids] = 0
        self.progress_buf[env_ids] = 0
        self.extras[env_ids] = 0

    def post_reset(self):
        self.set_dof_idxs()
        self.set_dof_limits()
        self.set_default_state()

        dof_limits = self._robots.get_dof_limits()
        self.robot_dof_lower_limits, self.robot_dof_upper_limits = torch.t(dof_limits[0].to(device=self._device))

        # reset ball pos / rot, and velocities
        self.initial_dof_positions = self._robots.get_joint_positions()
        self.initial_robot_pos, self.initial_robot_rot = self._robots.get_world_poses()
        self.initial_robot_velocities = self._robots.get_velocities()
        self.initial_ball_pos, self.initial_ball_rot = self._balls.get_world_poses()
        self.initial_ball_velocities = self._balls.get_velocities()

        self.dof_position_targets = self._robots.get_joints_default_state().positions

        self.actuated_dof_indices = torch.LongTensor([0, 1, 2, 3, 5, 7, 9, 10]).to(self._device)

    def calculate_metrics(self) -> None:
        # Distance from hand to the ball
        dist = torch.norm(self.obs_buf[..., 22:25] - self.obs_buf[..., 16:19], p=2, dim=-1)
        dist_reward = 1.0 / (1.0 + dist ** 2)
        dist_reward *= dist_reward
        dist_reward = torch.where(dist <= 0.02, dist_reward * 2, dist_reward)

        arm_velocity_penalty = -torch.norm(self.obs_buf[..., 19:22], p=2, dim=-1)
        ball_velocity_penalty = -torch.norm(self.obs_buf[..., 25:28], p=2, dim=-1)

        self.rew_buf[:] = dist_reward

    def is_done(self) -> None:
        reset = torch.where(
            self.progress_buf >= self._max_episode_length - 1, torch.ones_like(self.reset_buf), self.reset_buf
        )
        reset = torch.where(self.ball_positions[..., 2] < self._ball_radius * 1.5, torch.ones_like(self.reset_buf), reset)
        self.reset_buf[:] = reset

    def set_dof_idxs(self):
        [self.torso_dof_idx.append(self._robots.get_dof_index(name)) for name in self._torso_joint_name]
        [self.base_dof_idxs.append(self._robots.get_dof_index(name)) for name in self._base_joint_names]
        [self.arm_dof_idxs.append(self._robots.get_dof_index(name)) for name in self._arm_names]
        [self.gripper_proximal_dof_idxs.append(self._robots.get_dof_index(name)) for name in self._gripper_proximal_names]

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