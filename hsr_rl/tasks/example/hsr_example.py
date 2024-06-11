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
from hsr_rl.handlers.hsr_handler import HSRHandler

# Whole Body example task with holonomic robot base
class HSRExampleTask(RLTask):
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
        self._gamma = self._task_cfg["env"]["gamma"]
        self._max_episode_length = self._task_cfg["env"]["horizon"]
        self._randomize_robot_on_reset = self._task_cfg["env"]["randomize_robot_on_reset"]

        # Choose num_obs and num_actions based on task
        # (5 arm dofs pos/vel + 3 holo base pos/vel + 2 gripper pos/vel) * 2 = 10 * 2 = 20
        self._num_observations = self._task_cfg["env"]["num_observations"]
        self._move_group = self._task_cfg["env"]["move_group"]
        self._use_gripper = self._task_cfg["env"]["use_gripper"]

        # Position control. Actions are arm (5), base (3) and gripper (2) positions
        self._num_actions = self._task_cfg["env"]["num_actions"]

        # env specific velocity limits
        self.max_arm_vel = torch.tensor(self._task_cfg["env"]["max_rot_vel"], device=self._device)
        self.max_base_xy_vel = torch.tensor(self._task_cfg["env"]["max_base_xy_vel"], device=self._device)
        self.max_base_rz_vel = torch.tensor(self._task_cfg["env"]["max_base_rz_vel"], device=self._device)

        # Handler for HSR
        self.hsr_handler = HSRHandler(move_group=self._move_group, use_gripper=self._use_gripper, sim_config=self._sim_config, num_envs=self._num_envs, device=self._device)

        RLTask.__init__(self, name, env)

    def set_up_scene(self, scene) -> None:
        self.hsr_handler.import_robot_assets()
        super().set_up_scene(scene)
        self._robots = self.hsr_handler.create_articulation_view()
        scene.add(self._robots)

        # Optional viewport for rendering in a separate viewer
        import omni.kit
        from omni.isaac.synthetic_utils import SyntheticDataHelper
        self.viewport_window = omni.kit.viewport.utility.get_active_viewport()
        self.sd_helper = SyntheticDataHelper()
        self.sd_helper.initialize(sensor_names=["rgb"], viewport_api=self.viewport_window)

    def post_reset(self):
        # reset that takes place when the isaac world is reset (typically happens only once)
        self.hsr_handler.post_reset()

    def get_observations(self):
        # Get robot data: joint positions and velocities
        robot_pos_vels = self.hsr_handler.get_robot_obs()

        self.obs_buf = robot_pos_vels

        return self.obs_buf

    def get_render(self):
        # Get ground truth viewport rgb image
        gt = self.sd_helper.get_groundtruth(
            ["rgb"], self.viewport_window, verify_sensor_init=False, wait_for_sensor_data=0
        )
        return gt["rgb"][:, :, :3]
    
    def pre_physics_step(self, actions) -> None:
        if not self._env._world.is_playing():
            return

        # actions (num_envs, num_action)
        # Handle resets
        reset_env_ids = self.reset_buf.nonzero(as_tuple=False).squeeze(-1)
        if len(reset_env_ids) > 0:
            self.reset_idx(reset_env_ids)
            return

        # Scale and clip the actions (positions) before sending to robot
        actions = torch.clamp(actions, -1, 1)

        self.actions = actions.clone().to(self._device)

        self.hsr_handler.apply_actions(self.actions)

    def reset_idx(self, env_ids):
        # apply resets
        indices = env_ids.to(dtype=torch.int32)

        # reset dof values
        self.hsr_handler.reset(indices, randomize=self._randomize_robot_on_reset)

        # bookkeeping
        self.reset_buf[env_ids] = 0
        self.progress_buf[env_ids] = 0
        self.extras[env_ids] = 0

    def calculate_metrics(self) -> None:
        # data from obs buffer is available (get_observations() called before this function)
        test = self.obs_buf[:, 0]
        wp = self._robots.get_world_poses() # gets only root prim poses
        reward = torch.abs(test)

        self.rew_buf[:] = reward

    def is_done(self) -> None:
        resets = torch.zeros(self._num_envs, dtype=int, device=self._device)
        self.reset_buf[:] = resets
