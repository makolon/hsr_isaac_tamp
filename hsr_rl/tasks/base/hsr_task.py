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
from hsr_rl.robots.articulations.hsr import HSR
from hsr_rl.robots.articulations.views.hsr_view import HSRView
from hsr_rl.tasks.base.rl_task import RLTask
from hsr_rl.tasks.utils.ik_utils import DifferentialInverseKinematics, DifferentialInverseKinematicsCfg

from omni.isaac.core.utils.prims import get_prim_at_path
from omni.isaac.core.utils.stage import get_current_stage
from omni.isaac.core.utils.torch.rotations import quat_diff_rad, quat_mul, normalize
from omni.isaac.core.objects import FixedCuboid
from omni.isaac.core.simulation_context import SimulationContext
from omni.physx.scripts import utils, physicsUtils


class HSRBaseTask(RLTask):
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

        # Get dt for integrating velocity commands and checking limit violations
        self._dt = torch.tensor(self._sim_config.task_config["sim"]["dt"] * self._sim_config.task_config["env"]["controlFrequencyInv"], device=self._device)

        # Set environment properties
        self._table_height = self._task_cfg["env"]["table_height"]
        self._table_width = self._task_cfg["env"]["table_width"]
        self._table_depth = self._task_cfg["env"]["table_depth"]

        # Set physics parameters for gripper
        self._gripper_mass = self._sim_config.task_config["sim"]["gripper"]["mass"]
        self._gripper_density = self._sim_config.task_config["sim"]["gripper"]["density"]
        self._gripper_static_friction = self._sim_config.task_config["sim"]["gripper"]["static_friction"]
        self._gripper_dynamic_friction = self._sim_config.task_config["sim"]["gripper"]["dynamic_friction"]
        self._gripper_restitution = self._sim_config.task_config["sim"]["gripper"]["restitution"]

        # Choose num_obs and num_actions based on task.
        self._num_observations = self._task_cfg["env"]["num_observations"]
        self._num_actions = self._task_cfg["env"]["num_actions"]

        # Set inverse kinematics configurations
        self._action_type = self._task_cfg["env"]["action_type"]
        self._target_space = self._task_cfg["env"]["target_space"]

        # Set up environment from loaded demonstration
        self._hsr_position = torch.tensor([0.0, 0.0, 0.01], device=self._device)
        self._hsr_rotation = torch.tensor([1.0, 0.0, 0.0, 0.0], device=self._device)
        self._table_position = torch.tensor([1.2, -0.2, self._table_height/2], device=self._device)
        self._table_rotation = torch.tensor([1.0, 0.0, 0.0, 0.0], device=self._device)

        # Dof joint gains
        self.joint_kps = torch.tensor([1e9, 1e9, 5.7296e10, 1e9, 1e9, 5.7296e10, 5.7296e10, 5.7296e10,
            5.7296e10, 5.7296e10, 5.7296e10, 2.8648e4, 2.8648e4, 5.7296e10, 5.7296e10], device=self._device)
        self.joint_kds = torch.tensor([1.4, 1.4, 80.2141, 1.4, 0.0, 80.2141, 0.0, 80.2141, 0.0, 80.2141,
            80.2141, 17.1887, 17.1887, 17.1887, 17.1887], device=self._device)

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
        self.gripper_proximal_dof_lower = []
        self.gripper_proximal_dof_upper = []

        # Gripper settings
        self.gripper_close = torch.zeros(self._num_envs, device=self._device, dtype=torch.bool)
        self.gripper_open = torch.zeros(self._num_envs, device=self._device, dtype=torch.bool)
        self.gripper_hold = torch.zeros(self._num_envs, device=self._device, dtype=torch.bool)

        # Load ik controller
        self.ik_controller = self.set_ik_controller()

        RLTask.__init__(self, name, env)
        return
    
    def set_up_environment(self) -> None:
        raise NotImplementedError

    def set_up_scene(self, scene) -> None:
        # Create gripper materials
        self.create_gripper_material()

        # Import before environment parallelization
        self.add_hsr()
        self.add_table()

        # Set up scene
        super().set_up_scene(scene, replicate_physics=False)

        # Add robot to scene
        self._robots = HSRView(prim_paths_expr="/World/envs/.*/hsrb", name="hsrb_view")
        scene.add(self._robots)
        scene.add(self._robots._hands)
        scene.add(self._robots._lfingers)
        scene.add(self._robots._rfingers)
        scene.add(self._robots._fingertip_centered)

    def add_hsr(self):
        # Add HSR
        hsrb = HSR(prim_path=self.default_zero_env_path + "/hsrb",
                   name="hsrb",
                   translation=self._hsr_position,
                   orientation=self._hsr_rotation)
        self._sim_config.apply_articulation_settings("hsrb", get_prim_at_path(hsrb.prim_path), self._sim_config.parse_actor_config("hsrb"))

    def add_table(self):
        # Add table
        table = FixedCuboid(prim_path=self.default_zero_env_path + "/table",
                            name="table",
                            translation=self._table_position,
                            orientation=self._table_rotation,
                            size=self._table_size,
                            color=torch.tensor([0.75, 0.75, 0.75]),
                            scale=torch.tensor([self._table_width, self._table_depth, self._table_height]))
        self._sim_config.apply_articulation_settings("table", get_prim_at_path(table.prim_path), self._sim_config.parse_actor_config("table"))

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

    def get_observations(self):
        raise NotImplementedError()

    def pre_physics_step(self, actions) -> None:
        raise NotImplementedError()

    def reset_idx(self, env_ids):
        raise NotImplementedError()

    def post_reset(self):
        raise NotImplementedError()

    def calculate_metrics(self) -> None:
        raise NotImplementedError()

    def is_done(self) -> None:
        raise NotImplementedError()

    def post_physics_step(self):
        raise NotImplementedError()

    def load_dataset(self):
        raise NotImplementedError()

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
        self.gripper_proximal_dof_lower = dof_limits_lower[self.gripper_proximal_dof_idxs]
        self.gripper_proximal_dof_upper = dof_limits_upper[self.gripper_proximal_dof_idxs]

        self.robot_dof_lower_limits, self.robot_dof_upper_limits = torch.t(dof_limits[0].to(device=self._device))

    def set_default_state(self):
        # Start at 'home' positions
        self.torso_start = self.initial_dof_positions[:, self.torso_dof_idx]
        self.base_start = self.initial_dof_positions[:, self.base_dof_idxs]
        self.arm_start = self.initial_dof_positions[:, self.arm_dof_idxs]
        self.gripper_proximal_start = self.initial_dof_positions[:, self.gripper_proximal_dof_idxs]

        # Set default joint state
        joint_states = self._robots.get_joints_default_state()
        jt_pos = joint_states.positions
        jt_pos[:, self.torso_dof_idx] = self.torso_start.float()
        jt_pos[:, self.base_dof_idxs] = self.base_start.float()
        jt_pos[:, self.arm_dof_idxs] = self.arm_start.float()
        jt_pos[:, self.gripper_proximal_dof_idxs] = self.gripper_proximal_start.float()

        jt_vel = joint_states.velocities
        jt_vel[:, self.torso_dof_idx] = torch.zeros_like(self.torso_start, device=self._device, dtype=torch.float)
        jt_vel[:, self.base_dof_idxs] = torch.zeros_like(self.base_start, device=self._device, dtype=torch.float)
        jt_vel[:, self.arm_dof_idxs] = torch.zeros_like(self.arm_start, device=self._device, dtype=torch.float)
        jt_vel[:, self.gripper_proximal_dof_idxs] = torch.zeros_like(self.gripper_proximal_start, device=self._device, dtype=torch.float)

        self._robots.set_joints_default_state(positions=jt_pos, velocities=jt_vel)

        # Initialize target positions
        self.dof_position_targets = jt_pos

    def set_joint_gains(self):
        self._robots.set_gains(kps=self.joint_kps, kds=self.joint_kds)

    def set_joint_frictions(self):
        self._robots.set_friction_coefficients(self.joint_friction_coefficients)

    def _close_gripper(self, env_ids, sim_steps=1):
        env_ids_32 = env_ids.type(torch.int32)
        env_ids_64 = env_ids.type(torch.int64)

        gripper_dof_pos = torch.tensor([-50., -50.], device=self._device)

        # Step sim
        for _ in range(sim_steps):
            self._robots.set_joint_efforts(gripper_dof_pos, indices=env_ids_32, joint_indices=self.gripper_proximal_dof_idxs)
            SimulationContext.step(self._env._world, render=True)

        self.dof_position_targets[env_ids_64[:, None], self.gripper_proximal_dof_idxs] = self._robots.get_joint_positions(indices=env_ids_32, joint_indices=self.gripper_proximal_dof_idxs)
        self.gripper_hold[env_ids_64] = True

    def _open_gripper(self, env_ids, sim_steps=1):
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

    def set_ik_controller(self):
        command_type = "pose_rel" if self._action_type == 'relative' else "pose_abs"

        ik_control_cfg = DifferentialInverseKinematicsCfg(
            command_type=command_type,
            ik_method="dls",
            position_offset=(0.0, 0.0, 0.0),
            rotation_offset=(1.0, 0.0, 0.0, 0.0),
        )
        return DifferentialInverseKinematics(ik_control_cfg, self._num_envs, self._device)

    def enable_gravity(self):
        """Enable gravity."""

        gravity = [0.0, 0.0, -9.81]
        self._env._world._physics_sim_view.set_gravity(carb.Float3(gravity[0], gravity[1], gravity[2]))

    def disable_gravity(self):
        """Disable gravity."""

        gravity = [0.0, 0.0, 0.0]
        self._env._world._physics_sim_view.set_gravity(carb.Float3(gravity[0], gravity[1], gravity[2]))


@torch.jit.script
def norm_diff_pos(p1: torch.Tensor, p2: torch.Tensor) -> torch.Tensor:
    # Calculate norm
    diff_norm = torch.norm(p1 - p2, p=2, dim=-1)

    return diff_norm

@torch.jit.script
def norm_diff_rot(q1: torch.Tensor, q2: torch.Tensor) -> torch.Tensor:
    # Calculate norm
    diff_norm1 = torch.norm(q1 - q2, p=2, dim=-1)
    diff_norm2 = torch.norm(q2 - q1, p=2, dim=-1)

    diff_norm = torch.min(diff_norm1, diff_norm2)

    return diff_norm

@torch.jit.script
def quaternion_multiply(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    aw, ax, ay, az = torch.unbind(a, -1)
    bw, bx, by, bz = torch.unbind(b, -1)
    ow = aw * bw - ax * bx - ay * by - az * bz
    ox = aw * bx + ax * bw + ay * bz - az * by
    oy = aw * by - ax * bz + ay * bw + az * bx
    oz = aw * bz + ax * by - ay * bx + az * bw
    return torch.stack((ow, ox, oy, oz), -1)

@torch.jit.script
def calc_diff_pos(p1, p2):
    return p1 - p2

@torch.jit.script
def calc_diff_rot(q1, q2):
    # Normalize the input quaternions
    q1 = normalize(q1)
    q2 = normalize(q2)

    # Calculate the quaternion product between q2 and the inverse of q1
    scaling = torch.tensor([1, -1, -1, -1], device=q1.device)
    q1_inv = q1 * scaling
    q_diff = quaternion_multiply(q2, q1_inv)

    return q_diff