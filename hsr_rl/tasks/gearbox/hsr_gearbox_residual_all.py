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
from hsr_rl.robots.articulations.hsr import HSR
from hsr_rl.robots.articulations.views.hsr_view import HSRView
from hsr_rl.tasks.base.rl_task import RLTask
from hsr_rl.tasks.utils.scene_utils import spawn_dynamic_object, spawn_static_object
from hsr_rl.tasks.utils.ik_utils import DifferentialInverseKinematics, DifferentialInverseKinematicsCfg

from omni.isaac.core.prims import RigidPrimView
from omni.isaac.core.utils.prims import get_prim_at_path, delete_prim, is_prim_path_valid
from omni.isaac.core.utils.stage import get_current_stage
from omni.isaac.core.utils.torch.maths import torch_rand_float, tensor_clamp
from omni.isaac.core.utils.torch.rotations import quat_diff_rad, quat_mul, normalize
from omni.isaac.core.objects import FixedCuboid
from omni.isaac.core.simulation_context import SimulationContext
from omni.physx.scripts import utils, physicsUtils
from pxr import Gf, Sdf, UsdGeom, PhysxSchema, UsdPhysics


class HSRGearboxResidualAllTask(RLTask):
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
        self._num_envs = self._cfg["num_envs"]
        self._env_spacing = self._task_cfg["env"]["envSpacing"]

        # Get dt for integrating velocity commands and checking limit violations
        self._control_frequency = torch.tensor(1/self._sim_config.task_config["env"]["controlFrequencyInv"], device=self._device)
        self._dt = torch.tensor(self._sim_config.task_config["sim"]["dt"] * self._sim_config.task_config["env"]["controlFrequencyInv"], device=self._device)

        # Set environment properties
        self._table_height = self._task_cfg["env"]["table_height"]
        self._table_width = self._task_cfg["env"]["table_width"]
        self._table_depth = self._task_cfg["env"]["table_depth"]
        self._base_height = self._task_cfg["env"]["base_height"]
        self._base_width = self._task_cfg["env"]["base_width"]
        self._base_depth = self._task_cfg["env"]["base_depth"]
        self._gear_base_height = self._task_cfg["env"]["gear_base_height"]
        self._gear_base_width = self._task_cfg["env"]["gear_base_width"]
        self._gear_base_depth = self._task_cfg["env"]["gear_base_depth"]

        # Set physics parameters for gearbox parts
        self._gear_mass = self._sim_config.task_config["sim"]["gear_parts"]["mass"]
        self._gear_density = self._sim_config.task_config["sim"]["gear_parts"]["density"]
        self._gear_static_friction = self._sim_config.task_config["sim"]["gear_parts"]["static_friction"]
        self._gear_dynamic_friction = self._sim_config.task_config["sim"]["gear_parts"]["dynamic_friction"]
        self._gear_restitution = self._sim_config.task_config["sim"]["gear_parts"]["restitution"]
        self._shaft_mass = self._sim_config.task_config["sim"]["shaft_parts"]["mass"]
        self._shaft_density = self._sim_config.task_config["sim"]["shaft_parts"]["density"]
        self._shaft_static_friction = self._sim_config.task_config["sim"]["shaft_parts"]["static_friction"]
        self._shaft_dynamic_friction = self._sim_config.task_config["sim"]["shaft_parts"]["dynamic_friction"]
        self._shaft_restitution = self._sim_config.task_config["sim"]["shaft_parts"]["restitution"]

        # Set physics parameters for gripper
        self._gripper_mass = self._sim_config.task_config["sim"]["gripper"]["mass"]
        self._gripper_density = self._sim_config.task_config["sim"]["gripper"]["density"]
        self._gripper_static_friction = self._sim_config.task_config["sim"]["gripper"]["static_friction"]
        self._gripper_dynamic_friction = self._sim_config.task_config["sim"]["gripper"]["dynamic_friction"]
        self._gripper_restitution = self._sim_config.task_config["sim"]["gripper"]["restitution"]

        # Choose num_obs and num_actions based on task.
        self._num_observations = self._task_cfg["env"]["num_observations"]
        self._num_actions = self._task_cfg["env"]["num_actions"]
        self._num_objects = self._task_cfg["env"]["num_objects"]

        # Set object names
        self._obj_names = ['gear1', 'gear2', 'gear3', 'shaft1', 'shaft2']

        # Set inverse kinematics configurations
        self._action_type = self._task_cfg["env"]["action_type"]
        self._target_space = self._task_cfg["env"]["target_space"]

        # Set learning hyper-parameters
        self._action_speed_scale = self._task_cfg["env"]["actionSpeedScale"]
        self._max_episode_length = self._task_cfg["env"]["maxEpisodeLength"]

        # Set up environment from loaded demonstration
        self.set_up_environment()

        # Dof joint gains
        self.joint_kps = torch.tensor([1e9, 1e9, 5.7296e10, 1e9, 1e9, 5.7296e10, 5.7296e10, 5.7296e10,
            5.7296e10, 5.7296e10, 5.7296e10, 2.8648e4, 2.8648e4, 5.7296e10, 5.7296e10], device=self._device)
        self.joint_kds = torch.tensor([1.4, 1.4, 80.2141, 1.4, 0.0, 80.2141, 0.0, 80.2141, 0.0, 80.2141,
            80.2141, 17.1887, 17.1887, 17.1887, 17.1887], device=self._device)

        # Dof joint friction coefficients
        self.joint_friction_coefficients = torch.tensor([1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0], device=self._device)

        # Start at 'home' positions
        self.torso_start = torch.tensor([0.1], device=self._device)
        self.base_start = torch.tensor([0.0, 0.0, 0.0], device=self._device)
        self.arm_start = torch.tensor([0.1, -1.570796, 0.0, 0.0, 0.0], device=self._device)
        self.gripper_proximal_start = torch.tensor([0.75, 0.75], device=self._device)

        self.initial_dof_positions = torch.tensor([[0.0, 0.0, 0.0, 0.1, 0.1, -1.570796, 0.0, 0.0, 0.0, 0.0, 0.0, 0.5, 0.5, 0.0, 0.0]], device=self._device)
        self.initial_dof_velocities = torch.zeros_like(self.initial_dof_positions, device=self._device, dtype=torch.float)

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

        # Reward settings
        self.pick_success = torch.zeros(1, device=self._device, dtype=torch.long)
        self.place_success = torch.zeros(1, device=self._device, dtype=torch.long)
        self.insert_success = torch.zeros(1, device=self._device, dtype=torch.long)

        # Gripper settings
        self.gripper_close = torch.zeros(1, device=self._device, dtype=torch.bool)
        self.gripper_open = torch.zeros(1, device=self._device, dtype=torch.bool)
        self.gripper_hold = torch.zeros(1, device=self._device, dtype=torch.bool)

        RLTask.__init__(self, name, env)
        return

    def set_up_environment(self) -> None:
        # Environment object settings: (reset() randomizes the environment)
        self._hsr_position = torch.tensor([0.0, 0.0, 0.01], device=self._device)
        self._hsr_rotation = torch.tensor([1.0, 0.0, 0.0, 0.0], device=self._device)
        self._table_position = torch.tensor([1.2, -0.2, self._table_height/2], device=self._device)
        self._table_rotation = torch.tensor([1.0, 0.0, 0.0, 0.0], device=self._device)
        self._gear_base_position = torch.tensor([1.2, 0.0, self._table_height+self._gear_base_height/2], device=self._device)
        self._gear_base_rotation = torch.tensor([0.92387953, 0.0, 0.0, -0.38268343], device=self._device)
        self._base_positions = torch.tensor([[1.2, 0.30, self._table_height+self._base_height/2],
                                             [1.2, -0.50, self._table_height+self._base_height/2],
                                             [1.2, -0.75, self._table_height+self._base_height/2]], device=self._device)
        self._base_rotations = torch.tensor([[1.0, 0.0, 0.0, 0.0],
                                             [1.0, 0.0, 0.0, 0.0],
                                             [1.0, 0.0, 0.0, 0.0]], device=self._device)

        self.init_parts_pos = {
            'gear1': torch.tensor([1.2, 0.30, self._table_height+self._base_height], device=self._device),
            'gear2': torch.tensor([1.2, -0.50, self._table_height+self._base_height], device=self._device),
            'gear3': torch.tensor([1.2, -0.75, self._table_height+self._base_height], device=self._device),
            'shaft1': torch.tensor([1.2, 0.50, self._table_height+0.115], device=self._device),
            'shaft2': torch.tensor([1.2, -0.30, self._table_height+0.115], device=self._device),
            'gearbox_base': torch.tensor([1.1, 0.0, self._table_height+self._gear_base_height], device=self._device),
            'gear_base': self._gear_base_position,
            'table': self._table_position,
            'box_1': self._base_positions[0],
            'box_2': self._base_positions[1],
            'box_3': self._base_positions[2]}
        self.init_parts_orn = {
            'gear1': torch.tensor([1.0, 0.0, 0.0, 0.0], device=self._device),
            'gear2': torch.tensor([1.0, 0.0, 0.0, 0.0], device=self._device),
            'gear3': torch.tensor([1.0, 0.0, 0.0, 0.0], device=self._device),
            'shaft1': torch.tensor([0.0, 0.0, 1.0, 0.0], device=self._device),
            'shaft2': torch.tensor([0.0, 0.0, 1.0, 0.0], device=self._device),
            'gearbox_base': torch.tensor([0.92387953, 0.0, 0.0, -0.38268343], device=self._device),
            'gear_base': self._gear_base_rotation,
            'table': self._table_rotation,
            'box_1': self._base_rotations[0],
            'box_2': self._base_rotations[1],
            'box_3': self._base_rotations[2]}

    def set_up_scene(self, scene) -> None:
        # Create gearbox parts materials
        self.create_gearbox_material()

        # Create gripper materials
        self.create_gripper_material()

        # Set up scene
        super().set_up_scene(scene, replicate_physics=False)

        self.add_hsr()
        self.add_base()
        self.add_table()
        self.add_parts()
        self.add_gearbox_base()

        # Add robot to scene
        self._robots = HSRView(prim_paths_expr="/World/envs/.*/hsrb", name="hsrb_view")
        scene.add(self._robots)
        scene.add(self._robots._hands)
        scene.add(self._robots._lfingers)
        scene.add(self._robots._rfingers)
        scene.add(self._robots._fingertip_centered)

        # Add parts to scene
        self._shaft1 = RigidPrimView(prim_paths_expr="/World/envs/.*/shaft1/shaft1", name="shaft1_view", reset_xform_properties=False)
        self._shaft2 = RigidPrimView(prim_paths_expr="/World/envs/.*/shaft2/shaft2", name="shaft2_view", reset_xform_properties=False)
        self._gear1 = RigidPrimView(prim_paths_expr="/World/envs/.*/gear1/gear1", name="gear1_view", reset_xform_properties=False)
        self._gear2 = RigidPrimView(prim_paths_expr="/World/envs/.*/gear2/gear2", name="gear2_view", reset_xform_properties=False)
        self._gear3 = RigidPrimView(prim_paths_expr="/World/envs/.*/gear3/gear3", name="gear3_view", reset_xform_properties=False)

        scene.add(self._shaft1)
        scene.add(self._shaft2)
        scene.add(self._gear1)
        scene.add(self._gear2)
        scene.add(self._gear3)

        self._parts = {'shaft1': self._shaft1, 'shaft2': self._shaft2,
                       'gear1': self._gear1, 'gear2': self._gear2, 'gear3': self._gear3}

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

    def create_gearbox_material(self):
        self._stage = get_current_stage()
        self.gearPhysicsMaterialPath = "/World/Physics_Materials/GearMaterial"
        self.shaftPhysicsMaterialPath = "/World/Physics_Material/ShaftMaterial"

        utils.addRigidBodyMaterial(
            self._stage,
            self.gearPhysicsMaterialPath,
            density=self._gear_density,
            staticFriction=self._gear_static_friction,
            dynamicFriction=self._gear_dynamic_friction,
            restitution=self._gear_restitution
        )

        utils.addRigidBodyMaterial(
            self._stage,
            self.shaftPhysicsMaterialPath,
            density=self._shaft_density,
            staticFriction=self._shaft_static_friction,
            dynamicFriction=self._shaft_dynamic_friction,
            restitution=self._shaft_restitution
        )

    def add_hsr(self):
        # Add HSR
        hsrb = HSR(prim_path=f"/World/envs/env_0/hsrb",
                    name="hsrb",
                    translation=self._hsr_position,
                    orientation=self._hsr_rotation)
        self._sim_config.apply_articulation_settings("hsrb", get_prim_at_path(hsrb.prim_path), self._sim_config.parse_actor_config("hsrb"))

        # Add physics material
        physicsUtils.add_physics_material_to_prim(
            self._stage,
            self._stage.GetPrimAtPath(f"/World/envs/env_0/hsrb/hand_r_distal_link/collisions/mesh_0"),
            self.gripperPhysicsMaterialPath
        )
        physicsUtils.add_physics_material_to_prim(
            self._stage,
            self._stage.GetPrimAtPath(f"/World/envs/env_0/hsrb/hand_l_distal_link/collisions/mesh_0"),
            self.gripperPhysicsMaterialPath
        )

    def add_table(self):
        # Add table
        table = FixedCuboid(prim_path=f"/World/envs/env_0/table",
                            name="table",
                            translation=self.init_parts_pos['table'],
                            orientation=self.init_parts_orn['table'],
                            size=1.0,
                            color=torch.tensor([0.75, 0.75, 0.75]),
                            scale=torch.tensor([self._table_width, self._table_depth, self._table_height]))
        self._sim_config.apply_articulation_settings("table", get_prim_at_path(table.prim_path), self._sim_config.parse_actor_config("table"))

    def add_base(self):
        # Add base1
        base_1 = FixedCuboid(prim_path=f"/World/envs/env_0/base_1",
                                name="base_1",
                                translation=self.init_parts_pos['box_1'],
                                orientation=self.init_parts_orn['box_1'],
                                size=1.0,
                                color=torch.tensor([0.1, 0.1, 0.1]),
                                scale=torch.tensor([self._base_width, self._base_depth, self._base_height]))
        self._sim_config.apply_articulation_settings("base", get_prim_at_path(base_1.prim_path), self._sim_config.parse_actor_config("base"))

        # Add base2
        base_2 = FixedCuboid(prim_path=f"/World/envs/env_0/base_2",
                                name="base_2",
                                translation=self.init_parts_pos['box_2'],
                                orientation=self.init_parts_orn['box_2'],
                                size=1.0,
                                color=torch.tensor([0.1, 0.1, 0.1]),
                                scale=torch.tensor([self._base_width, self._base_depth, self._base_height]))
        self._sim_config.apply_articulation_settings("base", get_prim_at_path(base_2.prim_path), self._sim_config.parse_actor_config("base"))

        # Add base3
        base_3 = FixedCuboid(prim_path=f"/World/envs/env_0/base_3",
                                name="base_3",
                                translation=self.init_parts_pos['box_3'],
                                orientation=self.init_parts_orn['box_3'],
                                size=1.0,
                                color=torch.tensor([0.1, 0.1, 0.1]),
                                scale=torch.tensor([self._base_width, self._base_depth, self._base_height]))
        self._sim_config.apply_articulation_settings("base", get_prim_at_path(base_3.prim_path), self._sim_config.parse_actor_config("base"))

        # Add gear base
        gear_base = FixedCuboid(prim_path=f"/World/envs/env_0/gear_base",
                                name="gear_base",
                                translation=self.init_parts_pos['gear_base'],
                                orientation=self.init_parts_orn['gear_base'],
                                size=1.0,
                                color=torch.tensor([0.1, 0.1, 0.1]),
                                scale=torch.tensor([self._gear_base_width, self._gear_base_depth, self._gear_base_height]))
        self._sim_config.apply_articulation_settings("base", get_prim_at_path(gear_base.prim_path), self._sim_config.parse_actor_config("base"))

    def add_gearbox_base(self):
        parts = spawn_static_object(name='gearbox_base',
                                    prim_path=f"/World/envs/env_0",
                                    object_translation=self.init_parts_pos['gearbox_base'],
                                    object_orientation=self.init_parts_orn['gearbox_base'])
        self._sim_config.apply_articulation_settings("gearbox_base",
                                        get_prim_at_path(parts.prim_path),
                                        self._sim_config.parse_actor_config('gearbox_base'))

    def add_parts(self):
        # Add movable gearbox parts
        for i in range(self._num_objects):
            object_translation = self.init_parts_pos[self._obj_names[i]]
            object_orientation = self.init_parts_orn[self._obj_names[i]]
            parts = spawn_dynamic_object(name=self._obj_names[i],
                                            prim_path=f"/World/envs/env_0",
                                            object_translation=object_translation,
                                            object_orientation=object_orientation)
            self._sim_config.apply_articulation_settings(f"{self._obj_names[i]}",
                                            get_prim_at_path(parts.prim_path),
                                            self._sim_config.parse_actor_config(self._obj_names[i]))

            # Add physics material
            if 'gear' in self._obj_names[i]:
                physicsUtils.add_physics_material_to_prim(
                    self._stage,
                    self._stage.GetPrimAtPath(f"/World/envs/env_0/{self._obj_names[i]}/{self._obj_names[i]}/collisions/mesh_0"),
                    self.gearPhysicsMaterialPath
                )
            elif 'shaft' in self._obj_names[i]:
                physicsUtils.add_physics_material_to_prim(
                    self._stage,
                    self._stage.GetPrimAtPath(f"/World/envs/env_0/{self._obj_names[i]}/{self._obj_names[i]}/collisions/mesh_0"),
                    self.shaftPhysicsMaterialPath
                )

    ###############################
    ###      rl observation     ###
    ###############################

    def get_pick_observations(self, target_ee_pose, target_parts_pose, object_name):
        # Get end effector positions and orientations
        end_effector_positions, end_effector_orientations = self._robots._fingertip_centered.get_world_poses(clone=False)
        end_effector_positions = end_effector_positions[:, 0:3] - self._env_pos
        end_effector_orientations = end_effector_orientations[:, [3, 0, 1, 2]]

        target_obj_positions, target_obj_orientations = self._parts[object_name].get_world_poses()
        target_obj_positions -= self._env_pos
        target_obj_orientations[0, :] = target_obj_orientations[0, [3, 0, 1, 2]]

        # To torch.Tensor & device
        target_ee_pos = torch.tensor([target_ee_pose[0]], device=self._device)
        target_ee_orn = torch.tensor([target_ee_pose[1]], device=self._device)

        # To torch.Tensor & device
        target_parts_pos = torch.tensor([target_parts_pose[0]], device=self._device)
        target_parts_orn = torch.tensor([target_parts_pose[1]], device=self._device)

        # Calculate difference
        diff_ee_pos = calc_diff_pos(end_effector_positions, target_ee_pos)
        diff_ee_rot = calc_diff_rot(end_effector_orientations, target_ee_orn)

        # Calculate difference between target object pose and current object pose
        diff_obj_pos = calc_diff_pos(target_obj_positions, target_parts_pos)
        diff_obj_rot = calc_diff_rot(target_obj_orientations, target_parts_orn)

        # Get dof positions
        dof_pos = self._robots.get_joint_positions(clone=False)

        self.obs_buf[..., 0:5] = dof_pos[..., self.arm_dof_idxs]
        self.obs_buf[..., 5:8] = diff_ee_pos
        self.obs_buf[..., 8:12] = diff_ee_rot
        self.obs_buf[..., 12:15] = diff_obj_pos
        self.obs_buf[..., 15:19] = diff_obj_rot

        return self.obs_buf

    def get_place_observations(self, target_parts_pose, object_name):
        # Get end effector positions and orientations
        end_effector_positions, end_effector_orientations = self._robots._fingertip_centered.get_world_poses(clone=False)
        end_effector_positions = end_effector_positions[:, 0:3] - self._env_pos
        end_effector_orientations = end_effector_orientations[:, [3, 0, 1, 2]]

        target_obj_positions, target_obj_orientations = self._parts[object_name].get_world_poses()
        target_obj_orientations[0, :] = target_obj_orientations[0, [3, 0, 1, 2]]

        # To torch.Tensor & device
        target_parts_pos = torch.tensor([target_parts_pose[0]], device=self._device)
        target_parts_orn = torch.tensor([target_parts_pose[1]], device=self._device)

        # Calculate difference
        diff_ee_pos = calc_diff_pos(end_effector_positions, target_obj_positions)
        diff_ee_rot = calc_diff_rot(end_effector_orientations, target_obj_orientations)

        # Calculate difference between target object pose and current object pose
        diff_obj_pos = calc_diff_pos(target_obj_positions, target_parts_pos)
        diff_obj_rot = calc_diff_rot(target_obj_orientations, target_parts_orn)

        # Get dof positions
        dof_pos = self._robots.get_joint_positions(clone=False)

        self.obs_buf[..., 0:5] = dof_pos[..., self.arm_dof_idxs]
        self.obs_buf[..., 5:8] = diff_ee_pos
        self.obs_buf[..., 8:12] = diff_ee_rot
        self.obs_buf[..., 12:15] = diff_obj_pos
        self.obs_buf[..., 15:19] = diff_obj_rot

        return self.obs_buf

    def get_insert_observations(self, target_parts_pose, object_name):
        # Get end effector positions and orientations
        end_effector_positions, end_effector_orientations = self._robots._fingertip_centered.get_world_poses(clone=False)
        end_effector_positions = end_effector_positions[:, 0:3] - self._env_pos
        end_effector_orientations = end_effector_orientations[:, [3, 0, 1, 2]]

        target_obj_positions, target_obj_orientations = self._parts[object_name].get_world_poses()
        target_obj_positions -= self._env_pos
        target_obj_orientations[0, :] = target_obj_orientations[0, [3, 0, 1, 2]]

        # To torch.Tensor & device
        target_parts_pos = torch.tensor([target_parts_pose[0]], device=self._device)
        target_parts_orn = torch.tensor([target_parts_pose[1]], device=self._device)

        # Calculate difference
        diff_ee_pos = calc_diff_pos(end_effector_positions, target_obj_positions)
        diff_ee_rot = calc_diff_rot(end_effector_orientations, target_obj_orientations)

        # Calculate difference between target object pose and current object pose
        diff_obj_pos = calc_diff_pos(target_obj_positions, target_parts_pos)
        diff_obj_rot = calc_diff_rot(target_obj_orientations, target_parts_orn)

        # Get dof positions
        dof_pos = self._robots.get_joint_positions(clone=False)

        self.obs_buf[..., 0:5] = dof_pos[..., self.arm_dof_idxs]
        self.obs_buf[..., 5:8] = diff_ee_pos
        self.obs_buf[..., 8:12] = diff_ee_rot
        self.obs_buf[..., 12:15] = diff_obj_pos
        self.obs_buf[..., 15:19] = diff_obj_rot

        return self.obs_buf

    ################################
    ###       physics steps      ###
    ################################

    def pre_physics_step(self, actions) -> None:
        if not self._env._world.is_playing():
            return

        reset_env_ids = self.reset_buf.nonzero(as_tuple=False).squeeze(-1)
        if len(reset_env_ids) > 0:
            self.reset_idx(reset_env_ids)
            SimulationContext.step(self._env._world, render=True)

        # Set actions
        self.dof_position_targets[..., self.movable_dof_indices] = actions
        self.dof_position_targets[:] = tensor_clamp(
            self.dof_position_targets, self.robot_dof_lower_limits, self.robot_dof_upper_limits
        )

        # Modify torso joint positions
        dof_pos = self._robots.get_joint_positions()
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

        # Shaft1 pos / rot
        self.shaft1_pos = self.initial_shaft1_pos.clone()
        self.shaft1_rot = self.initial_shaft1_rot.clone()

        # Shaft2 pos / rot
        self.shaft2_pos = self.initial_shaft2_pos.clone()
        self.shaft2_rot = self.initial_shaft2_rot.clone()

        # Gear1 pos / rot
        self.gear1_pos = self.initial_gear1_pos.clone()
        self.gear1_rot = self.initial_gear1_rot.clone()

        # Gear2 pos / rot
        self.gear2_pos = self.initial_gear2_pos.clone()
        self.gear2_rot = self.initial_gear2_rot.clone()

        # Gear3 pos / rot
        self.gear3_pos = self.initial_gear3_pos.clone()
        self.gear3_rot = self.initial_gear3_rot.clone()

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
        self.progress_buf[env_ids] = 0
        self.pick_success[env_ids] = 0
        self.place_success[env_ids] = 0
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

        # Reset parts pos / rot, and velocities
        self.initial_shaft1_pos, self.initial_shaft1_rot = self._shaft1.get_world_poses()
        self.initial_shaft1_velocities = self._shaft1.get_velocities()

        self.initial_shaft2_pos, self.initial_shaft2_rot = self._shaft2.get_world_poses()
        self.initial_shaft2_velocities = self._shaft2.get_velocities()

        self.initial_gear1_pos, self.initial_gear1_rot = self._gear1.get_world_poses()
        self.initial_gear1_velocities = self._gear1.get_velocities()

        self.initial_gear2_pos, self.initial_gear2_rot = self._gear2.get_world_poses()
        self.initial_gear2_velocities = self._gear2.get_velocities()

        self.initial_gear3_pos, self.initial_gear3_rot = self._gear3.get_world_poses()
        self.initial_gear3_velocities = self._gear3.get_velocities()

        # Reset all environment
        indices = torch.arange(self._num_envs, dtype=torch.int64, device=self._device)
        self.reset_idx(indices)

    def is_done(self) -> None:
        self.reset_buf = torch.where(
            self.progress_buf[:] >= self._max_episode_length - 1,
            torch.ones_like(self.reset_buf),
            self.reset_buf
        )

    def post_physics_step(self, action_name, object_name=None, target_ee_pose=None, target_parts_pose=None):
        """Step buffers. Refresh tensors. Compute observations and reward. Reset environments."""
        self.progress_buf[:] += 1
        if self._env._world.is_playing():
            # In this policy, episode length is constant
            if action_name == 'pick':
                self.get_pick_observations(target_ee_pose, target_parts_pose, object_name)
                self.calculate_pick_metrics(object_name)
            elif action_name == 'place':
                self.get_place_observations(target_parts_pose, object_name)
                self.calculate_place_metrics(target_parts_pose, object_name)
            elif action_name == 'insert':
                self.get_insert_observations(target_parts_pose, object_name)
                self.calculate_insert_metrics(target_parts_pose, object_name)
            self.get_states()
            self.is_done()
            self.get_extras()

        return self.obs_buf, self.rew_buf, self.reset_buf, self.extras

    ################################
    ###     calculate metrics    ###
    ################################

    def calculate_pick_metrics(self, object_name) -> None:
        self.rew_buf[:] = torch.zeros(1, device=self._device, dtype=torch.float)

        end_effector_positions, _ = self._robots._fingertip_centered.get_world_poses(clone=False)
        end_effector_positions = end_effector_positions[:, 0:3] - self._env_pos

        # Get current parts positions and orientations
        curr_parts_positions, curr_parts_orientations = self._parts[object_name].get_world_poses()
        curr_parts_positions -= self._env_pos
        if 'shaft' in object_name:
            curr_parts_orientations[0, :] = curr_parts_orientations[0, [3, 0, 1, 2]]

        # Distance from hand to the target object
        dist = torch.norm(end_effector_positions - curr_parts_positions, p=2, dim=-1)
        dist_reward = 1.0 / (1.0 + dist ** 2)
        dist_reward *= dist_reward
        if 'shaft' in object_name:
            if dist <= 0.08:
                dist_reward *= 2
        elif 'gear' in object_name:
            if dist <= 0.15:
                dist_reward *= 2

        self.rew_buf[:] += dist_reward * self._task_cfg['rl']['distance_scale']

        # Check if block is picked up and close to target pose
        pick_success = self._check_pick_success(object_name)
        self.rew_buf[:] += pick_success * self._task_cfg['rl']['pick_success_bonus']
        self.extras['pick_successes'] = torch.mean(pick_success.float())
        self.pick_success = torch.where(
            pick_success[:] == 1,
            torch.ones_like(pick_success),
            -torch.ones_like(pick_success)
        )

    def calculate_place_metrics(self, target_parts_pose, object_name) -> None:
        self.rew_buf[:] = torch.zeros(1, device=self._device, dtype=torch.float)

        # Get current parts positions and orientations
        curr_parts_positions, curr_parts_orientations = self._parts[object_name].get_world_poses()
        curr_parts_positions -= self._env_pos
        if 'shaft' in object_name:
            curr_parts_orientations[0, :] = curr_parts_orientations[0, [3, 0, 1, 2]]

        target_parts_pos = torch.tensor([target_parts_pose[0]], device=self._device)
        target_parts_orn = torch.tensor([target_parts_pose[1]], device=self._device)

        # Calculate difference between target object pose and final object pose
        target_pos_dist = norm_diff_pos(curr_parts_positions, target_parts_pos)
        target_rot_dist = norm_diff_rot(curr_parts_orientations, target_parts_orn)

        target_pos_dist_reward = 1.0 / (1.0 + target_pos_dist ** 2)
        target_pos_dist_reward *= target_pos_dist_reward
        target_pos_dist_reward = torch.where(target_pos_dist <= 0.05, target_pos_dist_reward * 2, target_pos_dist_reward)

        target_rot_dist_reward = 1.0 / (1.0 + target_rot_dist ** 2)
        target_rot_dist_reward *= target_rot_dist_reward
        target_rot_dist_reward = torch.where(target_rot_dist <= 0.05, target_rot_dist_reward * 2, target_rot_dist_reward)

        self.rew_buf[:] += target_pos_dist_reward * self._task_cfg['rl']['target_position_distance_scale']
        self.rew_buf[:] += target_rot_dist_reward * self._task_cfg['rl']['target_rotation_distance_scale']

        # Check if block is picked up and above table
        place_success = self._check_place_success(target_parts_pose, object_name)
        self.rew_buf[:] += place_success * self._task_cfg['rl']['place_success_bonus']
        self.extras['place_successes'] = torch.mean(place_success.float())
        self.place_success = torch.where(
            place_success[:] == 1,
            torch.ones_like(place_success),
            -torch.ones_like(place_success)
        )

    def calculate_insert_metrics(self, target_parts_pose, object_name) -> None:
        self.rew_buf[:] = torch.zeros(1, device=self._device, dtype=torch.float)

        # Get current parts positions and orientations
        curr_parts_positions, curr_parts_orientations = self._parts[object_name].get_world_poses()
        curr_parts_positions -= self._env_pos
        if 'shaft' in object_name:
            curr_parts_orientations[0, :] = curr_parts_orientations[0, [3, 0, 1, 2]]

        target_parts_pos = torch.tensor([target_parts_pose[0]], device=self._device)
        target_parts_orn = torch.tensor([target_parts_pose[1]], device=self._device)

        # Calculate difference between target object pose and final object pose
        target_pos_dist = norm_diff_pos(curr_parts_positions, target_parts_pos)
        target_rot_dist = norm_diff_rot(curr_parts_orientations, target_parts_orn)

        target_pos_dist_reward = 1.0 / (1.0 + target_pos_dist ** 2)
        target_pos_dist_reward *= target_pos_dist_reward
        target_pos_dist_reward = torch.where(target_pos_dist <= 0.05, target_pos_dist_reward * 2, target_pos_dist_reward)

        target_rot_dist_reward = 1.0 / (1.0 + target_rot_dist ** 2)
        target_rot_dist_reward *= target_rot_dist_reward
        target_rot_dist_reward = torch.where(target_rot_dist <= 0.05, target_rot_dist_reward * 2, target_rot_dist_reward)

        self.rew_buf[:] += target_pos_dist_reward * self._task_cfg['rl']['target_position_distance_scale']
        self.rew_buf[:] += target_rot_dist_reward * self._task_cfg['rl']['target_rotation_distance_scale']

        # Check if block is picked up and above table
        insert_success = self._check_insert_success(target_parts_pose, object_name)
        self.rew_buf[:] += insert_success * self._task_cfg['rl']['insert_success_bonus']
        self.extras['insert_successes'] = torch.mean(insert_success.float())
        self.insert_success = torch.where(
            insert_success[:] == 1,
            torch.ones_like(insert_success),
            -torch.ones_like(insert_success)
        )

    ##############################
    ###     robot utikities    ###
    ##############################

    def set_dof_idxs(self):
        [self.torso_dof_idx.append(self._robots.get_dof_index(name)) for name in self._torso_joint_name]
        [self.base_dof_idxs.append(self._robots.get_dof_index(name)) for name in self._base_joint_names]
        [self.arm_dof_idxs.append(self._robots.get_dof_index(name)) for name in self._arm_names]
        [self.gripper_proximal_dof_idxs.append(self._robots.get_dof_index(name)) for name in self._gripper_proximal_names]

        # Movable joints
        self.actuated_dof_indices = torch.LongTensor(self.base_dof_idxs+self.arm_dof_idxs+self.gripper_proximal_dof_idxs).to(self._device) # torch.LongTensor([0, 1, 2, 3, 5, 7, 9, 10, 11, 12]).to(self._device)
        self.movable_dof_indices = torch.LongTensor(self.base_dof_idxs+self.arm_dof_idxs).to(self._device) # torch.LongTensor([0, 1, 2, 3, 5, 7, 9, 10]).to(self._device)

    def set_dof_limits(self): # dof position limits
        # (1, num_dofs, 2)
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

    ###############################
    ###     gripper commands    ###
    ###############################

    def _close_gripper(self, sim_steps=None):
        # Set gripper target force
        gripper_dof_effort = torch.tensor([-30., -30.], device=self._device)
        self._robots.set_joint_efforts(gripper_dof_effort, joint_indices=self.gripper_proximal_dof_idxs)

        # Step sim
        if sim_steps != None:
            for _ in range(sim_steps):
                SimulationContext.step(self._env._world, render=True)

        self.dof_position_targets[:, self.gripper_proximal_dof_idxs] = self._robots.get_joint_positions(joint_indices=self.gripper_proximal_dof_idxs)
        self.gripper_hold[:] = True

    def _open_gripper(self, sim_steps=None):
        gripper_dof_effort = torch.tensor([0., 0.], device=self._device)
        self._robots.set_joint_efforts(gripper_dof_effort, joint_indices=self.gripper_proximal_dof_idxs)

        gripper_dof_pos = torch.tensor([0.5, 0.5], device=self._device)
        self._robots.set_joint_position_targets(gripper_dof_pos, joint_indices=self.gripper_proximal_dof_idxs)

        # Step sim
        if sim_steps != None:
            for _ in range(sim_steps):
                SimulationContext.step(self._env._world, render=True)

        self.dof_position_targets[:, self.gripper_proximal_dof_idxs] = self._robots.get_joint_positions(joint_indices=self.gripper_proximal_dof_idxs)
        self.gripper_hold[:] = False

    def _hold_gripper(self):
        gripper_dof_effort = torch.tensor([-30., -30.], device=self._device)
        self._robots.set_joint_efforts(gripper_dof_effort, joint_indices=self.gripper_proximal_dof_idxs)
        self.dof_position_targets[:, self.gripper_proximal_dof_idxs] = self._robots.get_joint_positions(joint_indices=self.gripper_proximal_dof_idxs)

    ###############################
    ###   success evaluations   ###
    ###############################

    def _check_pick_success(self, object_name):
        parts_positions, _ = self._parts[object_name].get_world_poses()
        parts_positions -= self._env_pos

        # check z direction range
        pick_success = torch.where(
            parts_positions[:, 2] > torch.tensor([0.30], device=self._device),
            torch.ones((1,), device=self._device),
            torch.zeros((1,), device=self._device)
        )

        return pick_success

    def _check_place_success(self, target_parts_pose, object_name):
        parts_positions, parts_orientations = self._parts[object_name].get_world_poses()
        parts_positions -= self._env_pos
        if 'shaft' in object_name:
            parts_orientations[0, :] = parts_orientations[0, [3, 0, 1, 2]]

        target_parts_pos = torch.tensor([target_parts_pose[0]], device=self._device)
        target_parts_orn = torch.tensor([target_parts_pose[1]], device=self._device)

        # Check difference between target pose and current pose
        target_pos_dist = norm_diff_pos(parts_positions, target_parts_pos)
        target_rot_dist = norm_diff_rot(parts_orientations, target_parts_orn)

        place_success = torch.where(
            target_pos_dist < torch.tensor([0.02], device=self._device),
            torch.ones((1,), device=self._device),
            torch.zeros((1,), device=self._device)
        )
        place_success = torch.where(
            target_rot_dist < torch.tensor([0.02], device=self._device),
            place_success,
            torch.zeros((1,), device=self._device)
        )

        return place_success

    def _check_insert_success(self, target_parts_pose, object_name):
        parts_positions, parts_orientations = self._parts[object_name].get_world_poses()
        parts_positions -= self._env_pos
        if 'shaft' in object_name:
            parts_orientations[0, :] = parts_orientations[0, [3, 0, 1, 2]]

        target_parts_pos = torch.tensor([target_parts_pose[0]], device=self._device)
        target_parts_orn = torch.tensor([target_parts_pose[1]], device=self._device)

        # Check difference between target pose and current pose
        target_pos_dist = norm_diff_pos(parts_positions, target_parts_pos)
        target_rot_dist = norm_diff_rot(parts_orientations, target_parts_orn)

        insert_success = torch.where(
            target_pos_dist < torch.tensor([0.02], device=self._device),
            torch.ones((1,), device=self._device),
            torch.zeros((1,), device=self._device)
        )
        insert_success = torch.where(
            target_rot_dist < torch.tensor([0.02], device=self._device),
            insert_success,
            torch.zeros((1,), device=self._device)
        )

        return insert_success

    ##############################
    ###  simulation utilities  ###
    ##############################

    def enable_physics(self, object_name):
        """Enable physics properties."""
        self._parts[object_name].enable_rigid_body_physics()

    def disable_physics(self, object_name):
        """Disable physics properties."""
        self._parts[object_name].disable_rigid_body_physics()

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