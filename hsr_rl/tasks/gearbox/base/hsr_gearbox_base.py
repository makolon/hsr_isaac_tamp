import carb
import hydra
import torch
import numpy as np
from hsr_rl.robots.articulations.hsr import HSR
from hsr_rl.robots.articulations.views.hsr_view import HSRView
from hsr_rl.tasks.base.rl_task import RLTask
from hsr_rl.tasks.utils.scene_utils import spawn_dynamic_object, spawn_static_object
from hsr_rl.tasks.utils.ik_utils import DifferentialInverseKinematics, DifferentialInverseKinematicsCfg

from omni.isaac.core.prims import GeometryPrimView, RigidPrimView, XFormPrimView
from omni.isaac.core.utils.prims import get_prim_at_path
from omni.isaac.core.utils.stage import get_current_stage, print_stage_prim_paths
from omni.isaac.core.utils.torch.maths import torch_rand_float, tensor_clamp
from omni.isaac.core.utils.torch.rotations import quat_diff_rad, quat_mul, normalize
from omni.isaac.core.objects import FixedCuboid
from omni.isaac.core.simulation_context import SimulationContext
from omni.isaac.dynamic_control import _dynamic_control
from omni.physx.scripts import utils, physicsUtils
from pxr import Gf, Sdf, UsdGeom, PhysxSchema, UsdPhysics


class HSRGearboxBaseTask(RLTask):
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
        self._control_frequency = torch.tensor(1/self._sim_config.task_config["env"]["controlFrequencyInv"], device=self._device)
        self._dt = torch.tensor(self._sim_config.task_config["sim"]["dt"] * self._sim_config.task_config["env"]["controlFrequencyInv"], device=self._device)

        # Set environment properties
        self._table_height = self._task_cfg["env"]["table_height"]
        self._table_width = self._task_cfg["env"]["table_width"]
        self._table_depth = self._task_cfg["env"]["table_depth"]
        self._base_height = self._task_cfg["env"]["base_height"]
        self._base_width = self._task_cfg["env"]["base_width"]
        self._base_depth = self._task_cfg["env"]["base_depth"]
        self._shaft_height = self._task_cfg["env"]["shaft_height"]
        self._shaft_width = self._task_cfg["env"]["shaft_width"]
        self._shaft_depth = self._task_cfg["env"]["shaft_depth"]
        self._gear_base_height = self._task_cfg["env"]["gear_base_height"]
        self._gear_base_width = self._task_cfg["env"]["gear_base_width"]
        self._gear_base_depth = self._task_cfg["env"]["gear_base_depth"]
        self._block1_height = self._task_cfg["env"]["block1_height"]
        self._block1_width = self._task_cfg["env"]["block1_width"]
        self._block1_depth = self._task_cfg["env"]["block1_depth"]
        self._block2_height = self._task_cfg["env"]["block2_height"]
        self._block2_width = self._task_cfg["env"]["block2_width"]
        self._block2_depth = self._task_cfg["env"]["block2_depth"]

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

        # Set inverse kinematics configurations
        self._action_type = self._task_cfg["env"]["action_type"]
        self._target_space = self._task_cfg["env"]["target_space"]

        # Set learning hyper-parameters
        self._pick_success = self._task_cfg["rl"]["pick_threshold"]
        self._place_success = self._task_cfg["rl"]["place_threshold"]
        self._insert_success = self._task_cfg['rl']['insert_threshold']
        self._residual_observation = self._task_cfg['rl']['residual_observation']
        self._residual_dynamics = self._task_cfg['rl']['residual_dynamics']
        self._action_speed_scale = self._task_cfg["env"]["actionSpeedScale"]
        self._max_episode_length = self._task_cfg["env"]["maxEpisodeLength"]

        # Load experts demonstraions
        self.exp_actions, self.env_info = self.load_exp_dataset()

        # Set up environment from loaded demonstration
        self.set_up_environment()

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

        # Reward settings
        self.is_collided = torch.zeros(self._num_envs, device=self._device, dtype=torch.long)
        self.is_failure = torch.zeros(self._num_envs, device=self._device, dtype=torch.long)
        self.is_success = torch.zeros(self._num_envs, device=self._device, dtype=torch.long)
        self.pick_success = torch.zeros(self._num_envs, device=self._device, dtype=torch.long)
        self.place_success = torch.zeros(self._num_envs, device=self._device, dtype=torch.long)
        self.insert_success = torch.zeros(self._num_envs, device=self._device, dtype=torch.long)

        # Demonstration steps
        self.replay_count = torch.zeros(self._num_envs, device=self._device, dtype=torch.long)

        # Gripper settings
        self.gripper_close = torch.zeros(self._num_envs, device=self._device, dtype=torch.bool)
        self.gripper_open = torch.zeros(self._num_envs, device=self._device, dtype=torch.bool)
        self.gripper_hold = torch.zeros(self._num_envs, device=self._device, dtype=torch.bool)

        RLTask.__init__(self, name, env)
        return

    def set_up_environment(self) -> None:
        # Environment object settings: (reset() randomizes the environment)
        obj_name_candit = ["gear1", "gear2", "gear3", "shaft1", "shaft2"]
        dynamic_obj_name, static_obj_name = [], []
        for i in range(self._num_envs):
            dynamic_obj_name.append(self._target_obj_names[i])
            static_obj_name.append([obj for obj in obj_name_candit if obj != self._target_obj_names[i]])

        self._dynamic_obj_names = dynamic_obj_name
        self._static_obj_names = static_obj_name

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

        self.init_dynamic_pos = {
            'gear1': torch.tensor([1.2, 0.30, self._table_height+self._base_height], device=self._device),
            'gear2': torch.tensor([1.2, -0.50, self._table_height+self._base_height], device=self._device),
            'gear3': torch.tensor([1.2, -0.75, self._table_height+self._base_height], device=self._device),
            'shaft1': torch.tensor([1.2, 0.50, self._table_height+self._shaft_height], device=self._device),
            'shaft2': torch.tensor([1.2, -0.30, self._table_height+self._shaft_height], device=self._device),
            'gearbox_base': torch.tensor([1.1, 0.0, self._table_height+self._gear_base_height], device=self._device)}
        self.init_dynamic_orn = {
            'gear1': torch.tensor([1.0, 0.0, 0.0, 0.0], device=self._device),
            'gear2': torch.tensor([1.0, 0.0, 0.0, 0.0], device=self._device),
            'gear3': torch.tensor([1.0, 0.0, 0.0, 0.0], device=self._device),
            'shaft1': torch.tensor([0.0, 0.0, 1.0, 0.0], device=self._device),
            'shaft2': torch.tensor([0.0, 0.0, 1.0, 0.0], device=self._device),
            'gearbox_base': torch.tensor([0.92387953, 0.0, 0.0, -0.38268343], device=self._device)}
        self.init_static_pos = {
            'gear1': torch.tensor([1.2, 0.30, self._table_height+self._base_height], device=self._device),
            'gear2': torch.tensor([1.2, -0.50, self._table_height+self._base_height], device=self._device),
            'gear3': torch.tensor([1.2, -0.75, self._table_height+self._base_height], device=self._device),
            'shaft1': torch.tensor([1.2, 0.50, self._table_height+self._shaft_height], device=self._device),
            'shaft2': torch.tensor([1.2, -0.30, self._table_height+self._shaft_height], device=self._device),
            'table': torch.tensor([1.2, 0.0, 0.0], device=self._device),
            'box': torch.tensor([1.2, 0.0, 0.0, 0.0], device=self._device),
            'gearbox_base': torch.tensor([1.1, 0.0, self._table_height+self._gear_base_height], device=self._device)}
        self.init_static_orn = {
            'gear1': torch.tensor([1.0, 0.0, 0.0, 0.0], device=self._device),
            'gear2': torch.tensor([1.0, 0.0, 0.0, 0.0], device=self._device),
            'gear3': torch.tensor([1.0, 0.0, 0.0, 0.0], device=self._device),
            'shaft1': torch.tensor([0.0, 0.0, 1.0, 0.0], device=self._device),
            'shaft2': torch.tensor([0.0, 0.0, 1.0, 0.0], device=self._device),
            'table': torch.tensor([1.0, 0.0, 0.0, 0.0], device=self._device),
            'box': torch.tensor([1.0, 0.0, 0.0, 0.0], device=self._device),
            'gearbox_base': torch.tensor([0.92387953, 0.0, 0.0, -0.38268343], device=self._device)}

        # For target pose
        virtual_shaft1_positions = []
        virtual_shaft1_rotations = []
        virtual_shaft2_positions = []
        virtual_shaft2_rotations = []

        shaft1_static_env_ids = []
        shaft2_static_env_ids = []
        for i in range(self._num_envs):
            if 'gear1' in self._dynamic_obj_names[i]:
                virtual_shaft1_positions.append(self.env_info['initial_object_pose'][i]['shaft1'][0])
                virtual_shaft1_rotations.append(self.env_info['initial_object_pose'][i]['shaft1'][1])
                shaft1_static_env_ids.append(i)
            elif 'gear2' in self._dynamic_obj_names[i]:
                virtual_shaft2_positions.append(self.env_info['initial_object_pose'][i]['shaft2'][0])
                virtual_shaft2_rotations.append(self.env_info['initial_object_pose'][i]['shaft2'][1])
                shaft2_static_env_ids.append(i)

        self._virtual_shaft1_position = torch.tensor(virtual_shaft1_positions, dtype=torch.float, device=self._device)
        self._virtual_shaft1_rotation = torch.tensor(virtual_shaft1_rotations, dtype=torch.float, device=self._device)
        self._virtual_shaft2_position = torch.tensor(virtual_shaft2_positions, dtype=torch.float, device=self._device)
        self._virtual_shaft2_rotation = torch.tensor(virtual_shaft2_rotations, dtype=torch.float, device=self._device)

        self.shaft1_static_env_ids = torch.tensor(shaft1_static_env_ids, device=self._device)
        self.shaft2_static_env_ids = torch.tensor(shaft2_static_env_ids, device=self._device)

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
        self.add_dynamic_parts()
        self.add_static_parts()
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

        static_shaft1_env_ids = []
        static_shaft2_env_ids = []
        static_gear1_env_ids = []
        static_gear2_env_ids = []
        static_gear3_env_ids = []

        # Bookkeep for remove physics properties
        static_env_ids = torch.arange(0, self._num_envs)
        for env_id, obj_names in enumerate(self._static_obj_names):
            if env_id == 0:
                continue
            if 'shaft1' in obj_names:
                static_shaft1_env_ids.append(static_env_ids[env_id])
            if 'shaft2' in obj_names:
                static_shaft2_env_ids.append(static_env_ids[env_id])
            if 'gear1' in obj_names:
                static_gear1_env_ids.append(static_env_ids[env_id])
            if 'gear2' in obj_names:
                static_gear2_env_ids.append(static_env_ids[env_id])
            if 'gear3' in obj_names:
                static_gear3_env_ids.append(static_env_ids[env_id])

        self.static_shaft1_env_ids = torch.tensor(static_shaft1_env_ids, device=self._device)
        self.static_shaft2_env_ids = torch.tensor(static_shaft2_env_ids, device=self._device)
        self.static_gear1_env_ids = torch.tensor(static_gear1_env_ids, device=self._device)
        self.static_gear2_env_ids = torch.tensor(static_gear2_env_ids, device=self._device)
        self.static_gear3_env_ids = torch.tensor(static_gear3_env_ids, device=self._device)

        self._parts = {'shaft1': self._shaft1, 'shaft2': self._shaft2,
                       'gear1': self._gear1, 'gear2': self._gear2, 'gear3': self._gear3}

    def add_hsr(self):
        # Add HSR
        for i in range(self._num_envs):
            hsrb = HSR(prim_path=f"/World/envs/env_{i}/hsrb",
                       name="hsrb",
                       translation=self._hsr_position,
                       orientation=self._hsr_rotation)
            self._sim_config.apply_articulation_settings("hsrb", get_prim_at_path(hsrb.prim_path), self._sim_config.parse_actor_config("hsrb"))

            # Add physics material
            physicsUtils.add_physics_material_to_prim(
                self._stage,
                self._stage.GetPrimAtPath(f"/World/envs/env_{i}/hsrb/hand_r_distal_link/collisions/mesh_0"),
                self.gripperPhysicsMaterialPath
            )
            physicsUtils.add_physics_material_to_prim(
                self._stage,
                self._stage.GetPrimAtPath(f"/World/envs/env_{i}/hsrb/hand_l_distal_link/collisions/mesh_0"),
                self.gripperPhysicsMaterialPath
            )

    def add_table(self):
        # Add table
        for i in range(self._num_envs):
            table = FixedCuboid(prim_path=f"/World/envs/env_{i}/table",
                                name="table",
                                translation=self.env_info['initial_object_pose'][i]['table'][0],
                                orientation=self.init_static_orn['table'],
                                size=1.0,
                                color=torch.tensor([0.75, 0.75, 0.75]),
                                scale=torch.tensor([self._table_width, self._table_depth, self._table_height]))
            self._sim_config.apply_articulation_settings("table", get_prim_at_path(table.prim_path), self._sim_config.parse_actor_config("table"))

    def add_base(self):
        # Add base
        for i in range(self._num_envs):
            base_1 = FixedCuboid(prim_path=f"/World/envs/env_{i}/base_1",
                                 name="base_1",
                                 translation=self.env_info['initial_object_pose'][i]['box_1'][0],
                                 orientation=self.init_static_orn['box'],
                                 size=1.0,
                                 color=torch.tensor([0.1, 0.1, 0.1]),
                                 scale=torch.tensor([self._base_width, self._base_depth, self._base_height]))
            self._sim_config.apply_articulation_settings("base", get_prim_at_path(base_1.prim_path), self._sim_config.parse_actor_config("base"))

            base_2 = FixedCuboid(prim_path=f"/World/envs/env_{i}/base_2",
                                 name="base_2",
                                 translation=self.env_info['initial_object_pose'][i]['box_2'][0],
                                 orientation=self.init_static_orn['box'],
                                 size=1.0,
                                 color=torch.tensor([0.1, 0.1, 0.1]),
                                 scale=torch.tensor([self._base_width, self._base_depth, self._base_height]))
            self._sim_config.apply_articulation_settings("base", get_prim_at_path(base_2.prim_path), self._sim_config.parse_actor_config("base"))

            base_3 = FixedCuboid(prim_path=f"/World/envs/env_{i}/base_3",
                                 name="base_3",
                                 translation=self.env_info['initial_object_pose'][i]['box_3'][0],
                                 orientation=self.init_static_orn['box'],
                                 size=1.0,
                                 color=torch.tensor([0.1, 0.1, 0.1]),
                                 scale=torch.tensor([self._base_width, self._base_depth, self._base_height]))
            self._sim_config.apply_articulation_settings("base", get_prim_at_path(base_3.prim_path), self._sim_config.parse_actor_config("base"))

            # Add gear base
            gear_base = FixedCuboid(prim_path=f"/World/envs/env_{i}/gear_base",
                                    name="gear_base",
                                    translation=self.env_info['initial_object_pose'][i]['box_4'][0],
                                    orientation=self.init_static_orn['gearbox_base'],
                                    size=1.0,
                                    color=torch.tensor([0.1, 0.1, 0.1]),
                                    scale=torch.tensor([self._gear_base_width, self._gear_base_depth, self._gear_base_height]))
            self._sim_config.apply_articulation_settings("base", get_prim_at_path(gear_base.prim_path), self._sim_config.parse_actor_config("base"))

    def add_block(self):
        # Add virtual shaft1
        for i, idx in enumerate(self.shaft1_static_env_ids):
            block_position = self._virtual_shaft1_position[i]
            block_position[2] -= self._block1_height / 2
            block_rotation = self._virtual_shaft1_rotation[i]
            block = FixedCuboid(prim_path=f"/World/envs/env_{idx}" + "/virtual_shaft1", # TODO: modify
                                name="virtual_shaft1",
                                translation=block_position,
                                orientation=block_rotation,
                                size=1.0,
                                color=torch.tensor([0.1, 0.1, 0.1]),
                                scale=torch.tensor([self._block1_width, self._block1_depth, self._block1_height]))
            self._sim_config.apply_articulation_settings("block", get_prim_at_path(block.prim_path), self._sim_config.parse_actor_config("block"))

        # Add virtual shaft2
        for i, idx in enumerate(self.shaft2_static_env_ids):
            block_position = self._virtual_shaft2_position[i]
            block_position[2] -= self._block2_height / 2
            block_rotation = self._virtual_shaft2_rotation[i]
            block = FixedCuboid(prim_path=f"/World/envs/env_{idx}" + "/virtual_shaft2", # TODO: modify
                                name="virtual_shaft2",
                                translation=block_position,
                                orientation=block_rotation,
                                size=1.0,
                                color=torch.tensor([0.1, 0.1, 0.1]),
                                scale=torch.tensor([self._block2_width, self._block2_depth, self._block2_height]))
            self._sim_config.apply_articulation_settings("block", get_prim_at_path(block.prim_path), self._sim_config.parse_actor_config("block"))

    def add_gearbox_base(self):
        for i in range(self._num_envs):
            object_translation = self.env_info['initial_object_pose'][i]['gearbox_base'][0]
            object_orientation = self.init_static_orn['gearbox_base']
            parts = spawn_static_object(name='gearbox_base',
                                        prim_path=f"/World/envs/env_{i}",
                                        object_translation=object_translation,
                                        object_orientation=object_orientation)
            self._sim_config.apply_articulation_settings("gearbox_base",
                                            get_prim_at_path(parts.prim_path),
                                            self._sim_config.parse_actor_config('gearbox_base'))

    def add_static_parts(self):
        # Add static gearbox parts
        for i in range(self._num_envs):
            for idx in range(len(self._static_obj_names[i])):
                object_translation = self.env_info['initial_object_pose'][i][self._static_obj_names[i][idx]][0]
                object_orientation = self.env_info['initial_object_pose'][i][self._static_obj_names[i][idx]][1]
                parts = spawn_static_object(name=self._static_obj_names[i][idx],
                                            prim_path=f"/World/envs/env_{i}",
                                            object_translation=object_translation,
                                            object_orientation=object_orientation)
                self._sim_config.apply_articulation_settings(f"{self._static_obj_names[i]}",
                                                get_prim_at_path(parts.prim_path),
                                                self._sim_config.parse_actor_config(self._static_obj_names[i][idx]))

                # Add physics material
                if 'gear' in self._static_obj_names[i][idx]:
                    physicsUtils.add_physics_material_to_prim(
                        self._stage,
                        self._stage.GetPrimAtPath(f"/World/envs/env_{i}/{self._static_obj_names[i][idx]}/{self._static_obj_names[i][idx]}/collisions/mesh_0"),
                        self.gearPhysicsMaterialPath
                    )
                elif 'shaft' in self._static_obj_names[i][idx]:
                    physicsUtils.add_physics_material_to_prim(
                        self._stage,
                        self._stage.GetPrimAtPath(f"/World/envs/env_{i}/{self._static_obj_names[i][idx]}/{self._static_obj_names[i][idx]}/collisions/mesh_0"),
                        self.shaftPhysicsMaterialPath
                    )

    def add_dynamic_parts(self):
        # Add movable gearbox parts
        for i in range(self._num_envs):
            object_translation = self.env_info['initial_object_pose'][i][self._dynamic_obj_names[i]][0]
            object_orientation = self.env_info['initial_object_pose'][i][self._dynamic_obj_names[i]][1]
            parts = spawn_dynamic_object(name=self._dynamic_obj_names[i],
                                         prim_path=f"/World/envs/env_{i}",
                                         object_translation=object_translation,
                                         object_orientation=object_orientation)
            self._sim_config.apply_articulation_settings(f"{self._dynamic_obj_names[i]}",
                                            get_prim_at_path(parts.prim_path),
                                            self._sim_config.parse_actor_config(self._dynamic_obj_names[i]))

            # Add physics material
            if 'gear' in self._dynamic_obj_names[i]:
                physicsUtils.add_physics_material_to_prim(
                    self._stage,
                    self._stage.GetPrimAtPath(f"/World/envs/env_{i}/{self._dynamic_obj_names[i]}/{self._dynamic_obj_names[i]}/collisions/mesh_0"),
                    self.gearPhysicsMaterialPath
                )
            elif 'shaft' in self._dynamic_obj_names[i]:
                physicsUtils.add_physics_material_to_prim(
                    self._stage,
                    self._stage.GetPrimAtPath(f"/World/envs/env_{i}/{self._dynamic_obj_names[i]}/{self._dynamic_obj_names[i]}/collisions/mesh_0"),
                    self.shaftPhysicsMaterialPath
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

    def create_gearbox_material(self):
        self._stage = get_current_stage()
        self.gearPhysicsMaterialPath = "/World/Physics_Materials/GearMaterial"
        self.shaftPhysicsMaterialPath = "/World/Physics_Materials/ShaftMaterial"

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

    def post_physics_step(self):
        raise NotImplementedError()

    def calculate_command(self, curr_pose, replay_indices):
        # Calculate control step
        control_step = torch.unsqueeze(self.progress_buf % self.control_frequency_inv + 1, dim=1)

        # Calculate delta command
        target_pose = self.exp_actions[torch.arange(self._num_envs), replay_indices, :8]
        delta_pose = target_pose - curr_pose
        delta_pose = delta_pose / self.control_frequency_inv * control_step
        return delta_pose.clone().detach()

    def is_done(self) -> None:
        self.reset_buf = torch.where(
            self.progress_buf[:] >= self._max_episode_length - 1,
            torch.ones_like(self.reset_buf),
            self.reset_buf
        )

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

    def _close_gripper(self, env_ids, sim_steps=None):
        env_ids_32 = env_ids.type(torch.int32)
        env_ids_64 = env_ids.type(torch.int64)

        # Set gripper target force
        gripper_dof_effort = torch.tensor([-30., -30.], device=self._device)
        self._robots.set_joint_efforts(gripper_dof_effort, indices=env_ids_32, joint_indices=self.gripper_proximal_dof_idxs)

        # Step sim
        if sim_steps != None:
            for _ in range(sim_steps):
                SimulationContext.step(self._env._world, render=True)

        self.dof_position_targets[env_ids_64[:, None], self.gripper_proximal_dof_idxs] = self._robots.get_joint_positions(indices=env_ids_32, joint_indices=self.gripper_proximal_dof_idxs)
        self.gripper_hold[env_ids_64] = True

    def _open_gripper(self, env_ids, sim_steps=None):
        env_ids_32 = env_ids.type(torch.int32)
        env_ids_64 = env_ids.type(torch.int64)

        # Remove gripper force
        gripper_dof_effort = torch.tensor([0., 0.], device=self._device)
        self._robots.set_joint_efforts(gripper_dof_effort, indices=env_ids_32, joint_indices=self.gripper_proximal_dof_idxs)

        # Set gripper target angle
        gripper_dof_pos = torch.tensor([0.5, 0.5], device=self._device)
        self._robots.set_joint_position_targets(gripper_dof_pos, indices=env_ids_32, joint_indices=self.gripper_proximal_dof_idxs)

        # Step sim
        if sim_steps != None:
            for _ in range(sim_steps):
                SimulationContext.step(self._env._world, render=True)

        self.dof_position_targets[env_ids_64[:, None], self.gripper_proximal_dof_idxs] = self._robots.get_joint_positions(indices=env_ids_32, joint_indices=self.gripper_proximal_dof_idxs)
        self.gripper_hold[env_ids_64] = False

    def _hold_gripper(self, env_ids):
        env_ids_32 = env_ids.type(torch.int32)
        env_ids_64 = env_ids.type(torch.int64)

        gripper_dof_effort = torch.tensor([-30., -30.], device=self._device)
        self._robots.set_joint_efforts(gripper_dof_effort, indices=env_ids_32, joint_indices=self.gripper_proximal_dof_idxs)
        self.dof_position_targets[env_ids_64[:, None], self.gripper_proximal_dof_idxs] = self._robots.get_joint_positions(indices=env_ids_32, joint_indices=self.gripper_proximal_dof_idxs)

    def load_exp_dataset(self):
        raise NotImplementedError()

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