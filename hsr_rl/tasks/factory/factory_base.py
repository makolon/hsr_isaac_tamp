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

"""Factory: base class.

Inherits Gym's VecTask class and abstract base class. Inherited by environment classes. Not directly executed.

Configuration defined in FactoryBase.yaml. Asset info defined in factory_asset_info_hsr_table.yaml.
"""


from hsr_rl.tasks.base.rl_task import RLTask
from hsr_rl.tasks.factory.factory_schema_class_base import FactoryABCBase
from hsr_rl.tasks.factory.factory_schema_config_base import FactorySchemaConfigBase 
from hsr_rl.robots.articulations.hsr import HSR
import hsr_rl.tasks.factory.factory_control as fc

from omni.isaac.core.objects import FixedCuboid
from omni.isaac.core.utils.prims import get_prim_at_path
from omni.isaac.core.utils.stage import get_current_stage

from pxr import Usd, UsdGeom, Sdf, Gf, PhysxSchema, UsdPhysics

import hydra
import numpy as np
import torch
import math
import carb


class FactoryBase(RLTask, FactoryABCBase):
    def __init__(self, name, sim_config, env) -> None:
        """Initialize instance variables. Initialize environment superclass. Acquire tensors."""

        self._get_base_yaml_params()

        self._sim_config = sim_config
        self._cfg = sim_config.config
        self._task_cfg = sim_config.task_config

        self._device = self._cfg["sim_device"]
        self._env_spacing = self.cfg_base["env"]["env_spacing"]
        self._num_envs = self._task_cfg["env"]["numEnvs"]
        self._num_observations = self._task_cfg["env"]["numObservations"]
        self._num_actions = self._task_cfg["env"]["numActions"]
        self._action_speed_scale = self._task_cfg["env"]["actionSpeedScale"]
        self._max_episode_length = self._task_cfg["env"]["maxEpisodeLength"]
        self._dt = torch.tensor(self._task_cfg["sim"]["dt"] * self._task_cfg["env"]["controlFrequencyInv"], device=self._device)

        # Start at 'home' positions
        self.torso_start = torch.tensor([0.1], device=self._device)
        self.base_start = torch.tensor([0.0, 0.0, 0.0], device=self._device)
        self.arm_start = torch.tensor([0.1, -1.570796, 0.0, -0.392699, 0.0], device=self._device)
        self.gripper_proximal_start = torch.tensor([0.75, 0.75], device=self._device)

        self.initial_dof_positions = torch.tensor([0.0, 0.0, 0.0, 0.1, 0.1, -1.570796, 0.0, 0.0, 0.0, -0.392699, 0.0, 0.75, 0.75, 0.0, 0.0], device=self._device)
        self.initial_dof_velocities = torch.tensor([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], device=self._device)

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

        super().__init__(name, env)

    def _get_base_yaml_params(self):
        """Initialize instance variables from YAML files."""

        cs = hydra.core.config_store.ConfigStore.instance()
        cs.store(name='factory_schema_config_base', node=FactorySchemaConfigBase)

        config_path = 'task/FactoryBase.yaml' # relative to Gym's Hydra search path (cfg dir)
        self.cfg_base = hydra.compose(config_name=config_path)
        self.cfg_base = self.cfg_base['task'] # strip superfluous nesting

        asset_info_path = '../tasks/factory/yaml/factory_asset_info_hsr_table.yaml' # relative to Gym's Hydra search path (cfg dir)
        self.asset_info_hsr_table = hydra.compose(config_name=asset_info_path)
        self.asset_info_hsr_table = self.asset_info_hsr_table['']['']['']['tasks']['factory']['yaml'] # strip superfluous nesting

    def import_hsr_assets(self):
        """Set HSR and table asset options. Import assets."""
        self._stage = get_current_stage()

        hsr_translation = np.array([self.cfg_base.env.hsr_depth, 0.0, 0.03])
        hsr_orientation = np.array([0.0, 0.0, 0.0, 1.0])

        hsr = HSR(
            prim_path=self.default_zero_env_path + "/hsrb", 
            name="hsrb",
            translation=hsr_translation, 
            orientation=hsr_orientation,
        )
        self._sim_config.apply_articulation_settings(
            "hsrb", 
            get_prim_at_path(hsr.prim_path), 
            self._sim_config.parse_actor_config("hsrb")
        )

        for link_prim in hsr.prim.GetChildren():
            if link_prim.HasAPI(PhysxSchema.PhysxRigidBodyAPI): 
                rb = PhysxSchema.PhysxRigidBodyAPI.Get(self._stage, link_prim.GetPrimPath())
                rb.GetDisableGravityAttr().Set(True)
                rb.GetRetainAccelerationsAttr().Set(False)
                if self.cfg_base.sim.add_damping:
                    rb.GetLinearDampingAttr().Set(1.0) # default = 0.0; increased to improve stability
                    rb.GetMaxLinearVelocityAttr().Set(1.0) # default = 1000.0; reduced to prevent CUDA errors
                    rb.GetAngularDampingAttr().Set(5.0) # default = 0.5; increased to improve stability
                    rb.GetMaxAngularVelocityAttr().Set(2/math.pi*180) # default = 64.0; reduced to prevent CUDA errors
                else:
                    rb.GetLinearDampingAttr().Set(0.0)
                    rb.GetMaxLinearVelocityAttr().Set(1000.0)
                    rb.GetAngularDampingAttr().Set(0.5)
                    rb.GetMaxAngularVelocityAttr().Set(64/math.pi*180)

        table_translation = np.array([0.0, 0.0, self.cfg_base.env.table_height * 0.5])
        table_orientation = np.array([1.0, 0.0, 0.0, 0.0])

        table = FixedCuboid(
            prim_path=self.default_zero_env_path + "/table",
            name="table",
            translation=table_translation,
            orientation=table_orientation,
            scale=np.array([
                self.asset_info_hsr_table.table_depth, 
                self.asset_info_hsr_table.table_width, 
                self.cfg_base.env.table_height,
            ]),
            size=1.0,
            color=np.array([0, 0, 0]),
        )

        self.parse_controller_spec()

    def parse_controller_spec(self):
        """Parse controller specification into lower-level controller configuration."""

        cfg_ctrl_keys = {'num_envs',
                         'jacobian_type',
                         'gripper_prop_gains',
                         'gripper_deriv_gains',
                         'motor_ctrl_mode',
                         'gain_space',
                         'ik_method',
                         'joint_prop_gains',
                         'joint_deriv_gains',
                         'do_motion_ctrl',
                         'task_prop_gains',
                         'task_deriv_gains',
                         'do_inertial_comp',
                         'motion_ctrl_axes',
                         'do_force_ctrl',
                         'force_ctrl_method',
                         'wrench_prop_gains',
                         'force_ctrl_axes'}
        self.cfg_ctrl = {cfg_ctrl_key: None for cfg_ctrl_key in cfg_ctrl_keys}

        self.cfg_ctrl['num_envs'] = self.num_envs
        self.cfg_ctrl['jacobian_type'] = self.cfg_task.ctrl.all.jacobian_type
        self.cfg_ctrl['gripper_prop_gains'] = torch.tensor(self.cfg_task.ctrl.all.gripper_prop_gains,
                                                           device=self.device).repeat((self.num_envs, 1))
        self.cfg_ctrl['gripper_deriv_gains'] = torch.tensor(self.cfg_task.ctrl.all.gripper_deriv_gains,
                                                            device=self.device).repeat((self.num_envs, 1))

        ctrl_type = self.cfg_task.ctrl.ctrl_type
        if ctrl_type == 'gym_default':
            self.cfg_ctrl['motor_ctrl_mode'] = 'gym'
            self.cfg_ctrl['gain_space'] = 'joint'
            self.cfg_ctrl['ik_method'] = self.cfg_task.ctrl.gym_default.ik_method
            self.cfg_ctrl['joint_prop_gains'] = torch.tensor(self.cfg_task.ctrl.gym_default.joint_prop_gains,
                                                             device=self.device).repeat((self.num_envs, 1))
            self.cfg_ctrl['joint_deriv_gains'] = torch.tensor(self.cfg_task.ctrl.gym_default.joint_deriv_gains,
                                                              device=self.device).repeat((self.num_envs, 1))
            self.cfg_ctrl['gripper_prop_gains'] = torch.tensor(self.cfg_task.ctrl.gym_default.gripper_prop_gains,
                                                               device=self.device).repeat((self.num_envs, 1))
            self.cfg_ctrl['gripper_deriv_gains'] = torch.tensor(self.cfg_task.ctrl.gym_default.gripper_deriv_gains,
                                                                device=self.device).repeat((self.num_envs, 1))
        elif ctrl_type == 'joint_space_ik':
            self.cfg_ctrl['motor_ctrl_mode'] = 'manual'
            self.cfg_ctrl['gain_space'] = 'joint'
            self.cfg_ctrl['ik_method'] = self.cfg_task.ctrl.joint_space_ik.ik_method
            self.cfg_ctrl['joint_prop_gains'] = torch.tensor(self.cfg_task.ctrl.joint_space_ik.joint_prop_gains,
                                                             device=self.device).repeat((self.num_envs, 1))
            self.cfg_ctrl['joint_deriv_gains'] = torch.tensor(self.cfg_task.ctrl.joint_space_ik.joint_deriv_gains,
                                                              device=self.device).repeat((self.num_envs, 1))
            self.cfg_ctrl['do_inertial_comp'] = False
        elif ctrl_type == 'joint_space_id':
            self.cfg_ctrl['motor_ctrl_mode'] = 'manual'
            self.cfg_ctrl['gain_space'] = 'joint'
            self.cfg_ctrl['ik_method'] = self.cfg_task.ctrl.joint_space_id.ik_method
            self.cfg_ctrl['joint_prop_gains'] = torch.tensor(self.cfg_task.ctrl.joint_space_id.joint_prop_gains,
                                                             device=self.device).repeat((self.num_envs, 1))
            self.cfg_ctrl['joint_deriv_gains'] = torch.tensor(self.cfg_task.ctrl.joint_space_id.joint_deriv_gains,
                                                              device=self.device).repeat((self.num_envs, 1))
            self.cfg_ctrl['do_inertial_comp'] = True
        elif ctrl_type == 'task_space_impedance':
            self.cfg_ctrl['motor_ctrl_mode'] = 'manual'
            self.cfg_ctrl['gain_space'] = 'task'
            self.cfg_ctrl['do_motion_ctrl'] = True
            self.cfg_ctrl['task_prop_gains'] = torch.tensor(self.cfg_task.ctrl.task_space_impedance.task_prop_gains,
                                                            device=self.device).repeat((self.num_envs, 1))
            self.cfg_ctrl['task_deriv_gains'] = torch.tensor(self.cfg_task.ctrl.task_space_impedance.task_deriv_gains,
                                                             device=self.device).repeat((self.num_envs, 1))
            self.cfg_ctrl['do_inertial_comp'] = False
            self.cfg_ctrl['motion_ctrl_axes'] = torch.tensor(self.cfg_task.ctrl.task_space_impedance.motion_ctrl_axes,
                                                             device=self.device).repeat((self.num_envs, 1))
            self.cfg_ctrl['do_force_ctrl'] = False
        elif ctrl_type == 'operational_space_motion':
            self.cfg_ctrl['motor_ctrl_mode'] = 'manual'
            self.cfg_ctrl['gain_space'] = 'task'
            self.cfg_ctrl['do_motion_ctrl'] = True
            self.cfg_ctrl['task_prop_gains'] = torch.tensor(self.cfg_task.ctrl.operational_space_motion.task_prop_gains,
                                                            device=self.device).repeat((self.num_envs, 1))
            self.cfg_ctrl['task_deriv_gains'] = torch.tensor(
                self.cfg_task.ctrl.operational_space_motion.task_deriv_gains, device=self.device).repeat(
                (self.num_envs, 1))
            self.cfg_ctrl['do_inertial_comp'] = True
            self.cfg_ctrl['motion_ctrl_axes'] = torch.tensor(
                self.cfg_task.ctrl.operational_space_motion.motion_ctrl_axes, device=self.device).repeat(
                (self.num_envs, 1))
            self.cfg_ctrl['do_force_ctrl'] = False
        elif ctrl_type == 'open_loop_force':
            self.cfg_ctrl['motor_ctrl_mode'] = 'manual'
            self.cfg_ctrl['gain_space'] = 'task'
            self.cfg_ctrl['do_motion_ctrl'] = False
            self.cfg_ctrl['do_force_ctrl'] = True
            self.cfg_ctrl['force_ctrl_method'] = 'open'
            self.cfg_ctrl['force_ctrl_axes'] = torch.tensor(self.cfg_task.ctrl.open_loop_force.force_ctrl_axes,
                                                            device=self.device).repeat((self.num_envs, 1))
        elif ctrl_type == 'closed_loop_force':
            self.cfg_ctrl['motor_ctrl_mode'] = 'manual'
            self.cfg_ctrl['gain_space'] = 'task'
            self.cfg_ctrl['do_motion_ctrl'] = False
            self.cfg_ctrl['do_force_ctrl'] = True
            self.cfg_ctrl['force_ctrl_method'] = 'closed'
            self.cfg_ctrl['wrench_prop_gains'] = torch.tensor(self.cfg_task.ctrl.closed_loop_force.wrench_prop_gains,
                                                              device=self.device).repeat((self.num_envs, 1))
            self.cfg_ctrl['force_ctrl_axes'] = torch.tensor(self.cfg_task.ctrl.closed_loop_force.force_ctrl_axes,
                                                            device=self.device).repeat((self.num_envs, 1))
        elif ctrl_type == 'hybrid_force_motion':
            self.cfg_ctrl['motor_ctrl_mode'] = 'manual'
            self.cfg_ctrl['gain_space'] = 'task'
            self.cfg_ctrl['do_motion_ctrl'] = True
            self.cfg_ctrl['task_prop_gains'] = torch.tensor(self.cfg_task.ctrl.hybrid_force_motion.task_prop_gains,
                                                            device=self.device).repeat((self.num_envs, 1))
            self.cfg_ctrl['task_deriv_gains'] = torch.tensor(self.cfg_task.ctrl.hybrid_force_motion.task_deriv_gains,
                                                             device=self.device).repeat((self.num_envs, 1))
            self.cfg_ctrl['do_inertial_comp'] = True
            self.cfg_ctrl['motion_ctrl_axes'] = torch.tensor(self.cfg_task.ctrl.hybrid_force_motion.motion_ctrl_axes,
                                                             device=self.device).repeat((self.num_envs, 1))
            self.cfg_ctrl['do_force_ctrl'] = True
            self.cfg_ctrl['force_ctrl_method'] = 'closed'
            self.cfg_ctrl['wrench_prop_gains'] = torch.tensor(self.cfg_task.ctrl.hybrid_force_motion.wrench_prop_gains,
                                                              device=self.device).repeat((self.num_envs, 1))
            self.cfg_ctrl['force_ctrl_axes'] = torch.tensor(self.cfg_task.ctrl.hybrid_force_motion.force_ctrl_axes,
                                                            device=self.device).repeat((self.num_envs, 1))

        if self.cfg_ctrl['motor_ctrl_mode'] == 'gym':
            # base joints
            joint_prim = self._stage.GetPrimAtPath(self.default_zero_env_path + f"/hsrb/base_footprint/joint_x")
            drive = UsdPhysics.DriveAPI.Apply(joint_prim, "linear")
            drive.GetStiffnessAttr().Set(self.cfg_ctrl['joint_deriv_gains'][0, 0].item())
            drive.GetDampingAttr().Set(self.cfg_ctrl['joint_deriv_gains'][0, 0].item())

            joint_prim = self._stage.GetPrimAtPath(self.default_zero_env_path + f"/hsrb/link_x/joint_y")
            drive = UsdPhysics.DriveAPI.Apply(joint_prim, "linear")
            drive.GetStiffnessAttr().Set(self.cfg_ctrl['joint_deriv_gains'][0, 0].item())
            drive.GetDampingAttr().Set(self.cfg_ctrl['joint_deriv_gains'][0, 0].item())

            joint_prim = self._stage.GetPrimAtPath(self.default_zero_env_path + f"/hsrb/link_y/joint_rz")
            drive = UsdPhysics.DriveAPI.Apply(joint_prim, "angular")
            drive.GetStiffnessAttr().Set(self.cfg_ctrl['joint_deriv_gains'][0, 0].item()*np.pi/180)
            drive.GetDampingAttr().Set(self.cfg_ctrl['joint_deriv_gains'][0, 0].item()*np.pi/180)

            # arm joints
            joint_prim = self._stage.GetPrimAtPath(self.default_zero_env_path + f"/hsrb/base_link/arm_lift_joint")
            drive = UsdPhysics.DriveAPI.Apply(joint_prim, "linear")
            drive.GetStiffnessAttr().Set(self.cfg_ctrl['joint_deriv_gains'][0, 0].item())
            drive.GetDampingAttr().Set(self.cfg_ctrl['joint_deriv_gains'][0, 0].item())

            joint_prim = self._stage.GetPrimAtPath(self.default_zero_env_path + f"/hsrb/arm_lift_link/arm_flex_joint")
            drive = UsdPhysics.DriveAPI.Apply(joint_prim, "angular")
            drive.GetStiffnessAttr().Set(self.cfg_ctrl['joint_deriv_gains'][0, 0].item()*np.pi/180)
            drive.GetDampingAttr().Set(self.cfg_ctrl['joint_deriv_gains'][0, 0].item()*np.pi/180)

            joint_prim = self._stage.GetPrimAtPath(self.default_zero_env_path + f"/hsrb/arm_flex_link/arm_roll_joint")
            drive = UsdPhysics.DriveAPI.Apply(joint_prim, "angular")
            drive.GetStiffnessAttr().Set(self.cfg_ctrl['joint_deriv_gains'][0, 0].item()*np.pi/180)
            drive.GetDampingAttr().Set(self.cfg_ctrl['joint_deriv_gains'][0, 0].item()*np.pi/180)

            joint_prim = self._stage.GetPrimAtPath(self.default_zero_env_path + f"/hsrb/arm_roll_link/wrist_flex_joint")
            drive = UsdPhysics.DriveAPI.Apply(joint_prim, "angular")
            drive.GetStiffnessAttr().Set(self.cfg_ctrl['joint_deriv_gains'][0, 0].item()*np.pi/180)
            drive.GetDampingAttr().Set(self.cfg_ctrl['joint_deriv_gains'][0, 0].item()*np.pi/180)

            joint_prim = self._stage.GetPrimAtPath(self.default_zero_env_path + f"/hsrb/wrist_flex_link/wrist_roll_joint")
            drive = UsdPhysics.DriveAPI.Apply(joint_prim, "angular")
            drive.GetStiffnessAttr().Set(self.cfg_ctrl['joint_deriv_gains'][0, 0].item()*np.pi/180)
            drive.GetDampingAttr().Set(self.cfg_ctrl['joint_deriv_gains'][0, 0].item()*np.pi/180)

            # gripper joints
            joint_prim = self._stage.GetPrimAtPath(self.default_zero_env_path + f"/hsrb/hand_palm_link/hand_l_proximal_joint")
            drive = UsdPhysics.DriveAPI.Apply(joint_prim, "angular")
            drive.GetStiffnessAttr().Set(self.cfg_ctrl['gripper_deriv_gains'][0, 0].item()*np.pi/180)
            drive.GetDampingAttr().Set(self.cfg_ctrl['gripper_deriv_gains'][0, 0].item()*np.pi/180)

            joint_prim = self._stage.GetPrimAtPath(self.default_zero_env_path + f"/hsrb/hand_palm_link/hand_r_proximal_joint")
            drive = UsdPhysics.DriveAPI.Apply(joint_prim, "angular")
            drive.GetStiffnessAttr().Set(self.cfg_ctrl['gripper_deriv_gains'][0, 0].item()*np.pi/180)
            drive.GetDampingAttr().Set(self.cfg_ctrl['gripper_deriv_gains'][0, 0].item()*np.pi/180)

        elif self.cfg_ctrl['motor_ctrl_mode'] == 'manual':
            # base joints
            joint_prim = self._stage.GetPrimAtPath(self.default_zero_env_path + f"/hsrb/base_footprint/joint_x")
            joint_prim.RemoveAPI(UsdPhysics.DriveAPI, "linear")
            drive = UsdPhysics.DriveAPI.Apply(joint_prim, "None")
            drive.GetStiffnessAttr().Set(0.0)
            drive.GetDampingAttr().Set(0.0)

            joint_prim = self._stage.GetPrimAtPath(self.default_zero_env_path + f"/hsrb/link_x/joint_y")
            joint_prim.RemoveAPI(UsdPhysics.DriveAPI, "linear")
            drive = UsdPhysics.DriveAPI.Apply(joint_prim, "None")
            drive.GetStiffnessAttr().Set(0.0)
            drive.GetDampingAttr().Set(0.0)

            joint_prim = self._stage.GetPrimAtPath(self.default_zero_env_path + f"/hsrb/link_y/joint_rz")
            joint_prim.RemoveAPI(UsdPhysics.DriveAPI, "angular")
            drive = UsdPhysics.DriveAPI.Apply(joint_prim, "angular")
            drive.GetStiffnessAttr().Set(0.0)
            drive.GetDampingAttr().Set(0.0)

            # arm joints
            joint_prim = self._stage.GetPrimAtPath(self.default_zero_env_path + f"/hsrb/base_link/arm_lift_joint")
            joint_prim.RemoveAPI(UsdPhysics.DriveAPI, "linear")
            drive = UsdPhysics.DriveAPI.Apply(joint_prim, "None")
            drive.GetStiffnessAttr().Set(0.0)
            drive.GetDampingAttr().Set(0.0)

            joint_prim = self._stage.GetPrimAtPath(self.default_zero_env_path + f"/hsrb/arm_lift_link/arm_flex_joint")
            joint_prim.RemoveAPI(UsdPhysics.DriveAPI, "angular")
            drive = UsdPhysics.DriveAPI.Apply(joint_prim, "None")
            drive.GetStiffnessAttr().Set(0.0)
            drive.GetDampingAttr().Set(0.0)

            joint_prim = self._stage.GetPrimAtPath(self.default_zero_env_path + f"/hsrb/arm_flex_link/arm_roll_joint")
            joint_prim.RemoveAPI(UsdPhysics.DriveAPI, "angular")
            drive = UsdPhysics.DriveAPI.Apply(joint_prim, "None")
            drive.GetStiffnessAttr().Set(0.0)
            drive.GetDampingAttr().Set(0.0)

            joint_prim = self._stage.GetPrimAtPath(self.default_zero_env_path + f"/hsrb/arm_roll_link/wrist_flex_joint")
            joint_prim.RemoveAPI(UsdPhysics.DriveAPI, "angular")
            drive = UsdPhysics.DriveAPI.Apply(joint_prim, "None")
            drive.GetStiffnessAttr().Set(0.0)
            drive.GetDampingAttr().Set(0.0)

            joint_prim = self._stage.GetPrimAtPath(self.default_zero_env_path + f"/hsrb/wrist_flex_link/wrist_roll_joint")
            joint_prim.RemoveAPI(UsdPhysics.DriveAPI, "angular")
            drive = UsdPhysics.DriveAPI.Apply(joint_prim, "None")
            drive.GetStiffnessAttr().Set(0.0)
            drive.GetDampingAttr().Set(0.0)

            # gripper joints
            joint_prim = self._stage.GetPrimAtPath(self.default_zero_env_path + f"/hsrb/hand_palm_link/hand_l_proximal_joint")
            joint_prim.RemoveAPI(UsdPhysics.DriveAPI, "angular")
            drive = UsdPhysics.DriveAPI.Apply(joint_prim, "None")
            drive.GetStiffnessAttr().Set(0.0)
            drive.GetDampingAttr().Set(0.0)

            joint_prim = self._stage.GetPrimAtPath(self.default_zero_env_path + f"/hsrb/hand_palm_link/hand_r_proximal_joint")
            joint_prim.RemoveAPI(UsdPhysics.DriveAPI, "angular")
            drive = UsdPhysics.DriveAPI.Apply(joint_prim, "None")
            drive.GetStiffnessAttr().Set(0.0)
            drive.GetDampingAttr().Set(0.0)

    def post_reset(self):
        self.num_dofs = 10 # base(3) + arm(5) + gripper(2)
        self.env_pos = self._env_pos

        self.dof_pos = torch.zeros((self.num_envs, self.num_dofs), device=self.device)
        self.dof_vel = torch.zeros((self.num_envs, self.num_dofs), device=self.device)
        self.dof_torque = torch.zeros((self.num_envs, self.num_dofs), device=self.device)
        self.fingertip_contact_wrench = torch.zeros((self.num_envs, 6), device=self.device)

        self.ctrl_target_fingertip_midpoint_pos = torch.zeros((self.num_envs, 3), device=self.device)
        self.ctrl_target_fingertip_midpoint_quat = torch.zeros((self.num_envs, 4), device=self.device)
        self.ctrl_target_dof_pos = torch.zeros((self.num_envs, self.num_dofs), device=self.device)
        self.ctrl_target_gripper_dof_pos = torch.zeros((self.num_envs, 2), device=self.device)
        self.ctrl_target_fingertip_contact_wrench = torch.zeros((self.num_envs, 6), device=self.device)

        self.prev_actions = torch.zeros((self.num_envs, self._num_actions), device=self.device)
        self.initial_robot_pos, self.initial_robot_rot = self.hsrs.get_world_poses()

        self.set_dof_idxs()
        self.set_dof_limits()
        self.set_default_state()
        self.set_joint_gains()
        self.set_joint_frictions()

    def set_dof_idxs(self):
        [self.torso_dof_idx.append(self.hsrs.get_dof_index(name)) for name in self._torso_joint_name]
        [self.base_dof_idxs.append(self.hsrs.get_dof_index(name)) for name in self._base_joint_names]
        [self.arm_dof_idxs.append(self.hsrs.get_dof_index(name)) for name in self._arm_names]
        [self.gripper_proximal_dof_idxs.append(self.hsrs.get_dof_index(name)) for name in self._gripper_proximal_names]

        # Movable joints
        self.actuated_dof_indices = torch.LongTensor(self.base_dof_idxs+self.arm_dof_idxs+self.gripper_proximal_dof_idxs).to(self._device) # torch.LongTensor([0, 1, 2, 3, 5, 7, 9, 10, 11, 12]).to(self._device)
        self.movable_dof_indices = torch.LongTensor(self.base_dof_idxs+self.arm_dof_idxs).to(self._device) # torch.LongTensor([0, 1, 2, 3, 5, 7, 9, 10]).to(self._device)

    def set_dof_limits(self): # dof position limits
        # (num_envs, num_dofs, 2)
        dof_limits = self.hsrs.get_dof_limits()
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
        joint_states = self.hsrs.get_joints_default_state()
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

        self.hsrs.set_joints_default_state(positions=jt_pos, velocities=jt_vel)

        # Initialize target positions
        self.dof_position_targets = jt_pos

    def set_joint_gains(self):
        self.hsrs.set_gains(kps=self.joint_kps, kds=self.joint_kds)

    def set_joint_frictions(self):
        self.hsrs.set_friction_coefficients(self.joint_friction_coefficients)

    def refresh_base_tensors(self):
        """Refresh tensors."""

        # No net contact force nor dof force
        self.dof_pos = self.hsrs.get_joint_positions(clone=False, joint_indices=self.actuated_dof_indices)
        self.dof_vel = self.hsrs.get_joint_velocities(clone=False, joint_indices=self.actuated_dof_indices)

        # jacobian shape: [num_envs, 50, 6, num_joints] (The root does not have a jacobian)
        jacobian_shape = self.hsrs.get_jacobian_shape()
        print('jacobian_shape:', jacobian_shape)
        self.hsr_jacobian = self.hsrs._physics_view.get_jacobians()
        self.hsr_mass_matrix = self.hsrs.get_mass_matrices(clone=False)

        self.base_dof_pos = self.dof_pos[:, 0:3]
        self.base_mass_matrix = self.hsr_mass_matrix[:, 0:3, 0:3] # for HSR base (not gripper)

        self.arm_dof_pos = self.dof_pos[:, 3:8]
        self.arm_mass_matrix = self.hsr_mass_matrix[:, 3:8, 3:8] # for HSR arm (not gripper)

        self.hand_pos, self.hand_quat = self.hsrs._hands.get_world_poses(clone=False)
        self.hand_pos -= self.env_pos

        hand_velocities = self.hsrs._hands.get_velocities(clone=False)
        self.hand_linvel = hand_velocities[:, 0:3]
        self.hand_angvel = hand_velocities[:, 3:6]

        self.left_finger_pos, self.left_finger_quat = self.hsrs._lfingers.get_world_poses(clone=False)
        self.left_finger_pos -= self.env_pos
        left_finger_velocities = self.hsrs._lfingers.get_velocities(clone=False)
        self.left_finger_linvel = left_finger_velocities[:, 0:3]
        self.left_finger_angvel = left_finger_velocities[:, 3:6]
        self.left_finger_jacobian = self.hsr_jacobian[:, 8, 0:6, self.actuated_dof_indices] # TODO: Specified by index?

        self.right_finger_pos, self.right_finger_quat = self.hsrs._rfingers.get_world_poses(clone=False)
        self.right_finger_pos -= self.env_pos
        right_finger_velocities = self.hsrs._rfingers.get_velocities(clone=False)
        self.right_finger_linvel = right_finger_velocities[:, 0:3]
        self.right_finger_angvel = right_finger_velocities[:, 3:6]
        self.right_finger_jacobian = self.hsr_jacobian[:, 9, 0:6, self.actuated_dof_indices] # TODO: Specified by index?

        # Cannot acquire proper net contact force in Isaac Sim at this moment
        self.left_finger_force = torch.zeros((self.num_envs, 3), device=self.device)
        self.right_finger_force = torch.zeros((self.num_envs, 3), device=self.device)

        self.gripper_dof_pos = self.dof_pos[:, 8:10]

        self.fingertip_centered_pos, self.fingertip_centered_quat = self.hsrs._fingertip_centered.get_world_poses(clone=False)
        self.fingertip_centered_pos -= self.env_pos
        fingertip_centered_velocities = self.hsrs._fingertip_centered.get_velocities(clone=False)
        self.fingertip_centered_linvel = fingertip_centered_velocities[:, 0:3]
        self.fingertip_centered_angvel = fingertip_centered_velocities[:, 3:6]
        self.fingertip_centered_jacobian = self.hsr_jacobian[:, 10, 0:6, self.actuated_dof_indices] # TODO: Specified by index?

        self.finger_midpoint_pos = (self.left_finger_pos + self.right_finger_pos) / 2
        self.fingertip_midpoint_pos = fc.translate_along_local_z(pos=self.finger_midpoint_pos,
                                                                 quat=self.hand_quat,
                                                                 offset=self.asset_info_hsr_table.hsr_finger_length,
                                                                 device=self.device)
        self.fingertip_midpoint_quat = self.fingertip_centered_quat # always equal

        self.fingertip_midpoint_linvel = self.fingertip_centered_linvel + torch.cross(self.fingertip_centered_angvel,
                                                                                      (self.fingertip_midpoint_pos - self.fingertip_centered_pos),
                                                                                      dim=1)

        self.fingertip_midpoint_angvel = self.fingertip_centered_angvel # always equal

        self.fingertip_midpoint_jacobian = (self.left_finger_jacobian + self.right_finger_jacobian) * 0.5

    def generate_ctrl_signals(self):
        """Get Jacobian. Set HSR DOF position targets or DOF torques."""

        # Get desired Jacobian
        if self.cfg_ctrl['jacobian_type'] == 'geometric':
            self.fingertip_midpoint_jacobian_tf = self.fingertip_midpoint_jacobian
        elif self.cfg_ctrl['jacobian_type'] == 'analytic':
            self.fingertip_midpoint_jacobian_tf = fc.get_analytic_jacobian(
                fingertip_quat=self.fingertip_quat,
                fingertip_jacobian=self.fingertip_midpoint_jacobian,
                num_envs=self.num_envs,
                device=self.device
            )

        # Set PD joint pos target or joint torque
        if self.cfg_ctrl['motor_ctrl_mode'] == 'gym':
            self._set_dof_pos_target()
        elif self.cfg_ctrl['motor_ctrl_mode'] == 'manual':
            self._set_dof_torque()
    
    def _set_dof_pos_target(self):
        """Set HSR DOF position target to move fingertips towards target pose."""

        self.ctrl_target_dof_pos = fc.compute_dof_pos_target(
            cfg_ctrl=self.cfg_ctrl,
            base_arm_dof_pos=self.dof_pos,
            fingertip_midpoint_pos=self.fingertip_midpoint_pos,
            fingertip_midpoint_quat=self.fingertip_midpoint_quat,
            jacobian=self.fingertip_midpoint_jacobian_tf,
            ctrl_target_fingertip_midpoint_pos=self.ctrl_target_fingertip_midpoint_pos,
            ctrl_target_fingertip_midpoint_quat=self.ctrl_target_fingertip_midpoint_quat,
            ctrl_target_gripper_dof_pos=self.ctrl_target_gripper_dof_pos,
            device=self.device
        )

        self.hsrs.set_joint_position_targets(positions=self.ctrl_target_dof_pos, joint_indices=self.actuated_dof_indices)

    def _set_dof_torque(self):
        """Set HSR DOF torque to move fingertips towards target pose."""

        self.dof_torque = fc.compute_dof_torque(
            cfg_ctrl=self.cfg_ctrl,
            dof_pos=self.dof_pos,
            dof_vel=self.dof_vel,
            fingertip_midpoint_pos=self.fingertip_midpoint_pos,
            fingertip_midpoint_quat=self.fingertip_midpoint_quat,
            fingertip_midpoint_linvel=self.fingertip_midpoint_linvel,
            fingertip_midpoint_angvel=self.fingertip_midpoint_angvel,
            left_finger_force=self.left_finger_force,
            right_finger_force=self.right_finger_force,
            jacobian=self.fingertip_midpoint_jacobian_tf,
            arm_mass_matrix=self.arm_mass_matrix,
            ctrl_target_gripper_dof_pos=self.ctrl_target_gripper_dof_pos,
            ctrl_target_fingertip_midpoint_pos=self.ctrl_target_fingertip_midpoint_pos,
            ctrl_target_fingertip_midpoint_quat=self.ctrl_target_fingertip_midpoint_quat,
            ctrl_target_fingertip_contact_wrench=self.ctrl_target_fingertip_contact_wrench,
            device=self.device
        )

        self.hsrs.set_joint_efforts(efforts=self.dof_torque, joint_indices=self.actuated_dof_indices)

    def enable_gravity(self, gravity_mag):
        """Enable gravity."""

        gravity = [0.0, 0.0, -9.81]
        self._env._world._physics_sim_view.set_gravity(carb.Float3(gravity[0], gravity[1], gravity[2]))

    def disable_gravity(self):
        """Disable gravity."""

        gravity = [0.0, 0.0, 0.0]
        self._env._world._physics_sim_view.set_gravity(carb.Float3(gravity[0], gravity[1], gravity[2]))