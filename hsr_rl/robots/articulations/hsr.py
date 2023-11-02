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


import math
import torch
import carb
import numpy as np
from typing import Optional
from omni.isaac.core.robots.robot import Robot
from omni.isaac.core.utils.nucleus import get_assets_root_path
from omni.isaac.core.utils.prims import get_prim_at_path
from omni.isaac.core.utils.stage import add_reference_to_stage
from hsr_rl.tasks.utils.usd_utils import set_drive
from hsr_rl.utils.files import get_usd_path
from pxr import PhysxSchema


class HSR(Robot):
    def __init__(
        self,
        prim_path: str,
        name: Optional[str] = "hsrb",
        usd_path: Optional[str] = None,
        translation: Optional[np.ndarray] = None,
        orientation: Optional[np.ndarray] = None,
    ) -> None:

        self._usd_path = usd_path
        self._name = name

        if self._usd_path is None:
            self._usd_path = (get_usd_path() / 'hsrb4s' / 'hsrb4s.usd').as_posix()

        add_reference_to_stage(self._usd_path, prim_path)

        super().__init__(
            prim_path=prim_path,
            name=name,
            translation=translation,
            orientation=orientation,
            articulation_controller=None,
        )

        dof_paths = [
            "base_footprint/joint_x",
            "link_x/joint_y",
            "link_y/joint_rz",
            "base_link/arm_lift_joint",
            "arm_lift_link/arm_flex_joint",
            "arm_flex_link/arm_roll_joint",
            "arm_roll_link/wrist_flex_joint",
            "wrist_flex_link/wrist_roll_joint",
            "hand_palm_link/hand_l_proximal_joint",
            "hand_palm_link/hand_r_proximal_joint"
        ]

        drive_type = ["linear", "linear", "angular", "linear", "angular", "angular", "angular", "angular", "angular", "angular"]
        default_dof_pos = [0.0] * 10
        stiffness = [1e9] * 8 + [500] * 2 # (base, arm, gripper)
        damping = [1.4] * 8 + [0.3] * 2 # (base, arm, gripper)
        max_force = [5e6] * 3 + [10000.0] * 5 + [360*np.pi/180] * 2 # (base, arm, gripper)
        max_velocity = [5e6] * 3 + [10000.0] * 5 + [math.degrees(2.175)] * 2 # (base, arm, gripper)

        for i, dof in enumerate(dof_paths):
            set_drive(
                prim_path=f"{self.prim_path}/{dof}",
                drive_type=drive_type[i],
                target_type="position",
                target_value=default_dof_pos[i],
                stiffness=stiffness[i],
                damping=damping[i],
                max_force=max_force[i]
            )

            PhysxSchema.PhysxJointAPI(get_prim_at_path(f"{self.prim_path}/{dof}")).CreateMaxJointVelocityAttr().Set(max_velocity[i])

        additional_dof_paths = [
            "base_link/torso_lift_joint",
            "torso_lift_link/head_pan_joint",
            "head_pan_link/head_tilt_joint",
            "hand_l_mimic_distal_link/hand_l_distal_joint",
            "hand_r_mimic_distal_link/hand_r_distal_joint",
        ]

        drive_type = ["linear", "angular", "angular", "angular", "angular"]
        default_dof_pos = [0.0, 0.0, 0.0, 0.0, 0.0]
        stiffness = [1e9, 1e9, 1e9, 1e9, 1e9]
        damping = [0.0, 0.0, 0.0, 0.0, 0.0]
        max_force = [10000.0, 10000.0, 10000.0, 10000.0, 10000.0]
        max_velocity = [10000.0, 10000.0, 10000.0, 10000.0, 10000.0]

        for i, dof in enumerate(additional_dof_paths):
            set_drive(
                prim_path=f"{self.prim_path}/{dof}",
                drive_type=drive_type[i],
                target_type="position",
                target_value=default_dof_pos[i],
                stiffness=stiffness[i],
                damping=damping[i],
                max_force=max_force[i]
            )

            PhysxSchema.PhysxJointAPI(get_prim_at_path(f"{self.prim_path}/{dof}")).CreateMaxJointVelocityAttr().Set(max_velocity[i])