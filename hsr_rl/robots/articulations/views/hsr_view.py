from typing import Optional

from omni.isaac.core.articulations import ArticulationView
from omni.isaac.core.prims import RigidPrimView


class HSRView(ArticulationView):
    def __init__(
        self,
        prim_paths_expr: str,
        name: Optional[str] = "HSRView",
    ) -> None:
        """[summary]
        """

        super().__init__(
            prim_paths_expr=prim_paths_expr,
            name=name,
            reset_xform_properties=False
        )

        self._hands = RigidPrimView(prim_paths_expr="/World/envs/.*/hsrb/wrist_roll_link", name="hands_view", reset_xform_properties=False)
        self._lfingers = RigidPrimView(prim_paths_expr="/World/envs/.*/hsrb/hand_l_distal_link", name="lfingers_view", reset_xform_properties=False)
        self._rfingers = RigidPrimView(prim_paths_expr="/World/envs/.*/hsrb/hand_r_distal_link", name="rfingers_view", reset_xform_properties=False)
        self._fingertip_centered = RigidPrimView(prim_paths_expr="/World/envs/.*/hsrb/hand_palm_link", name="fingertips_view", reset_xform_properties=False)

    def initialize(self, physics_sim_view):
        super().initialize(physics_sim_view)

        self._gripper_indices = [self.get_dof_index("hand_l_proximal_joint"), self.get_dof_index("hand_r_proximal_joint")]

    @property
    def gripper_indices(self):
        return self._gripper_indices