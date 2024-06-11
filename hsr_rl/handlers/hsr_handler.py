import torch
import math

from hsr_rl.handlers.base.base_handler import BaseHandler
from hsr_rl.robots.articulations.hsr import HSR
from hsr_rl.robots.articulations.views.hsr_view import HSRView
from hsr_rl.tasks.utils.ikfast_utils import HSRIKSolver
from hsr_rl.tasks.utils.iktorch_utils import HSRKinematic
from omni.isaac.core.utils.prims import get_prim_at_path
from omni.isaac.core.articulations import ArticulationView
from omni.isaac.core.utils.torch.maths import torch_rand_float, tensor_clamp
from omni.isaac.core.utils.stage import get_current_stage

class HSRHandler(BaseHandler):
    def __init__(self, move_group, use_gripper, sim_config, num_envs, device):
        self._move_group = move_group
        self._use_gripper = use_gripper
        self._sim_config = sim_config
        self._num_envs = num_envs
        self._device = device
        self._robot_positions = torch.tensor([0.0, 0.0, 0.03], device=self._device)
        self._robot_rotations = torch.tensor([1.0, 0.0, 0.0, 0.0], device=self._device)

        # Start at 'home' positions
        self.torso_start = torch.tensor([0.1], device=self._device)
        self.base_start = torch.tensor([0.0, 0.0, 0.0], device=self._device)
        self.arm_start = torch.tensor([0.1, -1.570796, 0.0, -0.392699, 0], device=self._device)
        self.gripper_proximal_start = torch.tensor([0.75, 0.75], device=self._device) # Opened gripper by default

        self.default_zero_env_path = "/World/envs/env_0"

        # Get dt for integrating velocity commands
        self.dt = torch.tensor(self._sim_config.task_config["sim"]["dt"]*self._sim_config.task_config["env"]["controlFrequencyInv"], device=self._device)

        # Scale
        self.action_base_scale = torch.tensor(self._sim_config.task_config["sim"]["action_base_scale"], device=self._device)
        self.action_arm_scale = torch.tensor(self._sim_config.task_config["sim"]["action_arm_scale"], device=self._device)
        self.action_gripper_scale = torch.tensor(self._sim_config.task_config["sim"]["action_gripper_scale"], device=self._device)

        # articulation View will be created later
        self.robots = None

        # joint & body names
        self._ee_name =  ["hand_palm_link"]
        self._torso_joint_name = ["torso_lift_joint"]
        self._base_joint_names = ["joint_x", "joint_y", "joint_rz"]
        self._arm_names = ["arm_lift_joint", "arm_flex_joint", "arm_roll_joint", "wrist_flex_joint", "wrist_roll_joint"]
        self._gripper_proximal_names = ["hand_l_proximal_joint", "hand_r_proximal_joint"]

        # values are set in post_reset after model is loaded
        self.torso_dof_idx = []
        self.base_dof_idxs = []
        self.arm_dof_idxs = []
        self.gripper_proximal_dof_idxs = []

        self.whole_body_dof_idxs = []
        self.combined_dof_idxs = []

        # dof joint position limits
        self.torso_dof_lower = []
        self.torso_dof_upper = []
        self.base_dof_lower = []
        self.base_dof_upper = []
        self.arm_dof_lower = []
        self.arm_dof_upper = []
        self.gripper_p_dof_lower = []
        self.gripper_p_dof_upper = []

        # HSR kinematics for inverse kinematics
        self.hsr_kinematics = HSRKinematic(device)

    def import_robot_assets(self):
        # get stage
        self._stage = get_current_stage()

        # make it in task and use handler as getter for path
        hsr = HSR(prim_path=self.default_zero_env_path + "/hsrb", name="hsrb",
                          translation=self._robot_positions, orientation=self._robot_rotations)

        # Optional: Apply additional articulation settings
        self._sim_config.apply_articulation_settings("hsrb", get_prim_at_path(hsr.prim_path),
                                                self._sim_config.parse_actor_config("hsrb"))

    # call it in setup_up_scene in Task
    def create_articulation_view(self):
        self.robots = HSRView(prim_paths_expr="/World/envs/.*/hsrb", name="hsrb_view")
        return self.robots

    def post_reset(self):
        # reset that takes place when the isaac world is reset (typically happens only once)
        self._set_dof_idxs()

        # set dof limits
        self._set_dof_limits()

        # set new default state for reset
        self._set_default_state()

    def _set_dof_idxs(self):
        [self.torso_dof_idx.append(self.robots.get_dof_index(name)) for name in self._torso_joint_name]
        [self.base_dof_idxs.append(self.robots.get_dof_index(name)) for name in self._base_joint_names]
        [self.arm_dof_idxs.append(self.robots.get_dof_index(name)) for name in self._arm_names]
        [self.gripper_proximal_dof_idxs.append(self.robots.get_dof_index(name)) for name in self._gripper_proximal_names]

        # Add whole body joint idxs when move_group is "whole_body"
        if self._move_group == "whole_body":
            self.whole_body_dof_idxs = self.base_dof_idxs + self.arm_dof_idxs
        else:
            raise ValueError('move_group not defined')

        self.combined_dof_idxs = self.whole_body_dof_idxs + self.torso_dof_idx + self.gripper_proximal_dof_idxs

    def _set_dof_limits(self): # dof position limits
        # (num_envs, num_dofs, 2)
        dof_limits = self.robots.get_dof_limits()
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

    def _set_default_state(self):
        # Set default joint state
        joint_states = self.robots.get_joints_default_state()
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

        self.robots.set_joints_default_state(positions=jt_pos, velocities=jt_vel)

    ### Actions

    def modify_torso(self, jt_pos):
        dof_pos = self.get_arm_dof_pos()
        scaled_arm_lift_pos = dof_pos[:, 0] / self.arm_dof_upper[0]
        scaled_torso_lift_pos = scaled_arm_lift_pos * self.torso_dof_upper[0]
        jt_pos[:, self.torso_dof_idx] = scaled_torso_lift_pos.unsqueeze(dim=1)
        return jt_pos

    def apply_actions(self, actions):
        # Actions are velocity commands
        jt_pos = self.robots.get_joint_positions()
        whole_body_jt_pos = jt_pos[:, self.whole_body_dof_idxs]

        ### Version1.
        th = torch.tensor(whole_body_jt_pos.clone(), requires_grad=True, device=self._device)
        x_axis_vec = torch.zeros([whole_body_jt_pos.shape[0], 3], device=self._device)
        x_axis_vec[:, 0] = 1.
        vel_actions = self.hsr_kinematics.batch_ik_vel(actions[:, :3], x_axis_vec, actions[:, 3], th)

        # Scale actions
        whole_body_jt_pos += vel_actions * self.action_arm_scale * self.dt
        jt_pos[:, self.whole_body_dof_idxs] = whole_body_jt_pos
        ###

        ### Verion2.
        th = torch.tensor(whole_body_jt_pos.clone(), requires_grad=True, device=self._device)
        target_vel = torch.eye(4).repeat(self._num_envs, 1, 1)
        target_vel[:, :3, 3] = actions[:, :3]
        vel_actions = self.hsr_kinematics.batch_base_ik_vel(target_vel, th)

        # Scale actions
        whole_body_jt_pos += vel_actions * self.action_arm_scale * self.dt
        jt_pos[:, self.whole_body_dof_idxs] = whole_body_jt_pos
        ###

        # Modify torso value
        jt_pos = self.modify_torso(jt_pos)

        # Apply modified aciton
        self.robots.set_joint_positions(positions=jt_pos)

    def apply_actions(self, actions):
        # Actions are velocity commands
        jt_pos = self.robots.get_joint_positions()
        whole_body_jt_pos = jt_pos[:, self.whole_body_dof_idxs]

        # Scale actions
        whole_body_jt_pos += actions * self.action_arm_scale * self.dt
        jt_pos[:, self.whole_body_dof_idxs] = whole_body_jt_pos

        # Modify torso value
        jt_pos = self.modify_torso(jt_pos)

        # Apply modified aciton
        self.robots.set_joint_positions(positions=jt_pos)

    ### Set joint positions

    def set_torso_positions(self, jnt_position):
        # Set torso joint to specific position
        self.robots.set_joint_positions(positions=jnt_position, joint_indices=self.torso_dof_idx)

    def set_base_positions(self, jnt_positions):
        # Set base joints to specific positions
        self.robots.set_joint_positions(positions=jnt_positions, joint_indices=self.base_dof_idxs)

    def set_arm_positions(self, jnt_positions):
        # Set upper body joints to specific positions
        self.robots.set_joint_positions(positions=jnt_positions, joint_indices=self.arm_dof_idxs)

    def set_gripper_positions(self, jnt_positions):
        # Set gripper position to specific positions
        self.robots.set_joint_positions(positions=jnt_positions, joint_indices=self.gripper_proximal_dof_idxs)

    def set_whole_body_positions(self, jnt_positions):
        # Set whole body positions
        self.robots.set_joint_positions(positions=jnt_positions, joint_indices=self.whole_body_dof_idxs)

    ### Get robot observation

    def get_robot_obs(self):
        # Return positions and velocities of upper body and base joints
        combined_pos = self.robots.get_joint_positions(joint_indices=self.whole_body_dof_idxs)

        # Base rotation continuous joint should be in range -pi to pi
        limits = (combined_pos[:, 2] > torch.pi)
        combined_pos[limits, 2] -= 2*torch.pi
        limits = (combined_pos[:, 2] < -torch.pi)
        combined_pos[limits, 2] += 2*torch.pi

        return combined_pos

    ### Get joint positions / velocities

    def get_torso_dof_pos(self):
        dof_pos = self.robots.get_joint_positions()
        torso_pos = dof_pos[:, self.torso_dof_idx]
        return torso_pos

    def get_torso_dof_vel(self):
        dof_vel = self.robots.get_joint_velocities()
        torso_vel = dof_vel[:, self.torso_dof_idx]
        return torso_vel

    def get_torso_dof_values(self):
        torso_pos = self.get_torso_dof_pos()
        torso_vel = self.get_torso_dof_vel()
        return torso_pos, torso_vel

    def get_base_dof_pos(self):
        dof_pos = self.robots.get_joint_positions()
        base_pos = dof_pos[:, self.base_dof_idxs]
        return base_pos

    def get_base_dof_vel(self):
        dof_vel = self.robots.get_joint_velocities()
        base_vel = dof_vel[:, self.base_dof_idxs]
        return base_vel

    def get_base_dof_values(self):
        base_pos = self.get_base_dof_pos()
        base_vel = self.get_base_dof_vel()
        return base_pos, base_vel

    def get_arm_dof_pos(self):
        # (num_envs, num_dof)
        dof_pos = self.robots.get_joint_positions()
        arm_pos = dof_pos[:, self.arm_dof_idxs]
        return arm_pos

    def get_arm_dof_vel(self):
        # (num_envs, num_dof)
        dof_vel = self.robots.get_joint_velocities()
        arm_vel = dof_vel[:, self.arm_dof_idxs]
        return arm_vel

    def get_arm_dof_values(self):
        arm_pos = self.get_arm_dof_pos()
        arm_vel = self.get_arm_dof_vel()
        return arm_pos, arm_vel

    def get_gripper_dof_pos(self):
        # (num_envs, num_dof)
        dof_pos = self.robots.get_joint_positions()
        gripper_p_pos = dof_pos[:, self.gripper_proximal_dof_idxs]
        return gripper_p_pos

    def get_gripper_dof_vel(self):
        # (num_envs, num_dof)
        dof_vel = self.robots.get_joint_velocities()
        gripper_p_vel = dof_vel[:, self.gripper_proximal_dof_idxs]
        return gripper_p_vel

    def get_gripper_dof_values(self):
        gripper_p_pos = self.gripper_dof_pos()
        gripper_p_vel = self.gripper_dof_vel()

        return gripper_p_pos, gripper_p_vel

    ### Reset

    def reset(self, env_ids, randomize=False):
        indices = env_ids.to(dtype=torch.int32)

        joint_states = self.robots.get_joints_default_state()
        jt_pos = joint_states.positions
        jt_vel = joint_states.velocities
        if randomize:
            noise = torch_rand_float(-0.5, 0.5, jt_pos[:, self.combined_dof_idxs].shape, device=self._device) # default: 0.75
            jt_pos[:, self.combined_dof_idxs] += noise # Optional: Add to default instead

        self.robots.set_joint_positions(jt_pos, indices=indices)
        self.robots.set_joint_velocities(jt_vel, indices=indices)