import os
import sys
import yaml
import time
import torch
import hydra
import numpy as np
from omegaconf import DictConfig

from hsr_rl.utils.hydra_cfg.hydra_utils import *
from hsr_rl.utils.hydra_cfg.reformat import omegaconf_to_dict, print_dict

from hsr_rl.utils.task_utils import initialize_task
from hsr_rl.envs.isaac_env_rlgames import IsaacEnvRlgames

# TODO: modify
sys.path.append('/root/tamp-hsr/hsr_tamp/experiments/gearbox_3d/')
sys.path.append('/root/tamp-hsr/hsr_ros/hsr_ws/src/gearbox_3d/script/')
from tamp_planner import TAMPPlanner
from path_interface import PlanModifier
from rl_controller.rl_agent import ResidualRL


def load_config(policy_name: str = 'Pick'):
    skill_name = 'HSRGearboxResidual' + policy_name + 'PPO.yaml' # TODO: modify
    file_name = os.path.join('/root/tamp-hsr/hsr_rl/cfg/train', skill_name)
    with open(file_name, 'r') as f:
        config = yaml.load(f, Loader=yaml.SafeLoader)

    return config

class ExecutePlan(object):
    def __init__(self):
        self.tamp_planner = TAMPPlanner()
        self.path_modifier = PlanModifier()

        # Load policy
        self.load_policy()

    def load_policy(self):
        # Load policies
        pick_yaml = load_config('Pick')
        place_yaml = load_config('Place')
        insert_yaml = load_config('Insert')

        pick_policy_cfg = omegaconf_to_dict(pick_yaml)
        place_policy_cfg = omegaconf_to_dict(place_yaml)
        insert_policy_cfg = omegaconf_to_dict(insert_yaml)

        print('pick_policy_cfg:', pick_policy_cfg)
        print('place_policy_cfg:', place_policy_cfg)
        print('insert_policy_cfg:', insert_policy_cfg)

        # Set action scale
        self.pick_action_scale = torch.tensor(pick_policy_cfg["params"]["config"]["action_scale"], device='cuda:0')
        self.place_action_scale = torch.tensor(place_policy_cfg["params"]["config"]["action_scale"], device='cuda:0')
        self.insert_action_scale = torch.tensor(insert_policy_cfg["params"]["config"]["action_scale"], device='cuda:0')

        # Skill based residual policy agents
        self.pick_agent = ResidualRL(pick_policy_cfg.get('params'))
        self.place_agent = ResidualRL(place_policy_cfg.get('params'))
        self.insert_agent = ResidualRL(insert_policy_cfg.get('params'))

        # Restore learned params
        self.pick_agent.restore(pick_policy_cfg["params"]["load_path"])
        self.place_agent.restore(place_policy_cfg["params"]["load_path"])
        self.insert_agent.restore(insert_policy_cfg["params"]["load_path"])

    def get_object_poses(self):
        # Add gearbox parts
        object_poses = {}
        for object_name, parts in self.env._task._parts.items():
            object_pos, object_rot = parts.get_world_poses(clone=False)
            object_pos = object_pos.squeeze(dim=0).to('cpu').detach().numpy().copy()
            object_rot = object_rot.squeeze(dim=0).to('cpu').detach().numpy().copy()
            object_poses[object_name] = (object_pos, object_rot)

        def modify_pos(gearbox_base_pos):
            gearbox_base_pos[0] += 0.27 # TODO: modify
            return gearbox_base_pos

        def modify_rot(gearbox_base_rot):
            return gearbox_base_rot

        # Add gearbox_base
        gearbox_base_pos = modify_pos(self.env._task.init_parts_pos['gearbox_base'])
        gearbox_base_rot = modify_rot(self.env._task.init_parts_orn['gearbox_base'])

        # To cpu
        gearbox_base_pos = gearbox_base_pos.to('cpu').detach().numpy().copy()
        gearbox_base_rot = gearbox_base_rot.to('cpu').detach().numpy().copy()

        # Concatenate
        object_poses['base'] = (gearbox_base_pos, gearbox_base_rot)

        return object_poses

    def get_robot_poses(self, device='cpu'):
        # Get robot poses
        robot_poses = self.env._task._robots.get_joint_positions(
            joint_indices=self.env._task.movable_dof_indices, clone=False)

        # To cpu
        if device == 'cpu':
            robot_poses = robot_poses.squeeze(dim=0).to('cpu').detach().numpy().copy()
        return robot_poses

    def calculate_command(self, curr_pose, target_pose):
        # Calculate control step
        control_step = torch.unsqueeze(self.env._task.progress_buf % self.control_freq_inv + 1, dim=1)

        # Calculate delta command
        delta_pose = target_pose - curr_pose
        delta_pose = delta_pose / self.control_freq_inv * control_step
        return delta_pose

    def initialize_tamp(self):
        object_poses = self.get_object_poses()
        robot_poses = self.get_robot_poses()
        observations = (robot_poses, object_poses)

        # Initialize problem
        self.tamp_planner.initialize(observations)

    def plan(self):
        # Run TAMP
        plan, _, _ = self.tamp_planner.plan()

        return plan

    def augment_plan(self, plan):
        # Replay_trajectory
        return self.tamp_planner.execute(plan)

    def process(self, action_name, args):
        # Get object names
        object_names = self.tamp_planner.tamp_problem.body_names

        # Modify plan
        action_name, object_name, modified_action = self.path_modifier.post_process(action_name, object_names, args)

        return action_name, object_name, modified_action

    def execute(self, sim_cfg):
        # IsaacEnv settings
        rank = int(os.getenv("LOCAL_RANK", "0"))
        sim_cfg.task_name = 'HSRGearboxResidualAll'
        sim_cfg.device_id = rank
        sim_cfg.rl_device = f'cuda:{rank}'
        enable_viewport = "enable_cameras" in sim_cfg.task.sim and sim_cfg.task.sim.enable_cameras
        self.dt = sim_cfg.task.sim.dt * sim_cfg.task.env.controlFrequencyInv
        self.control_freq_inv = sim_cfg.task.env.controlFrequencyInv
        self.env = IsaacEnvRlgames(headless=False, sim_device=sim_cfg.device_id, enable_livestream=sim_cfg.enable_livestream, enable_viewport=enable_viewport)

        # Parse hydra config to dict
        cfg_dict = omegaconf_to_dict(sim_cfg)
        print_dict(cfg_dict)

        # Sets seed, if seed is -1 will pick a random one
        from omni.isaac.core.utils.torch.maths import set_seed
        sim_cfg.seed = set_seed(sim_cfg.seed, torch_deterministic=sim_cfg.torch_deterministic)
        cfg_dict['seed'] = sim_cfg.seed
        initialize_task(cfg_dict, self.env)

        # Execute trajectory
        simulation_count = 0
        while self.env._simulation_app.is_running():
            if self.env._world.is_playing():
                if self.env._world.current_time_step_index == 0:
                    self.env._world.reset(soft=True)

                if simulation_count == 0:
                    simulation_count += 1
                    self.env._world.step(render=True)
                    self.env.sim_frame_count += 1
                    continue

                # Initialize TAMP
                self.initialize_tamp()

                # Execute planning
                plan = self.plan()

                if plan is None:
                    return None

                # Augment plan
                move_metadata, pick_metadata, place_metadata, insert_metadata = self.augment_plan(plan)

                # For metadata
                move_cnt, pick_cnt, place_cnt, insert_cnt = 0, 0, 0, 0

                for i, (action_name, args) in enumerate(plan):
                    action_name, object_name, modified_action = self.process(action_name, args)

                    # Move_base action
                    if action_name == 'move_base':
                        move_traj = modified_action

                        for target_pose in move_traj:
                            target_base_pose = torch.tensor(target_pose[:3], dtype=torch.float, device='cuda:0')
                            target_arm_pose = self.env._task._robots.get_joint_positions(
                                joint_indices=self.env._task.arm_dof_idxs, clone=False).squeeze(dim=0)
                            action = torch.cat((target_base_pose, target_arm_pose), dim=0).to(torch.float32)

                            # Step simulation
                            self.env._task.pre_physics_step(action)
                            self.env._world.step(render=True)
                            self.env.sim_frame_count += 1
                            self.env._task.post_physics_step(action_name)

                        # Add move count
                        move_cnt += 1

                    # Pick action
                    elif action_name == 'pick':
                        pick_traj, return_traj = modified_action

                        # Get target poses from pick_metadata
                        target_ee_pose = pick_metadata['target_robot_pose'][pick_cnt]
                        target_parts_pose = pick_metadata['target_object_pose'][pick_cnt]

                        finish = False
                        loop_count = 0

                        # Execute pick trajectory
                        while not finish: # pick
                            target_pose = torch.tensor(pick_traj[loop_count], dtype=torch.float, device='cuda:0')
                            target_pose = torch.unsqueeze(target_pose, dim=0)

                            # Get observation
                            obs = self.env._task.get_pick_observations(target_ee_pose, target_parts_pose, object_name)

                            # Residual action
                            with torch.no_grad():
                                actions = self.pick_agent.get_action(obs)

                            # Multiply target 6D pose and residual 6D pose
                            jt_action = self.dt * self.pick_action_scale * actions.to('cuda:0')

                            # Add delta pose to reference trajectory
                            dof_pos = self.get_robot_poses(device='cuda:0')
                            target_pose += self.calculate_command(dof_pos, target_pose)
                            target_pose += jt_action

                            # Step simulation
                            self.env._task.pre_physics_step(target_pose)
                            self.env._world.step(render=True)
                            self.env.sim_frame_count += 1
                            self.env._task.post_physics_step(action_name, object_name=object_name,
                                                             target_ee_pose=target_ee_pose, target_parts_pose=target_parts_pose)

                            loop_count += 1
                            if loop_count >= len(pick_traj):
                                loop_count = len(pick_traj)-1
                                finish = True # bool(self.env._task._check_pick_success(object_name).squeeze(dim=0))

                        # Simulate close gripper steps
                        self.env._task._close_gripper(sim_steps=3)

                        for target_pose in return_traj: # return
                            target_pose = torch.tensor(target_pose, dtype=torch.float, device='cuda:0')

                            # Step simulation
                            self.env._task.pre_physics_step(target_pose)
                            self.env._world.step(render=True)
                            self.env.sim_frame_count += 1
                            self.env._task.post_physics_step(action_name, object_name=object_name,
                                                             target_ee_pose=target_ee_pose, target_parts_pose=target_parts_pose)

                        # Check pick success
                        pick_success = self.env._task._check_pick_success(object_name).squeeze(dim=0)
                        print('pick_scucess:', pick_success)

                        # Add pick count
                        pick_cnt += 1

                    # Place action
                    elif action_name == 'place':
                        place_traj = modified_action

                        # Get target poses from place_metadata
                        target_ee_pose = place_metadata['target_robot_pose'][place_cnt]
                        target_parts_pose = place_metadata['target_object_pose'][place_cnt]

                        finish = False
                        loop_count = 0

                        # Execute place trajectory
                        while not finish: # place
                            target_pose = torch.tensor(place_traj[loop_count], dtype=torch.float, device='cuda:0')
                            target_pose = torch.unsqueeze(target_pose, dim=0)

                            # Get observation
                            obs = self.env._task.get_place_observations(target_parts_pose, object_name)

                            # Residual action
                            with torch.no_grad():
                                actions = self.place_agent.get_action(obs)

                            # Multiply target 6D pose and residual 6D pose
                            jt_action = self.dt * self.place_action_scale * actions.to('cuda:0')

                            # Add delta pose to reference trajectory
                            dof_pos = self.get_robot_poses(device='cuda:0')
                            target_pose += self.calculate_command(dof_pos, target_pose)
                            target_pose += jt_action

                            # Step simulation
                            self.env._task.pre_physics_step(target_pose)
                            self.env._world.step(render=True)
                            self.env.sim_frame_count += 1
                            self.env._task.post_physics_step(action_name, object_name=object_name, target_parts_pose=target_parts_pose)

                            loop_count += 1
                            if loop_count >= len(place_traj):
                                loop_count = len(place_traj)-1
                                finish = True # bool(self.env._task._check_place_success(target_parts_pose, object_name).squeeze(dim=0))

                        # Check place success
                        place_success = self.env._task._check_place_success(target_parts_pose, object_name).squeeze(dim=0)
                        print('place_success:', place_success)

                        # Add place count
                        place_cnt += 1

                    # Insert action
                    elif action_name == 'insert':
                        insert_traj, depart_traj, return_traj = modified_action

                        # Get target poses from insert_metadata
                        target_ee_pose = insert_metadata['target_robot_pose'][insert_cnt]
                        target_parts_pose = insert_metadata['target_object_pose'][insert_cnt]

                        finish = False
                        loop_count = 0

                        # Execute insert trajectory
                        while not finish: # insert
                            target_pose = torch.tensor(insert_traj[loop_count], dtype=torch.float, device='cuda:0')
                            target_pose = torch.unsqueeze(target_pose, dim=0)

                            # Get observation
                            obs = self.env._task.get_insert_observations(target_parts_pose, object_name)

                            # Residual action
                            with torch.no_grad():
                                actions = self.insert_agent.get_action(obs)

                            # Multiply target 6D pose and residual 6D pose
                            jt_action = self.dt * self.insert_action_scale * actions.to('cuda:0')

                            # Add delta pose to reference trajectory
                            dof_pos = self.get_robot_poses(device='cuda:0')
                            target_pose += self.calculate_command(dof_pos, target_pose)
                            target_pose += jt_action

                            # Step simulation
                            self.env._task.pre_physics_step(target_pose)
                            self.env._world.step(render=True)
                            self.env.sim_frame_count += 1
                            self.env._task.post_physics_step(action_name, object_name=object_name, target_parts_pose=target_parts_pose)

                            loop_count += 1
                            if loop_count >= len(insert_traj):
                                loop_count = len(insert_traj)-1
                            finish = bool(self.env._task._check_insert_success(target_parts_pose, object_name).squeeze(dim=0))

                        # Simulate open gripper steps
                        self.env._task._open_gripper(sim_steps=3)

                        for target_pose in depart_traj: # depart
                            target_pose = torch.tensor(target_pose, dtype=torch.float, device='cuda:0')
                            target_pose = torch.unsqueeze(target_pose, dim=0)

                            # Step simulation
                            self.env._task.pre_physics_step(target_pose)
                            self.env._world.step(render=True)
                            self.env.sim_frame_count += 1
                            self.env._task.post_physics_step(action_name, object_name=object_name, target_parts_pose=target_parts_pose)

                        for target_pose in return_traj: # return
                            target_pose = torch.tensor(target_pose, dtype=torch.float, device='cuda:0')
                            target_pose = torch.unsqueeze(target_pose, dim=0)

                            # Step simulation
                            self.env._task.pre_physics_step(target_pose)
                            self.env._world.step(render=True)
                            self.env.sim_frame_count += 1
                            self.env._task.post_physics_step(action_name, object_name=object_name, target_parts_pose=target_parts_pose)

                        # Check insert success
                        insert_success = self.env._task._check_insert_success(target_parts_pose, object_name).squeeze(dim=0)
                        print('insert_success:', insert_success)

                        # Disable physics properties
                        if insert_success:
                            self.env._task.disable_physics(object_name)

                        # Add insert count
                        insert_cnt += 1

            else:
                self.env._world.step(render=True)

        self.env._simulation_app.close()


@hydra.main(config_name="config", config_path="../cfg")
def main(cfg: DictConfig):
    exec_plan = ExecutePlan()
    exec_plan.execute(cfg)

if __name__ == '__main__':
    main()