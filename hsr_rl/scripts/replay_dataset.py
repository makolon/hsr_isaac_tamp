import os
import torch
import hydra
import numpy as np
from omegaconf import DictConfig

from hsr_rl.utils.hydra_cfg.hydra_utils import *
from hsr_rl.utils.hydra_cfg.reformat import omegaconf_to_dict, print_dict

from hsr_rl.utils.task_utils import initialize_task
from hsr_rl.utils.dataset_utils import load_dataset, load_skill_dataset
from hsr_rl.envs.isaac_env_rlgames import IsaacEnvRlgames

@hydra.main(config_name="config", config_path="../cfg")
def parse_hydra_configs(cfg: DictConfig):
    headless = cfg.headless
    render = not headless
    rank = int(os.getenv("LOCAL_RANK", "0"))
    if cfg.multi_gpu:
        cfg.device_id = rank
        cfg.rl_device = f'cuda:{rank}'
    enable_viewport = "enable_cameras" in cfg.task.sim and cfg.task.sim.enable_cameras
    env = IsaacEnvRlgames(headless=headless, sim_device=cfg.device_id, enable_livestream=cfg.enable_livestream, enable_viewport=enable_viewport)

    cfg_dict = omegaconf_to_dict(cfg)
    print_dict(cfg_dict)

    action_replay = load_skill_dataset(cfg_dict['num_envs'], 'pick')

    # sets seed. if seed is -1 will pick a random one
    from omni.isaac.core.utils.torch.maths import set_seed
    cfg.seed = set_seed(cfg.seed, torch_deterministic=cfg.torch_deterministic)
    cfg_dict['seed'] = cfg.seed
    task = initialize_task(cfg_dict, env)

    while env._simulation_app.is_running():
        if env._world.is_playing():
            if env._world.current_time_step_index == 0:
                env._world.reset(soft=True)
            if env.sim_frame_count >= len(action_replay):
                actions = torch.tensor([np.random.uniform(env.action_space.low, env.action_space.high) for _ in range(env.num_envs)], dtype=torch.float, device=task.rl_device)
            else:
                actions = torch.tensor([action_replay[env.sim_frame_count] for _ in range(env.num_envs)], device=task.rl_device)
            env._task.pre_physics_step(actions)
            env._world.step(render=render)
            env.sim_frame_count += 1
            env._task.post_physics_step()
        else:
            env._world.step(render=render)

    env._simulation_app.close()

if __name__ == '__main__':
    parse_hydra_configs()