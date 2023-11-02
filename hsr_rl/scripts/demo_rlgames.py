from hsr_rl.utils.hydra_cfg.hydra_utils import *
from hsr_rl.utils.hydra_cfg.reformat import omegaconf_to_dict, print_dict
from hsr_rl.utils.task_utils import initialize_task
from hsr_rl.envs.isaac_env_rlgames import IsaacEnvRlgames
from hsr_rl.scripts.train_rlgames import RLGTrainer

import hydra
from omegaconf import DictConfig

import datetime
import os
import torch

class RLGDemo(RLGTrainer):
    def __init__(self, cfg, cfg_dict):
        RLGTrainer.__init__(self, cfg, cfg_dict)
        self.cfg.test = True

@hydra.main(config_name="config", config_path="../cfg")
def parse_hydra_configs(cfg: DictConfig):

    time_str = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

    headless = cfg.headless
    render = not headless
    sim_app_cfg_path = cfg.sim_app_cfg_path

    env = IsaacEnvRlgames(headless=headless, render=render, sim_app_cfg_path=sim_app_cfg_path)

    # ensure checkpoints can be specified as relative paths
    from hsr_rl.utils.config_utils.path_utils import retrieve_checkpoint_path
    if cfg.checkpoint:
        cfg.checkpoint = retrieve_checkpoint_path(cfg.checkpoint)
        if cfg.checkpoint is None:
            quit()

    cfg_dict = omegaconf_to_dict(cfg)
    print_dict(cfg_dict)

    # sets seed. if seed is -1 will pick a random one
    from omni.isaac.core.utils.torch.maths import set_seed
    cfg.seed = set_seed(cfg.seed, torch_deterministic=cfg.torch_deterministic)
    cfg_dict['seed'] = cfg.seed
    task = initialize_task(cfg_dict, env)

    if cfg.wandb_activate:
        # Make sure to install WandB if you actually use this.
        import wandb

        run_name = f"{cfg.wandb_name}_{time_str}"

        wandb.init(
            project=cfg.wandb_project,
            group=cfg.wandb_group,
            entity=cfg.wandb_entity,
            config=cfg_dict,
            sync_tensorboard=True,
            id=run_name,
            resume="allow",
            monitor_gym=True,
        )

    rlg_trainer = RLGDemo(cfg, cfg_dict)
    rlg_trainer.launch_rlg_hydra(env)
    rlg_trainer.run()
    env.close()

    if cfg.wandb_activate:
        wandb.finish()


if __name__ == '__main__':
    parse_hydra_configs()