import os
import d3rlpy
import argparse
import numpy as np


def load_dataset(skill_name: str, dataset_path: str):
    dataset_name = skill_name.lower() + '_dataset.h5' # NOTE: lower skill_name
    dataset_file = os.path.join(dataset_path, dataset_name)
    with open(dataset_file, "rb") as f:
        dataset = d3rlpy.dataset.ReplayBuffer.load(f, d3rlpy.dataset.InfiniteBuffer())

    return dataset

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--skill_name", type=str, default="Pick")
    parser.add_argument("--dataset", type=str, default="gearbo3d")
    parser.add_argument("--n_train", type=int, default=0)
    parser.add_argument("--n_steps", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--gpu", type=int, default=1)
    args = parser.parse_args()

    # Load dataset (TODO: modify)
    dataset_path = "/root/tamp-hsr/hsr_ros/hsr_ws/src/gearbox_3d/scripts/rl_controller/dataset"
    dataset = load_dataset(args.skill_name, dataset_path)

    # Set up algorithm
    sac = d3rlpy.algos.SACConfig().create()
    sac.build_with_dataset(dataset)

    # Load model
    task_name = "HSRGearboxResidual" + args.skill_name
    model_path = os.path.join('runs', task_name, 'nn', task_name+'.pth')
    sac.load_model(model_path)

    # Start offline training
    sac.fit(dataset, n_steps=args.n_steps)

    # Save model
    model_path = os.path.join('runs', task_name, 'nn', task_name+'_finetune.pth')
    sac.save(model_path)


if __name__ == "__main__":
    main()