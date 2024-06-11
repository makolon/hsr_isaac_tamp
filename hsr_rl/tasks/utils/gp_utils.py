import h5py
import torch
import argparse
import gpytorch
import numpy as np
import matplotlib.pyplot as plt
from gpytorch.means import ConstantMean
from gpytorch.kernels import RBFKernel, ScaleKernel
from gpytorch.distributions import MultivariateNormal
from torch.utils.data import Dataset, DataLoader, ConcatDataset


class HDF5dataset(Dataset):
    def __init__(self, hdf5_path, num_samples):
        self.hdf5_path = hdf5_path
        self.dataset = h5py.File(self.hdf5_path, 'r')
        self.input_data = self.dataset['train_x']
        self.output_data = self.dataset['train_y']

        self.num_samples = num_samples

    def __len__(self):
        assert len(self.input_data) == len(self.output_data), \
           "Need to match the size of dataset"
        return len(self.input_data)

    def __getitem__(self, idx):
        input_data = self.input_data[idx]
        output_data = self.output_data[idx]

        _input = torch.from_numpy(np.array([input_data])).clone() # (num_samples, input_dim)
        _output = torch.from_numpy(np.array([output_data])).clone() # (num_samples)

        return _input, _output

class ExactGPModel(gpytorch.models.ExactGP):
    def __init__(self, batch_size, train_x, train_y, likelihood):
        super(ExactGPModel, self).__init__(train_x, train_y, likelihood)
        self.mean_module = gpytorch.means.ConstantMean(batch_shape=torch.Size([batch_size]))
        self.covar_module = gpytorch.kernels.ScaleKernel(
            gpytorch.kernels.MaternKernel(batch_shape=torch.Size([batch_size])),
            batch_shape=torch.Size([batch_size])
        )

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return MultivariateNormal(mean_x, covar_x)

class ResidualGPModel(object):
    def set_up_model(self,
                     hdf5_paths,
                     num_envs=256,
                     num_samples=1,
                     input_dim=15,
                     training_iter=200):
        self.hdf5_paths = hdf5_paths
        self.num_envs = num_envs
        self.num_samples = num_samples
        self.input_dim = input_dim
        self.training_iter = training_iter

        # Concatenate all datasets to one
        datasets = []
        for hdf5_path in self.hdf5_paths:
            datasets.append(HDF5dataset(hdf5_path, num_samples))
        concat_dataset = ConcatDataset(datasets)
        dataloader = DataLoader(concat_dataset, batch_size=num_envs, shuffle=True)

        # Get full dataset (num_envs, num_samples, input_dim) / (num_env, num_samples)
        train_x, train_y = [], []
        for i in range(num_samples):
            data, target = next(iter(dataloader))
            train_x.append(data)
            train_y.append(target)

        self.train_x = torch.stack(train_x).view(num_envs, num_samples, input_dim)
        self.train_y = torch.stack(train_y).view(num_envs, num_samples)

        # Define likelihood / GP model
        self.likelihood = gpytorch.likelihoods.GaussianLikelihood(batch_shape=torch.Size([num_envs]))
        self.model = ExactGPModel(num_envs, self.train_x, self.train_y, self.likelihood)

    def train(self):
        # Use the adam optimizer
        optimizer = torch.optim.Adam(self.model.parameters(), lr=0.01)

        # "Loss" for GPs - the marginal log likelihood
        mll = gpytorch.mlls.ExactMarginalLogLikelihood(self.likelihood, self.model)

        # Training loop
        for i in range(self.training_iter):
            # Zero gradients from previous iteration
            optimizer.zero_grad()
            # Output from model
            output = self.model(self.train_x)
            # Calc loss and backprop gradients
            loss = -mll(output, self.train_y).sum()
            loss.backward()
            print('Iter %d/%d - Loss: %.3f' % (i + 1, self.training_iter, loss.item()))
            optimizer.step()

    def predict(self, data):
        # Evaluate mode
        self.model.eval()
        self.likelihood.eval()

        with torch.no_grad():
            # Make predictions
            pred = self.likelihood(self.model(data))
            # Get mean
            mean = pred.mean
            # Get lower and upper confidence bounds
            lower, upper = pred.confidence_region()
            return mean, lower, upper

    def save_model(self, skill_name):
        file_name = self.hdf5_paths[-1].split('/')[-1][-15:-5]
        model_name = './gp_models/model' + '_' + skill_name + '_' + file_name + '.pth'
        torch.save(self.model.state_dict(), model_name)

    def load_model(self, skill_name):
        file_name = self.hdf5_paths[-1].split('/')[-1][-15:-5]
        model_name = '/root/tamp-hsr/hsr_rl/tasks/utils/gp_models/model' + '_' + skill_name + '_' + file_name + '.pth'
        self.model.load_state_dict(torch.load(model_name))
        self.model.to()


def main():
    parser = argparse.ArgumentParser(description="Image benchmark using PyTorch bindings.")

    parser.add_argument("--input_dim", type=int, default=15, help="Number of input dimension")
    parser.add_argument("--output_dim", type=int, default=6, help="Number of output dimension")
    parser.add_argument("--training_iter", type=int, default=200, help="Number of training iterations")
    parser.add_argument("--num_envs", type=int, default=256, help="Number of batch size")
    parser.add_argument("--num_trials", type=int, default=2, help="Number of training steps")
    parser.add_argument("--num_samples", type=int, default=300, help="Number of samples")
    parser.add_argument("--skill_name", type=str, default='move', help="Skill name")
    parser.add_argument("--hdf5_base_path", type=str, default='hdf5_dataset.h5', help="HDF5 dataset path")
    parser.add_argument("--train", action="store_true", help="GP train")
    parser.add_argument("--test", action="store_true", help="GP test")

    args = parser.parse_args()

    hdf5_all_paths = []
    skill_names = ['move', 'pick', 'place', 'insert']
    for skill_name in skill_names:
        hdf5_position_x, hdf5_position_y, hdf5_position_z = [], [], []
        hdf5_rotation_x, hdf5_rotation_y, hdf5_rotation_z = [], [], []
        for num_trial in range(1, args.num_trials):
            hdf5_position_x.append(f'/root/tamp-hsr/hsr_ros/hsr_ws/src/gearbox_3d/script/rl_controller/residual_dataset/{skill_name}_dataset_position_x_{num_trial}.h5')
            hdf5_position_y.append(f'/root/tamp-hsr/hsr_ros/hsr_ws/src/gearbox_3d/script/rl_controller/residual_dataset/{skill_name}_dataset_position_y_{num_trial}.h5')
            hdf5_position_z.append(f'/root/tamp-hsr/hsr_ros/hsr_ws/src/gearbox_3d/script/rl_controller/residual_dataset/{skill_name}_dataset_position_z_{num_trial}.h5')
            hdf5_rotation_x.append(f'/root/tamp-hsr/hsr_ros/hsr_ws/src/gearbox_3d/script/rl_controller/residual_dataset/{skill_name}_dataset_rotation_x_{num_trial}.h5')
            hdf5_rotation_y.append(f'/root/tamp-hsr/hsr_ros/hsr_ws/src/gearbox_3d/script/rl_controller/residual_dataset/{skill_name}_dataset_rotation_y_{num_trial}.h5')
            hdf5_rotation_z.append(f'/root/tamp-hsr/hsr_ros/hsr_ws/src/gearbox_3d/script/rl_controller/residual_dataset/{skill_name}_dataset_rotation_z_{num_trial}.h5')
        hdf5_all_paths.append([hdf5_position_x, hdf5_position_y, hdf5_position_z,
                               hdf5_rotation_x, hdf5_rotation_y, hdf5_rotation_z])

    if args.train:
        # Train step
        for idx, skill_name in enumerate(skill_names): # Select 'move'/'pick'/'place'/'insert'
            for hdf5_paths in hdf5_all_paths[idx]: # hdf5_paths includes all trials, e.g.) hdf5_path = move_dataset_position_x_{1:9}.h5
                # Deinfe GP regression model
                gp_regression = ResidualGPModel()

                # Set up model
                gp_regression.set_up_model(
                    hdf5_paths=hdf5_paths,
                    num_envs=args.num_envs,
                    num_samples=args.num_samples,
                    input_dim=args.input_dim,
                    training_iter=args.training_iter
                )

                # Train GP model
                gp_regression.train()

                # Save GP model
                gp_regression.save_model(skill_name)

    if args.test:
        # Test step
        for idx, skill_name in enumerate(skill_names):
            for hdf5_paths in hdf5_all_paths[idx]:
                # Deinfe GP regression model
                gp_regression = ResidualGPModel()

                # Set up model
                gp_regression.set_up_model(
                    hdf5_paths=hdf5_paths,
                    num_envs=args.num_envs,
                    num_samples=args.num_samples,
                    input_dim=args.input_dim,
                    training_iter=args.training_iter
                )

                # Load model
                gp_regression.load_model(skill_name)

                # Tentative input/output data
                input_data = gp_regression.train_x
                output_data = gp_regression.train_y

                # Test model
                pred_mean, lower, upper = gp_regression.predict(input_data)
                loss = (pred_mean - output_data).sum()
                print('loss:', loss)


if __name__ == '__main__':
    main()