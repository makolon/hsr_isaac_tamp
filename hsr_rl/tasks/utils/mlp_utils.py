import os
import glob
import h5py
import argparse
import numpy as np

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, ConcatDataset
from torch.optim import Adam


class HDF5dataset(Dataset):
    def __init__(self, hdf5_path):
        self.hdf5_path = hdf5_path
        self.dataset = h5py.File(self.hdf5_path, 'r')
        self.input_data = self.dataset['data']
        self.output_data = self.dataset['target']

    def __len__(self):
        assert len(self.input_data) == len(self.output_data), \
           "Need to match the size of dataset"
        return len(self.input_data)

    def __getitem__(self, idx):
        input_data = self.input_data[idx]
        output_data = self.output_data[idx]

        _input = torch.from_numpy(input_data.astype(np.float32)).clone() # (num_samples, input_dim)
        _output = torch.from_numpy(output_data.astype(np.float32)).clone() # (num_samples)

        return _input, _output


class ResidualDeepModel(nn.Module):
    def __init__(self, input_size=23, output_size=8):
        super(ResidualDeepModel, self).__init__()
        self.fc1 = nn.Linear(input_size, 64)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(64, 32)
        self.relu2 = nn.ReLU()
        self.fc3 = nn.Linear(32, 16)
        self.relu3 = nn.ReLU()
        self.fc4 = nn.Linear(16, output_size)

    def forward(self, x):
        x = self.relu1(self.fc1(x))
        x = self.relu2(self.fc2(x))
        x = self.relu3(self.fc3(x))
        x = self.fc4(x)
        return x


def main():
	parser = argparse.ArgumentParser(description="Image benchmark using PyTorch bindings.")

	parser.add_argument("--input_dim", type=int, default=23, help="Number of input dimension")
	parser.add_argument("--output_dim", type=int, default=8, help="Number of output dimension")
	parser.add_argument("--batch_size", type=int, default=256, help="Number of batch size")
	parser.add_argument("--learning_rate", type=int, default=1e-2, help="Number of learning rate")
	parser.add_argument("--num_epochs", type=int, default=200, help="Number of training steps")
	parser.add_argument("--skill_name", type=str, default='move', help="Skill name")
	parser.add_argument("--hdf5_base_path", type=str, default='hdf5_dataset.h5', help="HDF5 dataset path")

	args = parser.parse_args()

	# Concatenate all datasets to one
	skill_dataset = os.path.join(args.hdf5_base_path, f"{args.skill_name}_mlp_dataset_*.h5")
	hdf5_paths = glob.glob(skill_dataset)

	datasets = []
	for hdf5_path in hdf5_paths:
		datasets.append(HDF5dataset(hdf5_path))
	concat_dataset = ConcatDataset(datasets)
	dataloader = DataLoader(concat_dataset, batch_size=args.batch_size, shuffle=True)

	# Define model, criterion, and optimizer
	model = ResidualDeepModel(args.input_dim, args.output_dim)

	criterion = nn.MSELoss()
	optimizer = Adam(model.parameters(), lr=args.learning_rate)

	# Train loop
	for epoch in range(args.num_epochs):
		for inputs, targets in dataloader:
			# Foward
			outputs = model(inputs)

			# Calculate loss
			loss = criterion(outputs, targets)

			# Update parameters
			optimizer.zero_grad()
			loss.backward()
			optimizer.step()

		print(f"Epoch {epoch+1}/{args.num_epochs}, Loss: {loss.item()}")

	# Save models
	model_name = os.path.join('./mlp_models', args.skill_name + '_' +'regression_model.pth')
	torch.save(model.state_dict(), model_name)


if __name__ == "__main__":
	main()