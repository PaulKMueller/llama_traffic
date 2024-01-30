import torch
from trajectory_encoder_dataset import TrajectoryEncoderDataset
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
import numpy as np

from ego_trajectory_encoder import EgoTrajectoryEncoder


encoder = EgoTrajectoryEncoder()


optimizer = torch.optim.Adam(encoder.parameters(), lr=1e-4)
criterion = torch.nn.MSELoss(reduction="mean").cuda()

dataset = TrajectoryEncoderDataset()

batch_size = 16
validation_split = .2
shuffle_dataset = True
random_seed= 42

# Creating data indices for training and validation splits:
dataset_size = len(dataset)
indices = list(range(dataset_size))
split = int(np.floor(validation_split * dataset_size))
if shuffle_dataset :
    np.random.seed(random_seed)
    np.random.shuffle(indices)
train_indices, val_indices = indices[split:], indices[:split]

# Creating PT data samplers and loaders:
train_sampler = SubsetRandomSampler(train_indices)
valid_sampler = SubsetRandomSampler(val_indices)

train_dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, 
                                           sampler=train_sampler)
validation_dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size,
                                                sampler=valid_sampler)


# train_dataloader = DataLoader(training_data, batch_size=32, shuffle=True)

import wandb

wandb.init()

# Magic
wandb.watch(encoder, log_freq=100)

device="cuda:0"
encoder.to(device)

for epoch in range(1):  # loop over the dataset multiple times
    running_loss = 0.0
    for i, data in enumerate(train_dataloader):
        # get the inputs; data is a list of [inputs, labels]
        inputs, labels = data

        inputs, labels = inputs.to(device), labels.to(device)
        # print(f"Inputs Shape: {inputs.shape}")

        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs = encoder(inputs)
        # print(f"Output Shape: {outputs.shape}")
        # print(f"Labels Shape: {labels.shape}")
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        if i % 10 == 0:
            wandb.log({"loss": loss})

        # print statistics
        running_loss += loss.item()
        if i % 2000 == 1999:  # print every 2000 mini-batches
            print(f"[{epoch + 1}, {i + 1:5d}] loss: {running_loss / 2000:.3f}")
            running_loss = 0.0

torch.save(encoder.state_dict(), "models/trajectory_encoder.pth")
