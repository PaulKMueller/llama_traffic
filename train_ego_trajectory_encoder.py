import torch
from trajectory_encoder_dataset import TrajectoryEncoderDataset
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler

from torch import nn
import numpy as np

from ego_trajectory_encoder import EgoTrajectoryEncoder


encoder = EgoTrajectoryEncoder()


optimizer = torch.optim.Adam(encoder.parameters(), lr=1e-4)
# criterion = torch.nn.MSELoss(reduction="mean").cuda()
# criterion = torch.nn.L1Loss().cuda()

criterion = nn.CosineSimilarity().cuda()

dataset = TrajectoryEncoderDataset()

batch_size = 16
validation_split = 0.2
shuffle_dataset = True
random_seed = 42

# Creating data indices for training and validation splits:
dataset_size = len(dataset)
indices = list(range(dataset_size))
split = int(np.floor(validation_split * dataset_size))
if shuffle_dataset:
    np.random.seed(random_seed)
    np.random.shuffle(indices)
train_indices, val_indices = indices[split:], indices[:split]

# Creating PT data samplers and loaders:
train_sampler = SubsetRandomSampler(train_indices)
valid_sampler = SubsetRandomSampler(val_indices)

train_dataloader = torch.utils.data.DataLoader(
    dataset, batch_size=batch_size, sampler=train_sampler
)
validation_dataloader = torch.utils.data.DataLoader(
    dataset, batch_size=batch_size, sampler=valid_sampler
)


# train_dataloader = DataLoader(training_data, batch_size=32, shuffle=True)

import wandb

wandb.init()

# Magic
wandb.watch(encoder, log_freq=100)

device = "cuda"
encoder.to(device)

# for epoch in range(1):  # loop over the dataset multiple times
#     running_loss = 0.0
#     for i, data in enumerate(train_dataloader):
#         # get the inputs; data is a list of [inputs, labels]
#         inputs, labels = data

#         inputs, labels = inputs.to(device), labels.to(device)
#         # print(f"Inputs Shape: {inputs.shape}")

#         # zero the parameter gradients
#         optimizer.zero_grad()

#         # forward + backward + optimize
#         outputs = encoder(inputs)
#         # print(f"Output Shape: {outputs.shape}")
#         # print(f"Labels Shape: {labels.shape}")
#         loss = criterion(outputs, labels)
#         loss.backward()
#         optimizer.step()

#         if i % 10 == 0:
#             wandb.log({"loss": loss})

#         # print statistics
#         running_loss += loss.item()
#         if i % 2000 == 1999:  # print every 2000 mini-batches
#             print(f"[{epoch + 1}, {i + 1:5d}] loss: {running_loss / 2000:.3f}")
#             running_loss = 0.0

for epoch in range(10):  # Assuming you want to train for 10 epochs
    encoder.train()  # Set model to training mode
    running_loss = 0.0
    for i, data in enumerate(train_dataloader):
        inputs, labels = data
        inputs, labels = inputs.to(device), labels.to(device)

        optimizer.zero_grad()

        outputs = encoder(inputs)
        # loss = criterion(outputs, labels)

        # For Cosine Similarity Loss function

        loss = torch.mean(torch.abs(criterion(labels, outputs)))
        print(loss)

        # back-propagation on the above *loss* will try cos(angle) = 0. But I want angle between the vectors to be 0 or cos(angle) = 1.

        loss = 1 - loss

        # End: For Cosine Similarity loss function

        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        if i % 10 == 0:  # Log training loss every 10 mini-batches
            wandb.log({"train_loss": loss.item()})

    # Validation phase
    encoder.eval()  # Set model to evaluation mode
    val_running_loss = 0.0
    val_loss = 0.0
    with torch.no_grad():
        for i, data in enumerate(validation_dataloader):
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)

            outputs = encoder(inputs)
            loss = criterion(outputs, labels)

            val_running_loss += loss.item()

        val_loss = val_running_loss / len(validation_dataloader)
        wandb.log({"val_loss": val_loss})

    print(
        f"Epoch {epoch + 1}, Train Loss: {running_loss / len(train_dataloader):.3f}, Val Loss: {val_loss:.3f}"
    )


torch.save(encoder.state_dict(), "models/trajectory_encoder_wv_cos.pth")
