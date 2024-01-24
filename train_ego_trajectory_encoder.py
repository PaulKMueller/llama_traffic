import torch
from trajectory_encoder_dataset import TrajectoryEncoderDataset
from torch.utils.data import DataLoader

from ego_trajectory_encoder import EgoTrajectoryEncoder


encoder = EgoTrajectoryEncoder()


optimizer = torch.optim.Adam(encoder.parameters(), lr=1e-4)
criterion = torch.nn.MSELoss(reduction="mean")

training_data = TrajectoryEncoderDataset()

train_dataloader = DataLoader(training_data, batch_size=32, shuffle=True)


import wandb

wandb.init()

# Magic
wandb.watch(encoder, log_freq=100)

for epoch in range(2):  # loop over the dataset multiple times
    running_loss = 0.0
    for i, data in enumerate(train_dataloader):
        # get the inputs; data is a list of [inputs, labels]
        inputs, labels = data
        print(f"Inputs Shape: {inputs.shape}")

        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs = encoder(inputs)
        print(f"Output Shape: {outputs.shape}")
        print(f"Labels Shape: {labels.shape}")
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
