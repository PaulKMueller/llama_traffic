import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
from torch import nn
import pytorch_lightning as pl


# class OneHotToFloatNN(pl.LightningModule):
#     def __init__(self):
#         super(OneHotToFloatNN, self).__init__()
#         self.model = nn.Sequential(
#             nn.Linear(12, 64),  # Input layer mapping from 12 inputs to 64 nodes
#             nn.ReLU(),  # Activation function
#             nn.Linear(64, 128),  # Hidden layer 1
#             nn.ReLU(),  # Activation function
#             nn.Linear(128, 256),  # Hidden layer 2
#             nn.ReLU(),  # Activation function
#             nn.Linear(256, 512),  # Hidden layer 3
#             nn.ReLU(),  # Activation function
#             nn.Linear(512, 1024),  # Output layer mapping to 1024 outputs
#         )

#     def forward(self, x):
#         # Forward pass through the network
#         return self.model(x)

#     def configure_optimizers(self):
#         # Configure optimizers (using Adam here)
#         optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
#         return optimizer

#     def training_step(self, batch, batch_idx):
#         # Training step
#         x, y = batch
#         y_hat = self(x)
#         loss = nn.MSELoss()(y_hat, y)
#         return loss


# # Example of how to create a DataLoader (dummy data)
# # from torch.utils.data import DataLoader, TensorDataset
# X_dummy = torch.randn(100, 12)  # Example input data
# y_dummy = torch.randn(100, 1024)  # Example target data
# dataset = TensorDataset(X_dummy, y_dummy)
# dataloader = DataLoader(dataset, batch_size=32)

# # Example of how to train the model
# model = OneHotToFloatNN()
# trainer = pl.Trainer(max_epochs=10)
# trainer.fit(model, dataloader)


# Plain Pytorch version:
import torch
from torch import nn, optim
from torch.utils.data import DataLoader, TensorDataset
import wandb

# Initialize WandB
wandb.init(project="onehot_to_float")


class OneHotToFloatNN(nn.Module):
    def __init__(self):
        super(OneHotToFloatNN, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(12, 64),
            nn.ReLU(),
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Linear(256, 512),
            nn.ReLU(),
            nn.Linear(512, 1024),
        )

    def forward(self, x):
        return self.model(x)


def train_model(model, dataloader, epochs=10):
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.MSELoss()

    for epoch in range(epochs):
        model.train()
        for batch_idx, (data, target) in enumerate(dataloader):
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()

            # Log metrics with WandB
            wandb.log({"epoch": epoch, "loss": loss.item()})

        print(f"Epoch {epoch+1}, Loss: {loss.item()}")


# Dummy dataset
X_dummy = torch.Tensor(np.load("output/scenario_features.npy"))  # Example input data
y_dummy = torch.Tensor(
    np.load("output/scenario_features_embeddings.npy")
)  # Example target data
print(X_dummy.shape)
print(y_dummy.shape)
dataset = TensorDataset(X_dummy, y_dummy)
dataloader = DataLoader(dataset, batch_size=32)

# Model, optimizer, and loss function
model = OneHotToFloatNN()

# Train the model
train_model(model, dataloader)

torch.save(model.state_dict(), "models/scenario_encoder_model.pth")

# Ensure to finish the WandB run after training
wandb.finish()
