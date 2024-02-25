import numpy as np
from npz_utils import list_vehicle_files_absolute
import pytorch_lightning as pl
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, TensorDataset
from scenario_encoder_dataset import ScenarioEncoderDataset
from pytorch_lightning.loggers import WandbLogger


class ScenarioCNN(pl.LightningModule):
    def __init__(self, input_shape=(25, 224, 224), output_dim=1024):
        super(ScenarioCNN, self).__init__()
        self.conv1 = nn.Conv2d(
            in_channels=input_shape[0], out_channels=32, kernel_size=3, padding=1
        )
        self.conv2 = nn.Conv2d(
            in_channels=32, out_channels=64, kernel_size=3, padding=1
        )
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.global_avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(64, output_dim)  # Directly map to output dimension

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.global_avg_pool(x)
        x = x.view(x.size(0), -1)  # Flatten the tensor
        x = self.fc(x)
        return x

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = F.mse_loss(y_hat, y.float())
        self.log("train_loss", loss)
        print(loss)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        return optimizer
