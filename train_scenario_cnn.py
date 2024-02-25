import numpy as np
from npz_utils import list_vehicle_files_absolute
import pytorch_lightning as pl
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, TensorDataset
from scenario_encoder_dataset import ScenarioEncoderDataset
from pytorch_lightning.loggers import WandbLogger
from scenario_cnn import SimpleCNN

# Dummy dataset with 1 sample for demonstration purposes
# You will need to replace this with your actual dataset
torch.cuda.empty_cache()
x_dummy = torch.randn(1, 25, 224, 224)
y_dummy = torch.randn(1, 1024)  # Output dummy vector with 1024 dimensions

dataset = ScenarioEncoderDataset()
dataloader = DataLoader(dataset, batch_size=1)

# Instantiate the model
model = SimpleCNN()


def print_allocated_memory():
    print("{:.2f} GB".format(torch.cuda.memory_allocated() / 1024**3))


# Train the model
wandb_logger = WandbLogger(log_model="all")
trainer = pl.Trainer(max_epochs=10, logger=wandb_logger)
wandb_logger.watch(model)

trainer.fit(model, dataloader)

path = list_vehicle_files_absolute()[0]
scenario = np.load(path)
vectors = scenario["raster"]

vectors = vectors.reshape(1, 25, 224, 224)


print("Model has been trained")
print(model(torch.Tensor(vectors)))
print(model(torch.Tensor(vectors)).shape)
