from torch.utils.data import Dataset
import torch
from uae_explore import encode_with_uae
from npz_utils import list_vehicle_files_absolute
import numpy as np

import json


class ScenarioEncoderDataset(Dataset):
    def __init__(self, device="cuda:0"):
        all_scenario_paths = list_vehicle_files_absolute()[:5]
        self.x_data = torch.Tensor(
            np.array(
                [
                    np.load(path)["raster"].reshape(25, 224, 224)
                    for path in all_scenario_paths
                ]
            )
        )
        print(self.x_data.shape)
        self.y_data = torch.rand(5, 1024)

        print(self.y_data.shape)

        self.x_data.to(device)
        self.y_data.to(device)

    def __len__(self):
        return self.x_data.shape[0]

    def __getitem__(self, idx):
        return self.x_data[idx], self.y_data[idx]
