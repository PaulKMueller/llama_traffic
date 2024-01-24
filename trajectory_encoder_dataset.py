from torch.utils.data import Dataset
import torch
from uae_explore import encode_with_uae

import json


class TrajectoryEncoderDataset(Dataset):
    def __init__(self):
        with open("datasets/processed_vehicle_a.json") as file:
            self.data_json = json.load(file)
        with open("datasets/uae_buckets_cache.json") as cache:
            self.direction_labels = json.load(cache)

    def __len__(self):
        return len(self.data_json)

    def __getitem__(self, idx):
        item = list(self.data_json.values())[idx]
        coordinates = torch.Tensor(item["Coordinates"])
        direction = item["Direction"]

        # Getting cached bucket embeddings
        encoded_input_text = torch.Tensor(self.direction_labels[direction])

        return coordinates, encoded_input_text
