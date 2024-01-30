from torch.utils.data import Dataset
import torch
from uae_explore import encode_with_uae

import json


class TrajectoryEncoderDataset(Dataset):
    def __init__(self, device="cuda:0"):
        with open("datasets/processed_vehicle_a.json") as file:
            self.data_json = json.load(file)
            self.items = list(self.data_json.values())
            self.coordinates = torch.Tensor([item["Coordinates"] for item in self.items])

        with open("datasets/uae_buckets_cache.json") as cache:
            self.direction_labels = json.load(cache)
            directions = [item["Direction"] for item in self.items]
            self.encoded_input_texts = torch.Tensor([self.direction_labels[direction] for direction in directions])

        self.coordinates.to(device)
        self.encoded_input_texts.to(device)


    def __len__(self):
        return len(self.data_json)

    def __getitem__(self, idx):
        # item = list(self.data_json.values())[idx]
        # coordinates = torch.Tensor(item["Coordinates"])
        # direction = item["Direction"]

        # Getting cached bucket embeddings
        # encoded_input_text = torch.Tensor(self.direction_labels[direction])

        return self.coordinates[idx], self.encoded_input_texts[idx]
