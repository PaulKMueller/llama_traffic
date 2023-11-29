import json
from torch.utils.data import Dataset, DataLoader
import torch


class TrainingData(Dataset):
    def __init__(self, data_path):
        self.data_path = data_path
        self.data = self.load_data(data_path)

    def __str__(self):
        return self.data_path

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        entries = list(self.data.keys())
        return (
            torch.tensor(self.data[entries[index]]["Coordinates"]),
            self.data[entries[index]]["Direction"],
        )

    @staticmethod
    def load_data(filepath):
        """Loads trajectory data from a JSON file.

        Args:
            filepath (str): The path to the JSON file containing the trajectory data.

        Returns:
            dict: The loaded trajectory data.
        """
        with open(filepath, "r") as json_file:
            data_dict = json.load(json_file)
        return data_dict


# import os
# import pandas as pd
# from torchvision.io import read_image

# class CustomImageDataset(Dataset):
#     def __init__(self, annotations_file, img_dir, transform=None, target_transform=None):
#         self.img_labels = pd.read_csv(annotations_file)
#         self.img_dir = img_dir
#         self.transform = transform
#         self.target_transform = target_transform

#     def __len__(self):
#         return len(self.img_labels)

#     def __getitem__(self, idx):
#         img_path = os.path.join(self.img_dir, self.img_labels.iloc[idx, 0])
#         image = read_image(img_path)
#         label = self.img_labels.iloc[idx, 1]
#         if self.transform:
#             image = self.transform(image)
#         if self.target_transform:
#             label = self.target_transform(label)
#         return image, label
