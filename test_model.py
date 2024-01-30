import torch

from ego_trajectory_encoder import EgoTrajectoryEncoder
from npz_trajectory import NpzTrajectory

model = EgoTrajectoryEncoder()
model.load_state_dict(torch.load("/home/pmueller/llama_traffic/models/trajectory_encoder.pth"))
model.eval()

npz_trajectory = NpzTrajectory("/storage_local/fzi_datasets_tmp/waymo_open_motion_dataset/unzipped/train-2e6/vehicle_d_13657_00002_4856147881.npz")

coordinates = list(
                zip(npz_trajectory.coordinates["X"], npz_trajectory.coordinates["Y"])
            )

coordinates = torch.Tensor(coordinates)

with open("output/test.txt", "w") as file:
    print(model(coordinates))