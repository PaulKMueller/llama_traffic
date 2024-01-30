import torch

from ego_trajectory_encoder import EgoTrajectoryEncoder
from npz_trajectory import NpzTrajectory
from trajectory_encoder_dataset import TrajectoryEncoderDataset
from torch.utils.data import DataLoader

right_u_turn = [
    "Rightward complete reversal",
    "180-degree turn to the right",
    "Clockwise U-turn",
    "Right circular turnaround",
    "Right-hand loopback",
    "Right flip turn",
    "Full right pivot",
    "Right about-face",
    "Rightward return turn",
    "Rightward reversing curve",
]
left_u_turn = [
    "Leftward complete reversal",
    "180-degree turn to the left",
    "Counterclockwise U-turn",
    "Left circular turnaround",
    "Left-hand loopback",
    "Left flip turn",
    "Full left pivot",
    "Left about-face",
    "Leftward return turn",
    "Leftward reversing curve",
]
stationary = [
    "At a standstill",
    "Motionless",
    "Unmoving",
    "Static position",
    "Immobilized",
    "Not in motion",
    "Fixed in place",
    "Idle",
    "Inert",
    "Anchored",
]
right = [
    "Rightward",
    "To the right",
    "Right-hand side",
    "Starboard",
    "Rightward direction",
    "Clockwise direction",
    "Right-leaning",
    "Rightward bound",
    "Bearing right",
    "Veering right",
]
left = [
    "Leftward",
    "To the left",
    "Left-hand side",
    "Port",
    "Leftward direction",
    "Counterclockwise direction",
    "Left-leaning",
    "Leftward bound",
    "Bearing left",
    "Veering left",
]
straight_right = [
    "Straight then right",
    "Forward followed by a right turn",
    "Proceed straight, then veer right",
    "Continue straight before turning right",
    "Advance straight, then bear right",
    "Go straight, then curve right",
    "Head straight, then pivot right",
    "Move straight, then angle right",
    "Straight-line, followed by a right deviation",
    "Directly ahead, then a rightward shift",
]
straight_left = [
    "Straight then left",
    "Forward followed by a left turn",
    "Proceed straight, then veer left",
    "Continue straight before turning left",
    "Advance straight, then bear left",
    "Go straight, then curve left",
    "Head straight, then pivot left",
    "Move straight, then angle left",
    "Straight-line, followed by a left deviation",
    "Directly ahead, then a leftward shift",
]
straight = [
    "Directly ahead",
    "Forward",
    "Straightforward",
    "In a straight line",
    "Linearly",
    "Unswervingly",
    "Onward",
    "Direct path",
    "True course",
    "Non-curving path",
]

bucket_synonym_lists = [
    right_u_turn,
    left_u_turn,
    stationary,
    right,
    left,
    straight_right,
    straight_left,
    straight,
]


model = EgoTrajectoryEncoder()
model.load_state_dict(torch.load("/home/pmueller/llama_traffic/models/trajectory_encoder.pth"))
model.eval()

# dataset = TrajectoryEncoderDataset()

# data_loader = DataLoader(dataset, batch_size=1)

npz_trajectory = NpzTrajectory("/storage_local/fzi_datasets_tmp/waymo_open_motion_dataset/unzipped/train-2e6/vehicle_d_13657_00002_4856147881.npz")

coordinates = list(
                zip(npz_trajectory.coordinates["X"], npz_trajectory.coordinates["Y"])
            )

coordinates = torch.Tensor(coordinates)

coordinates.to("cuda")

# input = next(iter(data_loader)).to("cuda")

print("Test")
coordinates = coordinates.unsqueeze(0)
with open("output/test.txt", "w") as file:
    model.eval()
    with torch.no_grad():
        output = model(coordinates)
        torch.set_printoptions(profile="full")
        file.write(str(output))
        print(output.shape)