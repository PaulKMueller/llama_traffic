from npz_utils import get_random_npz_trajectory
from npz_trajectory import NpzTrajectory
from traffic_lane import TrafficLane
import numpy as np

npz_trajectory = NpzTrajectory(get_random_npz_trajectory())

V = npz_trajectory.vector_data
X, idx = V[:, :44], V[:, 44].flatten()

lanes_fast = [
    TrafficLane(X[idx == i][:, 0], X[idx == i][:, 1])
    for i in np.unique(idx)
    if X[idx == i][:, 13:16].sum() > 0
]

print(len(lanes_fast))

lanes_slow = []

for i in np.unique(idx):
    _X = X[(idx == i)]
    if _X[:, 13:16].sum() > 0:  # Traffic lanes
        lanes_slow.append(TrafficLane(_X[:, 0], _X[:, 1]))

print(len(lanes_slow))
