from npz_trajectory import NpzTrajectory
from traffic_lane import TrafficLane
import numpy as np
from npz_utils import get_random_npz_trajectory


test_path = "/storage_local/fzi_datasets_tmp/waymo_open_motion_dataset/unzipped/train-2e6/vehicle_d_13657_00002_4856147881.npz"
trajectory = NpzTrajectory(test_path)

trajectory.plot_scenario()


def has_intersection(trajectory: NpzTrajectory):
    V = trajectory.vector_data
    X, idx = V[:, :44], V[:, 44].flatten()

    lanes = []

    for i in np.unique(idx):
        _X = X[idx == i]
        if _X[:, 13:16].sum() > 0:
            # print(i)
            lane = TrafficLane(_X[:, 0], _X[:, 1])
            lanes.append(lane)
            # print(lane.get_cumulative_delta_angle())
            # print(_X[:, 0])
            # print(_X[:, 1])

    crossings = 0
    filtered_lanes = []

    for lane in lanes:
        cum_delta_angle = abs(lane.get_cumulative_delta_angle())
        if cum_delta_angle > 80 and cum_delta_angle < 100:
            filtered_lanes.append(lane)

    for i in range(len(filtered_lanes)):
        # print(lane.get_cumulative_delta_angle())
        for j in range(len(filtered_lanes)):
            if i == j:
                continue
            else:
                min_dist = filtered_lanes[i].get_min_dist_to_other_lane(
                    filtered_lanes[j]
                )
                # print(min_dist)
                if min_dist == 0:
                    return True
                    crossings += 1

    return crossings > 0


i = 0

correct = 0
incorrect = 0

while i < 100:
    rnd_traj = NpzTrajectory(get_random_npz_trajectory())
    rnd_traj.plot_scenario()
    user_input = input("Does this have a parking lot?")
    if str(has_intersection(rnd_traj)) == user_input:
        correct += 1
    else:
        incorrect += 1
    print(has_intersection(rnd_traj))
    print(user_input)

    print(str(has_intersection(rnd_traj)) == user_input)

    print(incorrect)
    print(correct)
    help = input("Continue?")
