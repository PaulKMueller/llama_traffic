from npz_trajectory import NpzTrajectory
import numpy as np
from traffic_lane import TrafficLane
from npz_utils import get_random_npz_trajectory


def has_parking_lot(npz_trajectory: NpzTrajectory):

    V = npz_trajectory.vector_data
    X, idx = V[:, :44], V[:, 44].flatten()

    lanes = []
    statics = []

    for i in np.unique(idx):
        _X = X[idx == i]
        if _X[:, 13:16].sum() > 0:
            # print(i)
            lane = TrafficLane(_X[:, 0], _X[:, 1])
            lanes.append(lane)
            # print(_X[:, 0])
            # print(_X[:, 1])
        if _X[:, 8].sum() > 0:
            if not _X[-1, 2]:
                static_coordinate = np.array([_X[:, 0][0], _X[:, 1][0]])
                # print(_X[:, 0][0])
                # print(_X[:, 1][0])
                statics.append(static_coordinate)

    # print(len(statics))
    # rel_lane_dists = []
    counter = 0
    width = npz_trajectory.get_scenario_width()

    for static in statics:
        lane_dist = min([lane.min_distance_to_point(static) / width for lane in lanes])
        # print(lane_dist)
        if lane_dist > 0.01:
            counter += 1
        # rel_lane_dists.append(lane_dist)

    # print(rel_lane_dists)
    # print(f"Counter: {counter}")

    # print(counter)
    return counter > 1


def has_parking_lot_refined(npz_trajectory: NpzTrajectory):
    V = npz_trajectory.vector_data
    X, idx = V[:, :44], V[:, 44].flatten()

    lanes = []
    statics = []

    for i in np.unique(idx):
        _X = X[idx == i]
        if np.any(_X[:, 13:16].sum(axis=1) > 0):
            lane = TrafficLane(_X[:, 0], _X[:, 1])
            lanes.append(lane)
        if np.any(_X[:, 8] > 0) and not _X[-1, 2]:
            static_coordinate = np.array([_X[:, 0][0], _X[:, 1][0]])
            statics.append(static_coordinate)

    width = npz_trajectory.get_scenario_width()
    counter = 0

    for static in statics:
        lane_dist = min([lane.min_distance_to_point(static) / width for lane in lanes])
        if lane_dist > 0.01:
            counter += 1

    return counter > 1


# i = 0

# correct = 0
# incorrect = 0

# while i < 100:
#     rnd_traj = NpzTrajectory(get_random_npz_trajectory())
#     rnd_traj.plot_scenario()
#     user_input = input("Does this have a parking lot?")
#     if str(has_parking_lot(rnd_traj)) == user_input:
#         correct += 1
#     else:
#         incorrect += 1
#     print(has_parking_lot(rnd_traj))
#     print(user_input)

#     print(str(has_parking_lot(rnd_traj)) == user_input)

#     print(incorrect)
#     print(correct)
#     help = input("Continue?")
