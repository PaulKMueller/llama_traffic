from npz_trajectory import NpzTrajectory
import numpy as np
from traffic_lane import TrafficLane


def has_parking_lot(npz_trajectory: NpzTrajectory):
    npz_trajectory = NpzTrajectory(
        "/storage_local/fzi_datasets_tmp/waymo_open_motion_dataset/unzipped/train-2e6/vehicle_b_91155_00004_305435325.npz"
    )

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
    rel_lane_dists = []
    counter = 0

    for static in statics:
        lane_dist = min(
            [
                lane.min_distance_to_point(static) / npz_trajectory.get_scenario_width()
                for lane in lanes
            ]
        )
        print(lane_dist)
        if lane_dist > 0.01:
            counter += 1
        rel_lane_dists.append(lane_dist)

    print(counter)
    return counter > 0
