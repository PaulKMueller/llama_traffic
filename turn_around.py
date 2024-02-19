from npz_trajectory import NpzTrajectory
import numpy as np
from traffic_lane import TrafficLane
from npz_utils import get_random_npz_trajectory


# npz_trajectory_path = (
#     "/storage_local/fzi_datasets_tmp/waymo_open_motion_dataset/unzipped/train-2e6/"
#     + "vehicle_a_59799_00001_2926718064.npz"
# )

# /storage_local/fzi_datasets_tmp/waymo_open_motion_dataset/unzipped/train-2e6/vehicle_a_58166_00000_2084792320.npz
# /storage_local/fzi_datasets_tmp/waymo_open_motion_dataset/unzipped/train-2e6/vehicle_a_37878_00001_5103826733.npz
# /storage_local/fzi_datasets_tmp/waymo_open_motion_dataset/unzipped/train-2e6/vehicle_a_23385_00002_7041065400.npz
# /storage_local/fzi_datasets_tmp/waymo_open_motion_dataset/unzipped/train-2e6/vehicle_a_44796_00000_9124537468.npz
# /storage_local/fzi_datasets_tmp/waymo_open_motion_dataset/unzipped/train-2e6/vehicle_a_60258_00001_8463072180.npz
# /storage_local/fzi_datasets_tmp/waymo_open_motion_dataset/unzipped/train-2e6/vehicle_a_27952_00005_4983497228.npz


def has_turnaround(npz_trajectory: NpzTrajectory):

    # npz_trajectory_path = get_random_npz_trajectory()

    # npz_trajectory = NpzTrajectory(npz_trajectory_path)
    # npz_trajectory.plot_scenario()

    V = npz_trajectory.vector_data
    X, idx = V[:, :44], V[:, 44].flatten()

    lanes = []

    for i in np.unique(idx):
        _X = X[idx == i]
        if _X[:, 13:16].sum() > 0:
            # print(i)
            lane = TrafficLane(_X[:, 0], _X[:, 1])
            lanes.append(lane)
            # print(_X[:, 0])
            # print(_X[:, 1])

    # print(len(statics))
    # rel_lane_dists = []
    width = npz_trajectory.get_scenario_width()

    for lane in lanes:
        angles = lane.get_delta_angles()
        pos_angles = filter(lambda x: x > 0, angles)
        neg_angles = filter(lambda x: x < 0, angles)

        pos_angle_sum = abs(sum(pos_angles))
        neg_angle_sum = abs(sum(neg_angles))

        if pos_angle_sum > 190 or neg_angle_sum > 190:
            # print(pos_angle_sum)
            # print(neg_angle_sum)
            # cum_angle = lane.get_cumulative_delta_angle()
            # print(cum_angle)

            if (
                np.linalg.norm(lane.coordinates[0] - lane.coordinates[-1]) / width
                < 0.05
            ):
                # print("There is a turnaround!")
                return True

    return False


i = 0

correct = 0
incorrect = 0

while i < 100:
    rnd_traj = NpzTrajectory(get_random_npz_trajectory())
    rnd_traj.plot_scenario()
    user_input = input("Does this have a turnaround?")
    if str(has_turnaround(rnd_traj)) == user_input:
        correct += 1
    else:
        incorrect += 1
    print(has_turnaround(rnd_traj))
    print(user_input)

    print(str(has_turnaround(rnd_traj)) == user_input)

    print(incorrect)
    print(correct)
    help = input("Continue?")


# print(lane_dist)

# rel_lane_dists.append(lane_dist)

# print(rel_lane_dists)

# print(f"Counter: {counter}")

# print(counter)
# print(counter > 1)
