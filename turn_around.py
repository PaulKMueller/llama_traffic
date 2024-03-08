from npz_trajectory import NpzTrajectory
import numpy as np
from traffic_lane import TrafficLane
from npz_utils import get_random_npz_trajectory
from numba import jit

from joblib import Parallel, delayed


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

    lanes = [
        TrafficLane(X[idx == i][:, 0], X[idx == i][:, 1])
        for i in np.unique(idx)
        if X[idx == i][:, 13:16].sum() > 0
    ]

    width = npz_trajectory.get_scenario_width()

    for lane in lanes:
        angles = lane.get_delta_angles()

        pos_angle_sum = 0.0
        neg_angle_sum = 0.0
        for angle in angles:
            if angle > 0:
                pos_angle_sum += angle
            elif angle < 0:
                neg_angle_sum += angle

        if pos_angle_sum > 190 or neg_angle_sum > 190:
            print(pos_angle_sum)
            print(neg_angle_sum)
            # cum_angle = lane.get_cumulative_delta_angle()
            # print(cum_angle)

            if (
                np.linalg.norm(lane.coordinates[0] - lane.coordinates[-1]) / width
                < 0.05
            ):
                # print("There is a turnaround!")
                return True

    return False


@jit(nopython=True)
def calculate_turnaround(angles, threshold, distance):
    pos_angle_sum = 0.0
    neg_angle_sum = 0.0
    for angle in angles:
        if angle > 0:
            pos_angle_sum += angle
        elif angle < 0:
            neg_angle_sum += angle

    return (
        abs(pos_angle_sum) > threshold or abs(neg_angle_sum) > threshold
    ) and distance


def process_lane(lane, width):
    angles = lane.get_delta_angles()
    distance = np.linalg.norm(lane.coordinates[0] - lane.coordinates[-1]) / width < 0.05
    return calculate_turnaround(angles, 190, distance)


def has_turnaround_fast(npz_trajectory: NpzTrajectory):
    V = npz_trajectory.vector_data
    X, idx = V[:, :44], V[:, 44].flatten()
    width = npz_trajectory.get_scenario_width()

    lanes = [
        TrafficLane(X[idx == i][:, 0], X[idx == i][:, 1])
        for i in np.unique(idx)
        if X[idx == i][:, 13:16].sum() > 0
    ]

    results = Parallel(n_jobs=-1)(delayed(process_lane)(lane, width) for lane in lanes)
    return any(results)


# i = 0

# correct = 0
# incorrect = 0

# while i < 100:
#     rnd_traj = NpzTrajectory(get_random_npz_trajectory())
#     rnd_traj.plot_scenario()
#     user_input = input("Does this have a turnaround?")
#     if str(has_turnaround(rnd_traj)) == user_input:
#         correct += 1
#     else:
#         incorrect += 1
#     print(has_turnaround(rnd_traj))
#     # print(user_input)

#     # print(str(has_turnaround(rnd_traj)) == user_input)

#     print(incorrect)
#     print(correct)
#     help = input("Continue?")


# # print(lane_dist)

# # rel_lane_dists.append(lane_dist)

# # print(rel_lane_dists)

# # print(f"Counter: {counter}")

# # print(counter)
# # print(counter > 1)
