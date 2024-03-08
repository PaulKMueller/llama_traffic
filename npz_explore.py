import sys
import numpy as np
import os
from tqdm import tqdm
from npz_utils import list_vehicle_files_absolute
import matplotlib.pyplot as plt
import random

path = list_vehicle_files_absolute()[random.randint(0, 400000)]


# npz_trajectory = NpzTrajectory(
#     "/mrtstorage/datasets/tmp/waymo_open_motion_processed/train-2e6/vehicle_a_67582_00003_5913311279.npz"
# )

# with np.load(path) as data:
#     print
# test_vehicle_files = []
# for filename in os.listdir(
#     "/storage_local/fzi_datasets_tmp/waymo_open_motion_dataset/unzipped/train-2e6/"
# ):
#     if filename.startswith("vehicle_a"):
#         test_vehicle_files.append(filename)

# print(test_vehicle_files[:10])

print("Got data")

vectors = {
    0: "position_x",
    1: "position_y",
    2: "speed",
    3: "velocity_yaw",
    4: "bbox_yaw",
    5: "length",
    6: "width",
    # 7-11: Agent type one-hot encoded
    7: "agent_unset",
    8: "agent_vehicle",
    9: "agent_pedestrian",
    10: "agent_cyclist",
    11: "agent_other",
    12: "IDX_NOT_USED",
    # 13-32: Lane and road line type one-hot encoded
    13: "lane_center_undefined",
    14: "lane_center_freeway",
    15: "lane_center_surface_street",
    16: "lane_center_bike_lane",
    17: "road_line_unknowm",
    18: "road_line_broken_single_white",
    19: "road_line_solid_single_white",
    20: "road_line_solid_double_white",
    21: "road_line_broken_single_yellow",
    22: "road_line_broken_double_yellow",
    23: "road_line_solid_single_yellow",
    24: "road_line_solid_double_yellow",
    25: "road_line_passing_double_yellow",
    26: "road_edge_unknown",
    27: "road_edge_boundary",
    28: "road_edge_median",
    29: "stop_sign",
    30: "crosswalk",
    31: "speed_bump",
    32: "driveway",
    33: "IDX_NOT_USED",
    # 34-42: Traffic light state one-hot encoded
    34: "traffic_light_state_unknown",
    35: "traffic_light_state_arrow_stop",
    36: "traffic_light_state_arrow_caution",
    37: "traffic_light_state_arrow_go",
    38: "traffic_light_state_stop",
    39: "traffic_light_state_caution",
    40: "traffic_light_state_go",
    41: "traffic_light_state_flashing_stop",
    42: "traffic_light_state_flashing_caution",
    43: "timestamp",
    44: "global_idx",
}

result = {
    0: 339036,
    1: 281681,
    2: 468108,
    3: 240047,
    4: 366851,
    5: 468108,
    6: 468108,
    7: 0,
    8: 468108,
    9: 296067,
    10: 84441,
    11: 0,
    12: 0,
    13: 0,
    14: 42215,
    15: 466749,
    16: 361176,
    17: 0,
    18: 0,
    19: 392401,
    20: 406947,
    21: 1525,
    22: 71703,
    23: 8415,
    24: 222725,
    25: 289073,
    26: 101676,
    27: 0,
    28: 429464,
    29: 316558,
    30: 161908,
    31: 209908,
    32: 179437,
    33: 0,
    34: 0,
    35: 0,
    36: 0,
    37: 0,
    38: 0,
    39: 0,
    40: 0,
    41: 0,
    42: 0,
    43: 468108,
    44: 468108,
}

output = {
    0: 0,
    1: 0,
    2: 0,
    3: 0,
    4: 0,
    5: 0,
    6: 0,
    7: 0,
    8: 0,
    9: 0,
    10: 0,
    11: 0,
    12: 0,
    13: 0,
    14: 0,
    15: 0,
    16: 0,
    17: 0,
    18: 0,
    19: 0,
    20: 0,
    21: 0,
    22: 0,
    23: 0,
    24: 0,
    25: 0,
    26: 0,
    27: 0,
    28: 0,
    29: 0,
    30: 0,
    31: 0,
    32: 0,
    33: 0,
    34: 0,
    35: 0,
    36: 0,
    37: 0,
    38: 0,
    39: 0,
    40: 0,
    41: 0,
    42: 0,
    43: 0,
    44: 0,
}

scenario = np.load(path)
keys = list(scenario.keys())
print(keys)

# for key in keys:
#     print(key)
#     # print(scenario[key])
#     print(scenario[key].shape)
#     print("\n")

vectors = scenario["raster"][:, :, :3] / 255
plt.imshow(vectors)


# for i in range(224):
#     for j in range(224):
#         other_channels = vectors[i][j][3:] / 255
#         if other_channels.sum() > 0:
#             print(other_channels)


plt.savefig("output/raster.png")

# np.set_printoptions(threshold=sys.maxsize)
# for i in vectors:
#     # print(i)
#     input("continue?")
# # print(vectors.shape)
# V = vectors
# X, idx = V[:, :44], V[:, 44].flatten()

# for i in np.unique(idx):
#     _X = X[(idx == i)]
#     print(_X.shape)
# for j in range(44):
#     print(_X[:, j].shape)

vector_idx_meaning = {
    0: "position_x",
    1: "position_y",
    2: "speed",
    3: "velocity_yaw",
    4: "bbox_yaw",
    5: "length",
    6: "width",
    # 7-11: Agent type one-hot encoded
    7: "agent_unset",
    8: "agent_vehicle",
    9: "agent_pedestrian",
    10: "agent_cyclist",
    11: "agent_other",
    12: "IDX_NOT_USED",
    # 13-32: Lane and road line type one-hot encoded
    13: "lane_center_undefined",
    14: "lane_center_freeway",
    15: "lane_center_surface_street",
    16: "lane_center_bike_lane",
    17: "road_line_unknowm",
    18: "road_line_broken_single_white",
    19: "road_line_solid_single_white",
    20: "road_line_solid_double_white",
    21: "road_line_broken_single_yellow",
    22: "road_line_broken_double_yellow",
    23: "road_line_solid_single_yellow",
    24: "road_line_solid_double_yellow",
    25: "road_line_passing_double_yellow",
    26: "road_edge_unknown",
    27: "road_edge_boundary",
    28: "road_edge_median",
    29: "stop_sign",
    30: "crosswalk",
    31: "speed_bump",
    32: "driveway",
    33: "IDX_NOT_USED",
    # 34-42: Traffic light state one-hot encoded
    34: "traffic_light_state_unknown",
    35: "traffic_light_state_arrow_stop",
    36: "traffic_light_state_arrow_caution",
    37: "traffic_light_state_arrow_go",
    38: "traffic_light_state_stop",
    39: "traffic_light_state_caution",
    40: "traffic_light_state_go",
    41: "traffic_light_state_flashing_stop",
    42: "traffic_light_state_flashing_caution",
    43: "timestamp",
    44: "global_idx",
}


# object_id
# ()

# raster
# (224, 224, 25)

# yaw
# ()

# shift
# (1, 2)

# _gt_marginal
# (80, 2)

# gt_marginal
# (80, 2)

# future_val_marginal
# (80,)

# gt_joint
# (4, 80, 2)

# future_val_joint
# (4, 80)

# scenario_id
# ()

# self_type
# (1,)

# vector_data
# (6038, 48)

# keys = list(vectors.keys())

# formatted = {}

# for key in keys:
#     formatted[vectors[key]] = result[key]

# print(formatted)

# for index in tqdm(range(len(test_vehicle_files))):
#     file = test_vehicle_files[index]
#     path = (
#         "/storage_local/fzi_datasets_tmp/waymo_open_motion_dataset/unzipped/train-2e6/"
#     )
#     with np.load(path + file) as data:
#         object_id = data["object_id"]
#         raster = data["raster"]
#         yaw = data["yaw"]
#         shift = data["shift"]
#         _gt_marginal = data["_gt_marginal"]
#         gt_marginal = data["gt_marginal"]
#         future_val_marginal = data["future_val_marginal"]
#         gt_joint = data["gt_joint"]
#         scenario_id = data["scenario_id"]
#         type = data["self_type"]
#         vector_data = data["vector_data"]
#         # print(data.shape)

# V = vector_data
#         # print(V.shape)
#         X, idx = V[:, :45], V[:, 44].flatten()
#         # print(X.shape)
#         # print()

#         # print(idx)
#         # print(idx.shape)
#         # print(X.shape)
#         # for i in np.unique(idx):
#         #     _X = X[(i == idx)]
#         #     print(_X.shape)
#         for key in list(vectors.keys()):
#             if X[:, key].sum() > 0:
#                 output[key] += 1

# print(output)

# # crosswalk = X[:, 29]

# # if crosswalk.sum() > 1:
# #     print(file)
# #     print(crosswalk)
# #     print(crosswalk.shape)
# #     print()
