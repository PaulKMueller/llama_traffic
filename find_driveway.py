import numpy as np
from npz_utils import list_vehicle_files_absolute
import random


vehicles = list_vehicle_files_absolute(
    "/storage_local/fzi_datasets_tmp/waymo_open_motion_dataset/unzipped/train-2e6/"
)

vectors = {
    # 0: "position_x",
    # 1: "position_y",
    # 2: "speed",
    # 3: "velocity_yaw",
    # 4: "bbox_yaw",
    # 5: "length",
    # 6: "width",
    # 7-11: Agent type one-hot encoded
    # 7: "agent_unset",
    8: "vehicle",
    9: "pedestrian",
    10: "cyclist",
    # 11: "agent_other",
    # 13-32: Lane and road line type one-hot encoded
    # 13: "lane_center_undefined",
    14: "freeway",
    15: "surface_street",
    16: "bike_lane",
    # 17: "road_line_unknowm",
    18: "road_line_broken_single_white",
    19: "road_line_solid_single_white",
    20: "road_line_solid_double_white",
    21: "road_line_broken_single_yellow",
    22: "road_line_broken_double_yellow",
    23: "road_line_solid_single_yellow",
    24: "road_line_solid_double_yellow",
    25: "road_line_passing_double_yellow",
    # 26: "road_edge_unknown",
    # 27: "road_edge_boundary",
    # 28: "road_edge_median",
    30: "stop_sign",
    31: "crosswalk",
    32: "speed_bump",
    33: "driveway",
    # 34-42: Traffic light state one-hot encoded
    # 34: "traffic_light_state_unknown",
    # 35: "traffic_light_state_arrow_stop",
    # 36: "traffic_light_state_arrow_caution",
    # 37: "traffic_light_state_arrow_go",
    # 38: "traffic_light_state_stop",
    # 39: "traffic_light_state_caution",
    # 40: "traffic_light_state_go",
    # 41: "traffic_light_state_flashing_stop",
    # 42: "traffic_light_state_flashing_caution",
}
for vehicle_file_path in vehicles:
    label = ""
    with np.load(vehicle_file_path) as vehicle_data:
        X = vehicle_data["vector_data"][:, :45]
        for vector in list(vectors.keys()):
            if X[:, vector].sum() > 0:
                label += vectors[vector] + " "
    if "driveway" in label:
        print(vehicle_file_path)
