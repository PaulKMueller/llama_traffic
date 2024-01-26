import numpy as np
import os

path = "/home/pmueller/llama_traffic/datasets/npz_test_data/train-2e6/vehicle_d_119014_00001_5453322049.npz"

test_vehicle_files = []
for filename in os.listdir(
    "/mrtstorage/datasets/tmp/waymo_open_motion_processed/train-2e6/"
):
    if filename.startswith("vehicle"):
        test_vehicle_files.append(filename)

for file in test_vehicle_files:
    with np.load(path) as data:
        object_id = data["object_id"]
        raster = data["raster"]
        yaw = data["yaw"]
        shift = data["shift"]
        _gt_marginal = data["_gt_marginal"]
        gt_marginal = data["gt_marginal"]
        future_val_marginal = data["future_val_marginal"]
        gt_joint = data["gt_joint"]
        scenario_id = data["scenario_id"]
        type = data["self_type"]
        vector_data = data["vector_data"]

        V = vector_data
        X, idx = V[:, :44], V[:, 44].flatten()

        _X = X[(idx == 4)]
        crosswalk = _X[:, 29]

        if crosswalk.sum() > 1:
            print(file)
            print(crosswalk)
            print()
