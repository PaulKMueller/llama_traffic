import numpy as np
import os

path = "/home/pmueller/llama_traffic/datasets/npz_test_data/train-2e6/vehicle_d_119014_00001_5453322049.npz"


# npz_trajectory = NpzTrajectory(
#     "/mrtstorage/datasets/tmp/waymo_open_motion_processed/train-2e6/vehicle_a_67582_00003_5913311279.npz"
# )

# with np.load(path) as data:
#     print
test_vehicle_files = []
for filename in os.listdir(
    "/mrtstorage/datasets/tmp/waymo_open_motion_processed/train-2e6/"
):
    if filename.startswith("vehicle_a"):
        test_vehicle_files.append(filename)

print("Got data")

for file in test_vehicle_files:
    path = "/home/pmueller/llama_traffic/datasets/npz_test_data/train-2e6/vehicle_b_121122_00001_8644612152.npz"
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
        # print(data.shape)

        V = vector_data
        # print(V.shape)
        X, idx = V[:, :44], V[:, 44].flatten()
        # print(X.shape)
        # print()

        # print(idx)
        # print(idx.shape)
        # print(X.shape)
        for i in np.unique(idx):
            _X = X[(idx == i)]
            if _X[:, 16].sum() > 0:
                print("Something to plot!")
                print(str(_X[:, 15].shape))

        # crosswalk = X[:, 29]

        # if crosswalk.sum() > 1:
        #     print(file)
        #     print(crosswalk)
        #     print(crosswalk.shape)
        #     print()
