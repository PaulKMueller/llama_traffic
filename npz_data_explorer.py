import numpy as np
from npz_trajectory import NpzTrajectory
from waymo_inform import visualize_raw_coordinates_without_scenario

# with np.load(
#     "/mrtstorage/datasets/tmp/waymo_open_motion_processed/train-2e6/vehicle_b_78219_00001_4426517365.npz"
# ) as data:
# for key in data.keys():
#     print(key)
# print(data["raster"].shape)
# print(data["scenario_id"])
# print(f"Object ID: {data['object_id']}")
# print(f"Yaw: {data['yaw']}")
# print(f"Shift: {data['shift']}")
# print(data["_gt_marginal"].shape)
# print(f"_GT Marginal: {data['_gt_marginal']}")

# print(data["gt_marginal"].shape)
# print(f"GT Marginal: {data['gt_marginal']}")
# print(f"GT Joint: {data['gt_joint'].shape}")
# print(f"GT Joint: {data['gt_joint']}")
# print(f"Shape of GT Joint: {data['gt_joint'].shape}")
# print(f"Future Val Marginal: {data['future_val_marginal']}")
# print(f"Future Val Joint: {data['future_val_joint']}")
# print(f"Future Val Joint: {data['future_val_joint'].shape}")
# print(f"Scenario ID: {data['scenario_id']}")
# print(f"Self Type: {data['self_type']}")

# x = data["vector_data"][:, 0]
# y = data["vector_data"][:, 1]
# coordinates = list(zip(x, y))
# print(len(coordinates))
# print(f"Vector Data: {data['vector_data'][:, 0].min()}")
# print(f"Vector Data: {data['vector_data'][:, 0].max()}")
# print(f"Vector Data: {data['vector_data'][:, 1].min()}")
# print(f"Vector Data: {data['vector_data'][:, 1].max()}")


npz = NpzTrajectory(
    "/mrtstorage/datasets/tmp/waymo_open_motion_processed/train-2e6/vehicle_b_78219_00001_4426517365.npz"
)

print(npz.coordinates)
print(npz.direction)

plot = visualize_raw_coordinates_without_scenario(
    npz.coordinates["X"], npz.coordinates["Y"]
)
plot.savefig("output/test.png")

plot_2 = npz.plot_trajectory()
plot_2.savefig("output/test_2.png")

# print(npz.get_relative_displacement())
# Keys:
# object_id
# raster
# yaw
# shift
# _gt_marginal
# gt_marginal
# future_val_marginal
# gt_joint
# future_val_joint
# scenario_id
# self_type
# vector_data
