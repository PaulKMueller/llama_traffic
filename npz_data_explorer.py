import numpy as np

with np.load("datasets/vehicle_a_13_00002_1954337085.npz") as data:
    # for key in data.keys():
    #     print(key)
    # print(f"Object ID: {data['object_id']}")
    # print(f"Yaw: {data['yaw']}")
    # print(f"Shift: {data['shift']}")
    # print(f"GT Marginal: {data['_gt_marginal']}")
    print(f"GT Joint: {data['gt_joint']}")
    print(f"Shape of GT Joint: {data['gt_joint'].shape}")
    # print(f"Future Val Marginal: {data['future_val_marginal']}")
    # print(f"Future Val Joint: {data['future_val_joint']}")
    # print(f"Scenario ID: {data['scenario_id']}")
    # print(f"Self Type: {data['self_type']}")
    # print(f"Vector Data: {data['vector_data']}")


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