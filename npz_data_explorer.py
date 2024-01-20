import numpy as np

with np.load(
    "/home/pmueller/llama_traffic/datasets/npz_test_data/train-2e6/vehicle_c_847_00003_2857436651.npz"
) as data:
    for key in data.keys():
        print(key)
    # print(data["_gt_marginal"].shape)
    # print(data["raster"].shape)
    # print(data["scenario_id"])
    # print(f"Object ID: {data['object_id']}")
    # print(f"Yaw: {data['yaw']}")
    # print(f"Shift: {data['shift']}")
    # print(f"GT Marginal: {data['_gt_marginal']}")
    # print(f"GT Joint: {data['gt_joint'].shape}")
    # print(f"GT Joint: {data['gt_joint']}")
    # print(f"Shape of GT Joint: {data['gt_joint'].shape}")
    # print(f"Future Val Marginal: {data['future_val_marginal']}")
    print(f"Future Val Joint: {data['future_val_joint']}")
    print(f"Future Val Joint: {data['future_val_joint'].shape}")
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
