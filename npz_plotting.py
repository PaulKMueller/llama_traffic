from npz_trajectory import NpzTrajectory
import pandas as pd
import numpy as np
import json


# # vehicle is type 1, pedestrian type 2
# npz_trajectory = NpzTrajectory(
#     "/home/pmueller/llama_traffic/datasets/npz_test_data/train-2e6/vehicle_c_847_00003_2857436651.npz"
# )
# # npz_trajectory = NpzTrajectory(
# #     "/home/pmueller/llama_traffic/datasets/npz_test_data/train-2e6/pedestrian_c_77063_00004_7456769354.npz"
# # )

# # self.loaded_npz_trajectory = npz_trajectory

# # print(npz_trajectory._gt_marginal)
# x = npz_trajectory._gt_marginal[:, 0]
# y = npz_trajectory._gt_marginal[:, 1]
# # print(x)
# # print(y)
# coordinates = pd.DataFrame({"X": x, "Y": y})
# npz_plot = npz_trajectory.visualize_raw_coordinates_without_scenario(coordinates)
# # npz_plot.savefig("output/test_npz_plot.png")
# # print(npz_trajectory.get_delta_angles(npz_trajectory.coordinates))
# # print(npz_trajectory.movement_vectors)
# # print(npz_trajectory.get_sum_of_delta_angles())

# # plot = npz_trajectory.plot_marginal_predictions_3d(npz_trajectory.vector_data)

# predictions = np.zeros(npz_trajectory.future_val_marginal.shape)
# prediction_dummy = np.zeros((6, 10, 2))

# # print(type(prediction_dummy))
# # print(npz_trajectory.future_val_marginal.shape)
# # print(npz_trajectory.future_val_marginal)

# # print(predictions.shape)
# # plot = npz_trajectory.plot_marginal_predictions_3d(
# #     vector_data=npz_trajectory.vector_data,
# #     is_available=npz_trajectory.future_val_marginal,
# #     gt_marginal=npz_trajectory.gt_marginal,
# #     predictions=prediction_dummy,
# #     confidences=np.zeros((6,)),
# #     # gt_marginal=npz_trajectory.gt_marginal,
# # )

# npz_trajectory.animate_trajectory()

print("Before loading")
with open("output/processed.json", "r") as file:
    trajectories_data = json.load(file)
print("After loading")
print(trajectories_data.values()[0])
