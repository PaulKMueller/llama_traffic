import pickle
import json
import numpy as np
import time


# def timed(func):
#     start = time.time()
#     func()
#     end = time.time()
#     print(end - start)


# @timed
# def test_data_loading_json():
#     start = time.time()

#     with open("datasets/encoder_output_vehicle_a_mse.json") as file:
#         data = json.load(file)
#     #     values = np.array(list(data.values()))
#     #     print(values.shape)

#     # np.save("datasets/encoder_output_a_mse", values)


# @timed
# def test_data_loading_numpy():
#     data = np.load("datasets/encoder_output_a_mse.npy")
#     print(data.shape)


# test_data_loading_json()
# test_data_loading_numpy()

with open("output/processed.json") as file:
    data = np.array(list(json.load(file).values()))

np.save("datasets/raw_direction_labels", data)
