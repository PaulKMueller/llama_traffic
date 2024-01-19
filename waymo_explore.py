# Imports
import os
import tarfile
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import tqdm

from waymo_open_dataset.wdl_limited.sim_agents_metrics import metric_features
from waymo_open_dataset.wdl_limited.sim_agents_metrics import metrics

from waymo_open_dataset.protos import scenario_pb2
from waymo_open_dataset.protos import sim_agents_submission_pb2

from waymo_open_dataset.utils.sim_agents import submission_specs
from waymo_open_dataset.utils.sim_agents import test_utils as sim_agents_test_utils
from waymo_open_dataset.utils.sim_agents import visualizations
from waymo_open_dataset.utils import trajectory_utils
from waymo_open_dataset import dataset_pb2 as open_dataset

# Set matplotlib to jshtml so animations work with colab.
from matplotlib import rc

rc("animation", html="jshtml")

# from waymo_open_dataset.protos import scenario_pb2

dataset = tf.data.TFRecordDataset(
    "/mrtstorage/datasets/tmp/waymo_open_motion_v_1_2_0/uncompressed/tf_example/training/training_tfexample.tfrecord-00998-of-01000",
    compression_type="",
)

print(dataset)

data = next(dataset.as_numpy_iterator())

with open("text.txt", "w") as file:
    file.write(str(data))

print(data)
print(type(data))
# scenario = scenario_pb2.Scenario.FromString(data)

# scenario_data = []
# for data in dataset:
#     proto_string = data.numpy()
#     proto = scenario_pb2.Scenario()
#     proto.ParseFromString(proto_string)
#     scenario_data.append(proto)

# print(scenario_data)


# with open("test.txt", "w") as file:
#     file.write(str(data))

# scenario_id = {"scenario/id": tf.io.FixedLenFeature([1], tf.string, default_value=None)}
# state = {"state/id": tf.io.FixedLenFeature([128], tf.float32, default_value=None)}
# state_type = {
#     "state/type": tf.io.FixedLenFeature([128], tf.float32, default_value=None)
# }

# parsed = tf.io.parse_single_example(data, state)
# print(parsed)

# print(data)
