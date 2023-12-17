from waymo_open_dataset import dataset_pb2 as open_dataset
import tensorflow as tf

FILENAME = "/mrtstorage/datasets/tmp/waymo_open_motion_v_1_2_0/uncompressed/tf_example/training/training_tfexample.tfrecord-00499-of-01000"
train_set = tf.data.TFRecordDataset(FILENAME, compression_type="")
for i, data in enumerate(train_set):
    frame = open_dataset.Frame()
    data = frame.ParseFromString(bytearray(data.numpy()))
    with open("explore.txt", "a") as output:
        output.write(str(data)+"\n")
