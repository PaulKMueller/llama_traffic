import tensorflow as tf


def get_feature_description(example_proto):
    # Create a description of the features
    feature_description = {}
    for key, feature in example_proto.features.feature.items():
        kind = feature.WhichOneof("kind")
        if kind == "bytes_list":
            dtype = tf.string
            feature_description[key] = tf.io.VarLenFeature(dtype)
        elif kind == "float_list":
            dtype = tf.float32
            feature_description[key] = tf.io.VarLenFeature(dtype)
        elif kind == "int64_list":
            dtype = tf.int64
            feature_description[key] = tf.io.VarLenFeature(dtype)
        else:
            raise ValueError(f"Unsupported feature type: {kind}")

    return feature_description


def parse_tfrecord(tfrecord_path):
    # Load the dataset
    dataset = tf.data.TFRecordDataset(tfrecord_path)
    output = ""
    for raw_record in dataset.take(1):  # Taking only one record to infer structure
        example = tf.train.Example()
        example.ParseFromString(raw_record.numpy())

        feature_description = get_feature_description(example)

        # Parse the record into tensors
        parsed_record = tf.io.parse_single_example(raw_record, feature_description)

        for key, feature in parsed_record.items():
            if isinstance(feature, tf.SparseTensor):
                value = tf.sparse.to_dense(feature).numpy()
            else:
                value = feature.numpy()
            output = output + "\n" + f"Feature: {key}"
            output = output + "\n" + f" - Value: {value}"
            output = output + "\n" + f" - Shape: {value.shape}"
            output = output + "\n" + f" - DataType: {feature.dtype}\n"

    return output


# Provide the path to your TFRecord file
tfrecord_path = "/mrtstorage/datasets/tmp/waymo_open_motion_v_1_2_0/uncompressed/tf_example/training/training_tfexample.tfrecord-00998-of-01000"
structure = parse_tfrecord(tfrecord_path)


with open("data/structure_with_datatypes.txt", "w") as file:
    file.write(str(structure))
