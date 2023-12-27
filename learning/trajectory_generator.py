import wandb

import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
    Dense,
    MultiHeadAttention,
    Dropout,
    LayerNormalization,
    Input,
)
from tensorflow.keras.optimizers import Adam, SGD
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import load_model

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

import pandas as pd

from tqdm.keras import TqdmCallback

from sklearn.model_selection import train_test_split

from bert_encoder import get_bert_embedding

from bert_encoder import init_bucket_embeddings

from trajectory import Trajectory

from wandb.keras import WandbMetricsLogger

import numpy as np
import json
import os


def create_neural_network() -> Sequential:
    """Creates a neural network with the dimensions to handle the input of a word embedding
    and a starting point for the trajectory. Its output dimension corresponds to 101 x coordinates
    and 101 y coordinates.

    Returns:
        Sequential: A model with the specified dimensions.
    """    
    model = Sequential()
    # Bert Embedding of size768 and two dimensions for the starting point
    model.add(Dense(770, activation="relu", input_shape=(770,)))  # Input layer
    model.add(Dense(101, activation="relu"))  # Hidden layer 1
    model.add(Dense(64, activation="relu"))  # Hidden layer 2
    model.add(Dense(64, activation="relu"))  # Hidden layer 3
    model.add(Dense(64, activation="relu"))  # Hidden layer 4
    model.add(Dense(202, activation="linear"))  # Output layer for regression
    model.compile(
        optimizer="adam", loss="mean_squared_error"
    )  # Loss function for regression
    model.save("models/my_model.h5")
    return model


def create_simple_neural_network() -> Sequential:
    model = Sequential()
    model.add(Dense(100, activation="relu"))  # Input layer
    model.add(Dense(64, activation="relu"))  # Hidden layer 1
    model.add(Dense(64, activation="relu"))  # Hidden layer 2
    model.add(Dense(64, activation="relu"))  # Hidden layer 3
    model.add(Dense(100, activation="linear"))  # Output layer for regression

    mse_loss = tf.keras.losses.MeanSquaredError()
    model.compile(
        optimizer="adam", loss=mse_loss
    )  # Loss function for regression
    return model


def infer_with_neural_network(input_data) -> np.array:
    model = load_model("models/my_model.h5")

    # Ensure the input is in the form of a 2D array with shape (batch_size, input_features)
    if input_data.ndim == 1:
        input_data = np.expand_dims(input_data, axis=0)

    predictions = model.predict(input_data)
    return predictions


def infer_with_simple_neural_network(trajectory: Trajectory) -> np.array:
    model = load_model("models/my_simple_model.h5")
    coordinates = trajectory.splined_coordinates[0:50]
    input_x = [row["X"] for _, row in coordinates.iterrows()]
    input_y = [row["Y"] for _, row in coordinates.iterrows()]

    input_data = np.array(list(zip(input_x, input_y)))

    if input_data.ndim == 1:
        input_data = np.expand_dims(input_data, axis=0)

    input_data = input_data.flatten()
    print(input_data.shape)

    predictions = model.predict(tf.expand_dims(input_data, axis=0))
    return predictions


def infer_with_right_neural_network(input_data) -> np.array:
    model = load_model("models/right_model.h5")

    # Ensure the input is in the form of a 2D array
    # with shape (batch_size, input_features)
    if input_data.ndim == 1:
        input_data = np.expand_dims(input_data, axis=0)

    predictions = model.predict(input_data)
    return predictions

def infer_with_stationary_neural_network(input_data) -> np.array:
    model = load_model("models/my_stationary_model.h5")

    # Ensure the input is in the form of a 2D array
    # with shape (batch_size, input_features)
    if input_data.ndim == 1:
        input_data = np.expand_dims(input_data, axis=0)

    predictions = model.predict(input_data)
    return predictions


def train_simple_neural_network():
    direction_counter_dict = {
        "Left": 0,
        "Right": 0,
        "Stationary": 0,
        "Right-U-Turn": 0,
        "Left-U-Turn": 0,
        "Straight-Right": 0,
        "Straight-Left": 0,
        "Straight": 0,
    }
    # Load labeled trajectory data
    with open("datasets/zipped_labeled_trajectories.json", "r") as file:
        trajectories_data = json.load(file)

    X, Y = [], []

    for value in trajectories_data.values():
        direction = value["Direction"]
        skip = False

        for counter in direction_counter_dict.keys():
            if direction_counter_dict[counter] >= 20000:
                pass
                # skip = True
            if direction == counter:
                direction_counter_dict[counter] += 1

        if not skip:
            starting_points = np.array(value["Coordinates"][0:50]).flatten()

            # Coordinates as Numpy array
            coordinates = np.array(value["Coordinates"]).flatten()
            print(starting_points)
            X.append(starting_points)
            Y.append(starting_points)

    X = np.array(X)
    print(X.shape)
    print("Test")
    Y = np.array(Y)
    print(Y.shape)

    X_train, X_test, Y_train, Y_test = train_test_split(
        X, Y, test_size=0.2, random_state=42
    )

    model = create_simple_neural_network()

    wandb.init(config={"bs": 12})

    model.fit(
        X_train,
        Y_train,
        epochs=200,
        batch_size=32,
        validation_split=0.1,
        verbose=0,
        callbacks=[TqdmCallback(verbose=2), WandbMetricsLogger()],
    )

    model.save("models/my_simple_model.h5")
    test_loss = model.evaluate(X_test, Y_test)
    print(f"Test Loss: {test_loss}")
    wandb.finish()


def train_neural_network():
    # Load labeled trajectory data
    with open("datasets/labeled_trajectories.json", "r") as file:
        trajectories_data = json.load(file)

    bucket_embeddings = {}

    # Check if embeddings have already been initialized
    if not os.path.exists("bucket_embeddings.json"):
        bucket_embeddings = init_bucket_embeddings()
    else:
        with open("bucket_embeddings.json", "r") as file:
            bucket_embeddings = json.load(file)

    # for key, value in bucket_embeddings.items():
    #     print(len(value))

    X, Y = [], []
    for value in trajectories_data.values():
        embedding = []

        bucket = value["Direction"]
        # print(bucket)
        # Bucket embedding as list
        embedding = bucket_embeddings[bucket].copy()

        starting_x = value["X"][0]
        starting_y = value["Y"][0]
        embedding.append(starting_x)
        embedding.append(starting_y)

        # Coordinates as Numpy array
        coordinates = np.column_stack((value["X"], value["Y"]))
        X.append(embedding)
        Y.append(coordinates.flatten())

    X = np.array(X)
    Y = np.array(Y)

    X_train, X_test, Y_train, Y_test = train_test_split(
        X, Y, test_size=0.2, random_state=42
    )

    X_train = X_train.reshape(-1, 770)
    X_test = X_test.reshape(-1, 770)

    model = create_neural_network()

    wandb.init(config={"bs": 12})

    model.fit(
        X_train,
        Y_train,
        epochs=20,
        batch_size=32,
        validation_split=0.1,
        verbose=0,
        callbacks=[TqdmCallback(verbose=2), WandbMetricsLogger()],
    )

    model.save("models/my_model.h5")
    test_loss = model.evaluate(X_test, Y_test)
    print(f"Test Loss: {test_loss}")

    # Prediction and saving predicted trajectory
    text = "Right"
    embedding = get_bert_embedding(text)
    predicted_trajectory = infer_with_neural_network(embedding).tolist()

    with open("predicted_trajectory.json", "w") as file:
        json.dump(predicted_trajectory, file)

def train_stationary_neural_network():
    # Load labeled trajectory data
    with open("datasets/labeled_trajectories.json", "r") as file:
        trajectories_data = json.load(file)

    X, Y = [], []

    for value in trajectories_data.values():
        direction = value["Direction"]
        if direction == "Stationary":
            starting_xs = value["X"][0:20]
            starting_ys = value["Y"][0:20]

            # Coordinates as Numpy array
            coordinates = np.concatenate((value["X"], value["Y"]))
            X.append(starting_xs + starting_ys)
            Y.append(coordinates)

    X = np.array(X)
    print(X.shape)
    Y = np.array(Y)
    print(Y.shape)

    X_train, X_test, Y_train, Y_test = train_test_split(
        X, Y, test_size=0.2, random_state=42
    )

    model = create_simple_neural_network()

    wandb.init(config={"bs": 12})

    model.fit(
        X_train,
        Y_train,
        epochs=200,
        batch_size=32,
        validation_split=0.1,
        verbose=0,
        callbacks=[TqdmCallback(verbose=2), WandbMetricsLogger()],
    )

    model.save("models/my_stationary_model.h5")
    test_loss = model.evaluate(X_test, Y_test)
    print(f"Test Loss: {test_loss}")
    wandb.finish()


def train_right_neural_network():
    # Load labeled trajectory data
    with open("datasets/labeled_trajectories.json", "r") as file:
        trajectories_data = json.load(file)

    bucket_embeddings = {}

    # Check if embeddings have already been initialized
    if not os.path.exists("bucket_embeddings.json"):
        bucket_embeddings = init_bucket_embeddings()
    else:
        with open("bucket_embeddings.json", "r") as file:
            bucket_embeddings = json.load(file)

    # for key, value in bucket_embeddings.items():
    #     print(len(value))

    X, Y = [], []
    for value in trajectories_data.values():
        embedding = []

        bucket = value["Direction"]
        # print(bucket)
        # Bucket embedding as list
        if bucket == "Right":
            embedding = bucket_embeddings[bucket].copy()

            starting_x = value["X"][0]
            starting_y = value["Y"][0]
            embedding.append(starting_x)
            embedding.append(starting_y)

            # Coordinates as Numpy array
            coordinates = np.concatenate((value["X"], value["Y"]))
            X.append(embedding)
            Y.append(coordinates)

    X = np.array(X)
    Y = np.array(Y)

    X_train, X_test, Y_train, Y_test = train_test_split(
        X, Y, test_size=0.2, random_state=42
    )

    X_train = X_train.reshape(-1, 770)
    X_test = X_test.reshape(-1, 770)

    model = create_neural_network()

    wandb.init(config={"bs": 12})

    model.fit(
        X_train,
        Y_train,
        epochs=50,
        batch_size=32,
        validation_split=0.1,
        verbose=0,
        callbacks=[TqdmCallback(verbose=2), WandbMetricsLogger()],
    )

    model.save("models/right_model.h5")
    test_loss = model.evaluate(X_test, Y_test)
    print(f"Test Loss: {test_loss}")

    # # Prediction and saving predicted trajectory
    # text = "Right"
    # embedding = get_bert_embedding(text)
    # np.append(embedding, [0, 0])
    # predicted_trajectory = infer_with_neural_network(embedding).tolist()

    # with open("predicted_trajectory.json", "w") as file:
    #     json.dump(predicted_trajectory, file)


def positional_encoding(positions, d_model):
    def get_angles(pos, i, d_model):
        angle_rates = 1 / tf.pow(10000, (2 * (i // 2)) / tf.cast(d_model, tf.float32))
        return pos * angle_rates

    angle_rads = get_angles(
        tf.range(positions, dtype=tf.float32)[:, tf.newaxis],
        tf.range(d_model, dtype=tf.float32)[tf.newaxis, :],
        d_model,
    )

    # Apply sin to even indices in the array; 2i
    sines = tf.math.sin(angle_rads[:, 0::2])

    # Apply cos to odd indices in the array; 2i+1
    cosines = tf.math.cos(angle_rads[:, 1::2])

    angle_rads = tf.concat([sines, cosines], axis=-1)
    pos_encoding = angle_rads[tf.newaxis, ...]

    return pos_encoding


class TransformerEncoderLayer(tf.keras.layers.Layer):
    def __init__(self, d_model, num_heads, ff_dim, rate=0.1):
        super(TransformerEncoderLayer, self).__init__()
        self.mha = MultiHeadAttention(key_dim=d_model, num_heads=num_heads)
        self.ffn = tf.keras.Sequential(
            [Dense(ff_dim, activation="relu"), Dense(d_model)]
        )
        self.layernorm1 = LayerNormalization(epsilon=1e-6)
        self.layernorm2 = LayerNormalization(epsilon=1e-6)
        self.dropout1 = Dropout(rate)
        self.dropout2 = Dropout(rate)

    def call(self, x, training):
        attn_output = self.mha(x, x, x)  # Self attention
        attn_output = self.dropout1(attn_output, training=training)
        out1 = self.layernorm1(x + attn_output)  # Skip connection

        ffn_output = self.ffn(out1)
        ffn_output = self.dropout2(ffn_output, training=training)
        out2 = self.layernorm2(out1 + ffn_output)  # Skip connection

        return out2


def create_transformer_network(
    input_shape=(1, 768), num_layers=4, d_model=768, num_heads=8, ff_dim=2048, rate=0.1
):
    inputs = Input(shape=input_shape)
    x = inputs

    pos_encoding = positional_encoding(
        input_shape[0], d_model
    )  # Adjusted for input shape
    x += pos_encoding[:, : input_shape[0], :]

    for _ in range(num_layers):
        x = TransformerEncoderLayer(d_model, num_heads, ff_dim, rate)(x)

    x = tf.keras.layers.GlobalAveragePooling1D()(x)
    x = Dense(202, activation="linear")(x)  # Output layer

    return Model(inputs=inputs, outputs=x)


def create_transformer_model():
    # Create the transformer model
    model = create_transformer_network()
    model.compile(optimizer=Adam(1e-4), loss="mean_squared_error")
    model.save("models/my_transformer_model.h5")
    return model


def infer_with_transformer_network(
    embedding, model_path="models/my_trained_transformer_model.h5", seq_length=1
):
    """
    Infer trajectory using a transformer network.

    :param embedding: The input embedding for the trajectory prediction.
    :param model_path: Path to the saved transformer model.
    :param seq_length: The sequence length expected by the transformer model.
    :return: Predicted trajectory.
    """
    # Load the trained transformer model
    model = tf.keras.models.load_model(model_path)

    # Reshape the input embedding to match the transformer's expected input shape
    # Assuming the embedding is flat and needs to be reshaped into a sequence
    embedding = np.array(embedding).reshape(
        -1, seq_length, int(len(embedding) / seq_length)
    )

    # Predict the trajectory
    predicted_trajectory = model.predict(embedding)

    # Reshape or process the output as needed, here assuming it's flattened
    predicted_trajectory = predicted_trajectory.flatten()

    return predicted_trajectory


def train_transformer_network():
    # Load labeled trajectory data
    with open(
        "/home/pmueller/llama_traffic/datasets/labeled_trajectories.json", "r"
    ) as file:
        trajectories_data = json.load(file)

    # Dictionary with buckets as key
    bucket_embeddings = init_bucket_embeddings()

    X, Y = [], []
    for key, value in trajectories_data.items():
        bucket = value["Direction"]
        coordinates = np.column_stack((value["X"], value["Y"]))
        embedding = bucket_embeddings[bucket]
        X.append(embedding)
        Y.append(coordinates.flatten())

    X = np.array(X)
    Y = np.array(Y)

    X_train, X_test, Y_train, Y_test = train_test_split(
        X, Y, test_size=0.2, random_state=42
    )
    print(f"X_train Shape before: {X_train.shape}")

    # Reshape the data for the transformer model
    # Assuming each embedding is a sequence of vectors
    # seq_length = 48  # This is an example, adjust it according to your data's characteristics
    # feature_per_step = int(768 / seq_length)

    # seq_length = 1  # Number of time steps in the sequence
    # # feature_per_step = int(768 / seq_length)  # Number of features per time step

    # feature_per_step = 768

    # X_train = X_train.reshape(32, seq_length, feature_per_step)
    # X_test = X_test.reshape(32, seq_length, feature_per_step)

    # print(f"X_train Shape after: {X_train.shape}")

    # Create the transformer model
    model = create_transformer_network()
    model.compile(optimizer="adam", loss="mean_squared_error")
    model.fit(
        X_train,
        Y_train,
        epochs=10,
        batch_size=32,
        validation_split=0.1,
        verbose=0,
        callbacks=[TqdmCallback(verbose=2)],
    )

    model.save("/home/pmueller/llama_traffic/models/my_trained_transformer_model.h5")
    test_loss = model.evaluate(X_test, Y_test)
    print(f"Test Loss: {test_loss}")

    # Prediction and saving predicted trajectory
    text = "Right"
    embedding = get_bert_embedding(text)
    predicted_trajectory = infer_with_neural_network(embedding).tolist()

    with open("predicted_trajectory.json", "w") as file:
        json.dump(predicted_trajectory, file)
