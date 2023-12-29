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
from tensorflow.keras.models import load_model
from tensorflow.keras import Sequential

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
    model.add(Dense(202, activation="linear"))  # Output layer for regression

    mse_loss = tf.keras.losses.MeanSquaredError()
    model.compile(optimizer="adam", loss=mse_loss)  # Loss function for regression
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

    (
        rotated_ego_coords,
        rotation_angle,
        original_first_x,
        original_first_y,
    ) = Trajectory.get_rotated_ego_coordinates_from_coordinates(coordinates)
    input_x = [row["X"] for _, row in rotated_ego_coords.iterrows()]
    input_y = [row["Y"] for _, row in rotated_ego_coords.iterrows()]

    input_data = np.array(list(zip(input_x, input_y)))

    if input_data.ndim == 1:
        input_data = np.expand_dims(input_data, axis=0)

    input_data = input_data.flatten()

    prediction = model.predict(tf.expand_dims(input_data, axis=0))

    # Parsing the prediction
    x_coordinates = prediction[0][0::2]
    y_coordinates = prediction[0][1::2]

    print(len(x_coordinates))
    print(len(y_coordinates))

    prediction_df = pd.DataFrame(({"X": x_coordinates, "Y": y_coordinates}))
    print(prediction_df)
    prediction_df = Trajectory.get_coordinates_from_rotated_ego_coordinates(
        prediction_df, rotation_angle, original_first_x, original_first_y
    )

    return prediction_df


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
    with open("datasets/labeled_ego_trajectories.json", "r") as file:
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
            Y.append(coordinates)

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
