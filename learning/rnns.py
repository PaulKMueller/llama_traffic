import numpy as np
import tensorflow as tf
import keras
from keras import layers
import json
from sklearn.model_selection import train_test_split

from wandb.keras import WandbMetricsLogger
import wandb
from tqdm.keras import TqdmCallback


def create_rnn_neural_network():
    model = keras.Sequential([
    layers.SimpleRNN(1, input_shape=[None, 2]),
    layers.Dense(2)
    ])
    model.compile(optimizer="adam", loss="mean_squared_error")
    model.save("models/my_rnn_model.h5")
    return model

def train_rnn_neural_network():
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
            if direction_counter_dict[counter] >= 500:
                skip = True
            if direction == counter:
                direction_counter_dict[counter] += 1

        if not skip:
            all_points = np.array(value["Coordinates"])
            starting_points = np.array(all_points[0:99])
            end_point = np.array(all_points[100])

            # Coordinates as Numpy array
            X.append(starting_points)
            Y.append(end_point)

    X = np.array(X)
    print(X.shape)
    Y = np.array(Y)
    print(Y.shape)

    X_train, X_test, Y_train, Y_test = train_test_split(
        X, Y, test_size=0.2, random_state=42
    )

    model = create_rnn_neural_network()

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

    model.save("models/my_rnn_model.h5")
    test_loss = model.evaluate(X_test, Y_test)
    print(f"Test Loss: {test_loss}")


def create_lstm_neural_network():
    model = keras.Sequential([
        layers.LSTM(202, return_sequences=True, input_shape=(None, 2)),
        layers.Dense(2)])
    model.compile(
        optimizer="adam", loss="mean_squared_error"
    )  # Loss function for regression
    model.save("models/my_lstm_model.h5")
    return model


def train_lstm_neural_network():
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
            if direction_counter_dict[counter] >= 500:
                skip = True
            if direction == counter:
                direction_counter_dict[counter] += 1

        if not skip:
            starting_points = np.array(value["Coordinates"][0:100])

            # Coordinates as Numpy array
            all_points = np.array(value["Coordinates"])
            X.append(starting_points)
            Y.append(all_points.flatten())

    X = np.array(X)
    print(X.shape)
    Y = np.array(Y)
    print(Y.shape)

    X_train, X_test, Y_train, Y_test = train_test_split(
        X, Y, test_size=0.2, random_state=42
    )

    model = create_lstm_neural_network()

    wandb.init(config={"bs": 12})

    model.fit(
        X_train,
        Y_train,
        epochs=2000,
        batch_size=32,
        validation_split=0.1,
        verbose=0,
        callbacks=[TqdmCallback(verbose=2), WandbMetricsLogger()],
    )

    model.save("models/my_lstm_model.h5")
    test_loss = model.evaluate(X_test, Y_test)
    print(f"Test Loss: {test_loss}")

# shape: (batch_size, sequence length, features per sequence)
# inputs = np.random.random((32, 40, 2))
# model =layers.LSTM(202, input_shape=(32, 40, 2))
# print(inputs)
# print(model(inputs))
# print(model(inputs).shape)