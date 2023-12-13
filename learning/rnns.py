import numpy as np
import tensorflow as tf
import keras
from keras import layers
import json
from sklearn.model_selection import train_test_split

from wandb.keras import WandbMetricsLogger
import wandb
from tqdm.keras import TqdmCallback



def create_lstm_neural_network():
    model = keras.Sequential()
    model.add(layers.LSTM(40, input_shape=(20, 2)))
    model.add(layers.Dense(202))
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
    with open("datasets/labeled_trajectories.json", "r") as file:
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
            starting_xs = value["X"][0:20]
            starting_ys = value["Y"][0:20]

            # Coordinates as Numpy array
            coordinates = np.column_stack((value["X"], value["Y"]))
            X.append(starting_xs + starting_ys)
            Y.append(coordinates.flatten())

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
        epochs=200,
        batch_size=32,
        validation_split=0.1,
        verbose=0,
        callbacks=[TqdmCallback(verbose=2), WandbMetricsLogger()],
    )

    model.save("models/my_lstm_model.h5")
    test_loss = model.evaluate(X_test, Y_test)
    print(f"Test Loss: {test_loss}")
