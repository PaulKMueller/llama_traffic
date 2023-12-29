import wandb
import json
import numpy as np
from sklearn.model_selection import train_test_split
from wandb.keras import WandbMetricsLogger
from tqdm.keras import TqdmCallback
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense


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
