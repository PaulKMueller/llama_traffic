import json
import numpy as np
import wandb
import os

from tensorflow.keras.optimizers import Adam, SGD
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import load_model

from bert_encoder import init_bucket_embeddings

from sklearn.model_selection import train_test_split

from trajectory_generator import create_simple_neural_network
from trajectory_generator import create_neural_network

from wandb.keras import WandbMetricsLogger
from tqdm.keras import TqdmCallback

def infer_with_right_neural_network(input_data):
    model = load_model("models/right_model.h5")

    # Ensure the input is in the form of a 2D array
    # with shape (batch_size, input_features)
    if input_data.ndim == 1:
        input_data = np.expand_dims(input_data, axis=0)

    predictions = model.predict(input_data)
    print()
    return predictions

def infer_with_stationary_neural_network(input_data):
    model = load_model("models/my_stationary_model.h5")

    # Ensure the input is in the form of a 2D array
    # with shape (batch_size, input_features)
    if input_data.ndim == 1:
        input_data = np.expand_dims(input_data, axis=0)

    predictions = model.predict(input_data)
    return predictions

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