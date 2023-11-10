from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import load_model

from tqdm.keras import TqdmCallback

from sklearn.model_selection import train_test_split

from bert_encoder import get_bert_embedding

from bert_encoder import init_bucket_embeddings

import numpy as np
import json


def create_neural_network():
    model = Sequential()
    model.add(Dense(768, activation='relu', input_shape=(768,)))  # Input layer
    model.add(Dense(101, activation='relu'))  # Hidden layer 1
    model.add(Dense(64, activation='relu'))  # Hidden layer 2
    model.add(Dense(64, activation='relu'))  # Hidden layer 3
    model.add(Dense(64, activation='relu'))  # Hidden layer 4
    model.add(Dense(202, activation='linear'))  # Output layer for regression
    model.compile(optimizer='adam', loss='mean_squared_error')  # Loss function for regression
    model.save('models/my_model.h5')
    return model


def infer_with_neural_network(input_data):
    model = load_model('models/my_model.h5')

    # Ensure the input is in the form of a 2D array with shape (batch_size, input_features)
    if input_data.ndim == 1:
        input_data = np.expand_dims(input_data, axis=0)

    predictions = model.predict(input_data)
    return predictions


def train_neural_network():
    # Load labeled trajectory data
    with open('datasets/labeled_trajectories.json', 'r') as file:
        trajectories_data = json.load(file)

    bucket_embeddings = init_bucket_embeddings()

    X, Y = [], []
    for key, value in trajectories_data.items():
        bucket = value['Direction']
        coordinates = np.column_stack((value['X'], value['Y']))
        embedding = bucket_embeddings[bucket]
        X.append(embedding)
        Y.append(coordinates.flatten())

    X = np.array(X)
    Y = np.array(Y)

    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

    X_train = X_train.reshape(-1, 768)
    X_test = X_test.reshape(-1, 768)

    model = create_neural_network()

    model.fit(X_train, Y_train, epochs=10, batch_size=32, validation_split=0.1, verbose=0, callbacks=[TqdmCallback(verbose=2)])

    model.save('models/my_trained_model.h5')
    test_loss = model.evaluate(X_test, Y_test)
    print(f'Test Loss: {test_loss}')

    # Prediction and saving predicted trajectory
    text = 'Right'
    embedding = get_bert_embedding(text)
    predicted_trajectory = infer_with_neural_network(embedding).tolist()

    with open('predicted_trajectory.json', 'w') as file:
        json.dump(predicted_trajectory, file)