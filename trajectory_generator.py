from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import load_model

import numpy as np


def create_neural_network():
    model = Sequential()
    model.add(Dense(101, activation='relu', input_shape=(101,)))  # Input layer
    model.add(Dense(64, activation='relu'))  # Hidden layer 1
    model.add(Dense(64, activation='relu'))  # Hidden layer 2
    model.add(Dense(64, activation='relu'))  # Hidden layer 3
    model.add(Dense(101, activation='linear'))  # Output layer for regression
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