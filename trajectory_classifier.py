import ast
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

import json


def train_classifier():
    # Step 1: Load the data
    # def load_data(file_path):
    #     with open(file_path, 'r') as file:
    #         data_dict_str = file.read()
    #         data_dict = ast.literal_eval(data_dict_str)
    #     return data_dict

    def load_data(filepath):
        """Loads trajectory data from a JSON file.

        Args:
            filepath (str): The path to the JSON file containing the trajectory data.

        Returns:
            dict: The loaded trajectory data.
        """
        with open(filepath, "r") as json_file:
            data_dict = json.load(json_file)
        return data_dict

    # Assuming the file is named 'trajectories.txt'
    data_dict = load_data("./dataset/labeled_trajectories.json")

    # Step 2: Preprocess the data
    def preprocess_data(data_dict):
        X = []
        Y = []
        for key, value in data_dict.items():
            # Assuming that X and Y coordinates are already normalized or processed.
            x_coords = value["X"]
            y_coords = value["Y"]
            direction = value["Direction"]
            # Flatten the x and y coordinates into a single feature vector
            trajectory_features = np.hstack((x_coords, y_coords))
            X.append(trajectory_features)
            Y.append(direction)
        return np.array(X), np.array(Y)

    X, Y = preprocess_data(data_dict)

    # Step 3: Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, Y, test_size=0.2, random_state=42
    )

    # Step 4: Define the classification model
    model = RandomForestClassifier(n_estimators=100, random_state=42)

    # Step 5: Train the model
    model.fit(X_train, y_train)

    # Step 6: Evaluate the model
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Model accuracy: {accuracy:.2f}")
