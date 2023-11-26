import numpy as np
import pandas as pd
import tensorflow as tf
from scipy import interpolate
import math
from scenario import Scenario


class Trajectory:
    def __init__(self, scenario: Scenario, specific_id):
        self.scenario = scenario
        self.coordinates = self.get_coordinates(scenario.data, specific_id)
        self.splined_coordinates = self.get_spline_for_coordinates(self.coordinates)
        self.x_coordinates = self.splined_coordinates["X"]
        self.y_coordinates = self.splined_coordinates["Y"]

    @staticmethod
    def get_coordinates_one_step(
        states,
        mask,
        agent_ids=None,
        specific_id: float = None,
    ):
        """Get coordinates for one vehicle for one step."""

        # If a specific ID is provided, filter the states,
        # masks, and colors to only include that ID.

        if specific_id is not None:
            n = 128
            mask = np.full(n, False)
            index_of_id = np.where(agent_ids == float(specific_id))
            mask[index_of_id] = True
        else:
            print("Please provide a specific vehicle ID!")
            return

        masked_x = states[:, 0][mask]
        masked_y = states[:, 1][mask]

        return {"X": masked_x[0], "Y": masked_y[0]}

    def get_coordinates(self, decoded_example, specific_id: float = None):
        """Returns the coordinates of the vehicle identified by its
        specific_id and stores them as a CSV in the output folder.

        Args:
            decoded_example: Dictionary containing agent info about all modeled agents.
            specific_id: The idea for which to store the coordinates.

        Returns:
            pandas.dataframe: The coordinates of the vehicle identified by its
            specific_id.
        """

        output_df = pd.DataFrame(columns=["X", "Y"])

        agent_ids = decoded_example["state/id"].numpy()

        # [num_agents, num_past_steps, 2] float32.
        past_states = tf.stack(
            [decoded_example["state/past/x"], decoded_example["state/past/y"]], -1
        ).numpy()
        past_states_mask = decoded_example["state/past/valid"].numpy() > 0.0

        # [num_agents, 1, 2] float32.
        current_states = tf.stack(
            [decoded_example["state/current/x"], decoded_example["state/current/y"]], -1
        ).numpy()
        current_states_mask = decoded_example["state/current/valid"].numpy() > 0.0

        # [num_agents, num_future_steps, 2] float32.
        future_states = tf.stack(
            [decoded_example["state/future/x"], decoded_example["state/future/y"]], -1
        ).numpy()
        future_states_mask = decoded_example["state/future/valid"].numpy() > 0.0

        _, num_past_steps, _ = past_states.shape
        num_future_steps = future_states.shape[1]

        for _, (s, m) in enumerate(
            zip(
                np.split(past_states, num_past_steps, 1),
                np.split(past_states_mask, num_past_steps, 1),
            )
        ):
            coordinates_for_step = self.get_coordinates_one_step(
                s[:, 0], m[:, 0], agent_ids=agent_ids, specific_id=specific_id
            )
            coordinates_for_step_df = pd.DataFrame([coordinates_for_step])
            output_df = pd.concat(
                [output_df, coordinates_for_step_df], ignore_index=True
            )

        # Generate one image for the current time step.
        s = current_states
        m = current_states_mask

        coordinates_for_step = self.get_coordinates_one_step(
            s[:, 0], m[:, 0], agent_ids=agent_ids, specific_id=specific_id
        )
        coordinates_for_step_df = pd.DataFrame([coordinates_for_step])

        output_df = pd.concat([output_df, coordinates_for_step_df], ignore_index=True)

        # Generate images from future time steps.
        for _, (s, m) in enumerate(
            zip(
                np.split(future_states, num_future_steps, 1),
                np.split(future_states_mask, num_future_steps, 1),
            )
        ):
            coordinates_for_step = self.get_coordinates_one_step(
                s[:, 0], m[:, 0], agent_ids=agent_ids, specific_id=specific_id
            )
            coordinates_for_step_df = pd.DataFrame([coordinates_for_step])
            output_df = pd.concat(
                [output_df, coordinates_for_step_df], ignore_index=True
            )

        # Delete all rows where both X and Y are -1.0
        output_df = output_df[~((output_df["X"] == -1.0) & (output_df["Y"] == -1.0))]

        output_df = output_df.reset_index(drop=True)

        return output_df

    def get_spline_for_coordinates(self, coordinates):
        """Returns the splined coordinates for the given trajectory coordinates.

        Args:
            coordinates (pd.DataFrame): The coordinates of a vehicle represented as a DataFrame
        """
        # Get the x and y coordinates
        x = coordinates["X"]
        y = coordinates["Y"]

        filtered_x = [x[0]]
        filtered_y = [y[0]]

        threshold = 1e-5
        for i in range(1, len(x)):
            if np.sqrt((x[i] - x[i - 1]) ** 2 + (y[i] - y[i - 1]) ** 2) > threshold:
                filtered_x.append(x[i])
                filtered_y.append(y[i])

        # Check if there are more data points than the minimum required for a spline
        if len(filtered_x) < 4 or len(filtered_y) < 4:
            return self.get_adjusted_coordinates(coordinates)

        # Check if the coordinates are constant

        if len(set(filtered_x)) <= 1 and len(set(filtered_y)) <= 1:
            print("Both x and y are constant. Cannot fit a spline.")
            return self.get_adjusted_coordinates(coordinates)
        elif len(set(filtered_x)) <= 1:
            print("x is constant. Cannot fit a spline.")
            return self.get_adjusted_coordinates(coordinates)
        elif len(set(filtered_y)) <= 1:
            print("y is constant. Cannot fit a spline.")
            return self.get_adjusted_coordinates(coordinates)
        else:
            # Call splprep
            tck, u = interpolate.splprep([filtered_x, filtered_y], s=12)

        # Get the spline for the x and y coordinates
        unew = np.arange(0, 1.01, 0.01)
        spline = interpolate.splev(unew, tck)

        result = pd.DataFrame({"X": spline[0], "Y": spline[1]})

        return result

    @staticmethod
    def get_adjusted_coordinates(coordinates):
        """For given coordinates returns their adjusted coordinates.
        This means that the coordinates are adapted to have 101 X coordinates and 101 Y coordinates.

        Args:
            coordinates (pd.DataFrame): The coordinates of a vehicle represented as a DataFrame
        """

        adjusted_coordinates = pd.DataFrame(columns=["X", "Y"])
        x = coordinates["X"]
        y = coordinates["Y"]

        # Copy the first X coordinate 101 times
        adjusted_coordinates["X"] = [x[0]] * 101
        # Copy the first Y coordinate 101 times
        adjusted_coordinates["Y"] = [y[0]] * 101

        return adjusted_coordinates

    def get_sum_of_delta_angles(self, coordinates: pd.DataFrame):
        """Returns the sum of the angles between each segment in the trajectory.

        Args:
            coordinates (pd.DataFrame): A dataframe containing the coordinates
                                        of the vehicle trajectory.
        """
        delta_angles = self.get_delta_angles(coordinates)
        filtered_delta_angles = self.remove_outlier_angles(delta_angles)
        return sum(filtered_delta_angles)

    def get_delta_angles(self, coordinates: pd.DataFrame):
        """Returns the angle between each segment in the trajectory.

        Args:
            coordinates (pd.DataFrame): A dataframe containing the coordinates
                                        of the vehicle trajectory.
        """
        delta_angles = []

        for i in range(1, len(coordinates) - 1):
            # Calculate the direction vector of the current segment
            current_vector = np.array(
                (
                    coordinates.iloc[i + 1]["X"] - coordinates.iloc[i]["X"],
                    coordinates.iloc[i + 1]["Y"] - coordinates.iloc[i]["Y"],
                )
            )

            # Calculate the direction vector of the previous segment
            previous_vector = np.array(
                (
                    coordinates.iloc[i]["X"] - coordinates.iloc[i - 1]["X"],
                    coordinates.iloc[i]["Y"] - coordinates.iloc[i - 1]["Y"],
                )
            )

            # Compute the angle between the current and previous direction vectors
            angle = self.get_angle_between_vectors(current_vector, previous_vector)

            direction = self.get_gross_direction_for_three_points(
                coordinates.iloc[i - 1], coordinates.iloc[i], coordinates.iloc[i + 1]
            )

            if direction == "Right":
                angle = -angle

            delta_angles.append(angle)

        return delta_angles

    @staticmethod
    def remove_outlier_angles(delta_angles: list):
        """Removes outlier angles from a list of angles.

        Args:
            delta_angles (list): A list of angles.
        """

        filtered_delta_angles = []

        for angle in delta_angles:
            if angle < 20 and angle > -20:
                filtered_delta_angles.append(angle)

        return filtered_delta_angles

    @staticmethod
    def get_gross_direction_for_three_points(
        start: pd.DataFrame, intermediate: pd.DataFrame, end: pd.DataFrame
    ):
        """Returns left, right, or straight depending on the direction of the trajectory.

        Args:
            start (pd.DataFrame): The coordinates of the starting point.
            intermediate (pd.DataFrame): The coordinates of the intermediate point.
            end (pd.DataFrame): The coordinates of the ending point.
        """
        # Calculate vectors
        vector1 = np.array(
            (intermediate["X"] - start["X"], intermediate["Y"] - start["Y"])
        )
        vector2 = np.array((end["X"] - intermediate["X"], end["Y"] - intermediate["Y"]))

        # Calculate the cross product of the two vectors
        cross_product = np.cross(vector1, vector2)

        # Determine direction based on cross product
        if cross_product > 0:
            direction = "Left"
        elif cross_product < 0:
            direction = "Right"
        else:
            direction = "Straight"

        return direction

    def get_angle_between_vectors(self, v1, v2):
        """Returns the angle between two vectors.

        Args:
            v1 (np.array): The first vector.
            v2 (np.array): The second vector.
        """
        v1_length = np.linalg.norm(v1)
        v2_length = np.linalg.norm(v2)
        if v1_length == 0 or v2_length == 0:
            return 0

        product = (v1 @ v2) / (v1_length * v2_length)

        if product > 1:
            return 0
        if product < -1:
            return 180

        acos = math.acos(product)
        result_angle = acos * (180 / math.pi)

        if result_angle > 180:
            result_angle = 360 - result_angle
        return result_angle
