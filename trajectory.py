from typing import Tuple
import numpy as np
import pandas as pd
import tensorflow as tf
from scipy import interpolate
import math
from scenario import Scenario

import matplotlib.pyplot as plt


class Trajectory:
    def __init__(self, scenario: Scenario, specific_id):
        self.scenario = scenario
        self.coordinates = self.get_coordinates(scenario.data, specific_id)
        self.splined_coordinates = self.get_spline_for_coordinates(self.coordinates)
        self.x_coordinates = self.splined_coordinates["X"]
        self.y_coordinates = self.splined_coordinates["Y"]
        self.relative_displacement = self.get_relative_displacement()
        self.normalized_splined_coordinates = self.normalize_coordinates()
        self.total_displacement = self.get_total_displacement()
        self.sum_of_delta_angles = self.get_sum_of_delta_angles()
        self.direction = self.get_direction_of_vehicle()
        self.ego_coordinates = self.get_ego_coordinates()
        self.x_axis_angle = self.get_x_axis_angle()
        self.rotated_coordinates = self.get_rotated_ego_coordinates()

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

    def normalize_coordinates(self):
        viewport = self.scenario.get_viewport()
        center_y = viewport[0]
        center_x = viewport[1]
        width = viewport[2]
        normalized_coordinates = pd.DataFrame(columns=["X", "Y"])

        for i in range(len(self.splined_coordinates)):
            normalized_x = (self.splined_coordinates["X"][i] - center_x) / width
            normalized_y = (self.splined_coordinates["Y"][i] - center_y) / width
            # Concatenate the normalized coordinates to the normalized_coordinates dataframe
            normalized_coordinates = pd.concat(
                [
                    normalized_coordinates,
                    pd.DataFrame({"X": [normalized_x], "Y": [normalized_y]}),
                ],
                ignore_index=True,
            )
        return normalized_coordinates

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

    def get_ego_coordinates(self) -> pd.DataFrame:
        """Returns the ego coordinates for the trajectory.

        Returns:
            pd.DataFrame: Ego coordinates. These are the coordinates that start at (0, 0).
        """

        first_x_coordinate = self.splined_coordinates["X"][0]
        first_y_coordinate = self.splined_coordinates["Y"][0]

        ego_coordinates = self.splined_coordinates.copy()
        ego_coordinates["X"] = ego_coordinates["X"] - first_x_coordinate
        ego_coordinates["Y"] = ego_coordinates["Y"] - first_y_coordinate

        return ego_coordinates

    def get_rotated_ego_coordinates(self) -> pd.DataFrame:
        rotated_coordinates = self.ego_coordinates.copy()
        for index, row in self.ego_coordinates.iterrows():
            rotated_x = (
                math.cos(self.x_axis_angle) * row["X"]
                - math.sin(self.x_axis_angle) * row["Y"]
            )
            rotated_y = (
                math.sin(self.x_axis_angle) * row["X"]
                + math.cos(self.x_axis_angle) * row["Y"]
            )
            rotated_coordinates.at[index, "X"] = rotated_x
            rotated_coordinates.at[index, "Y"] = rotated_y

        return rotated_coordinates

    def get_x_axis_angle(self) -> float:
        first_x_coordinate = self.splined_coordinates["X"][0]
        first_y_coordinate = self.splined_coordinates["Y"][0]
        second_x_coordinate = self.splined_coordinates["X"][1]
        second_y_coordinate = self.splined_coordinates["Y"][1]

        first_move_vector = np.array(
            [
                second_x_coordinate - first_x_coordinate,
                second_y_coordinate - first_y_coordinate,
            ]
        )
        x_axis_vector = np.array([1, 0])

        dot = first_move_vector @ x_axis_vector
        det = (
            first_move_vector[0] * x_axis_vector[1]
            - first_move_vector[1] * x_axis_vector[0]
        )

        # dot = x1*x2 + y1*y2
        # det = x1*y2 - y1*x2
        x_axis_angle = math.atan2(det, dot)

        return x_axis_angle

    def get_sum_of_delta_angles(self) -> float:
        """Returns the sum of the angles between each segment in the trajectory.

        Args:
            coordinates (pd.DataFrame): A dataframe containing the coordinates
                                        of the vehicle trajectory.
        """
        delta_angles = self.get_delta_angles(self.splined_coordinates)
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

    def visualize_raw_coordinates_without_scenario(
        self, coordinates, title="Trajectory Visualization", padding=10
    ):
        """
        Visualize the trajectory specified by coordinates, scaling to fit the trajectory size.

        Args:
        - coordinates: A DataFrame with 'X' and 'Y' columns, or an array-like structure representing trajectory points.
        - title: The title of the plot.
        - padding: Extra space around the trajectory bounds.
        """

        fig, ax = plt.subplots(
            figsize=(10, 10)
        )  # Create a figure and a set of subplots

        # Scale the normalized trajectory to fit the figure

        # Plot the trajectory
        ax.plot(
            coordinates["X"],
            coordinates["Y"],
            "ro-",
            markersize=5,
            linewidth=2,
        )  # 'ro-' creates a red line with circle markers

        # Set aspect of the plot to be equal
        ax.set_aspect("equal")

        # Set title of the plot
        # ax.set_title(title)

        # Remove axes for a cleaner look since there's no map
        # ax.axis("off")

        return plt

    def get_direction_of_vehicle(self):
        """Sorts a given trajectory into one of the
        following buckets:

        - Straight
        - Straight-Left
        - Straight-Right
        - Left
        - Right
        - Left-U-Turn
        - Right-U-Turn
        - Stationary

        These buckets are inspired by the paper:
        "MotionLM: Multi-Agent Motion Forecasting as Language Modeling"

        Returns:
            str: Label of the bucket to which the vehicle trajectory was assigned.
        """
        total_delta_angle = self.get_sum_of_delta_angles()
        direction = ""
        bucket = ""

        if total_delta_angle < 0:
            direction = "Right"
        elif total_delta_angle > 0:
            direction = "Left"
        else:
            direction = "Straight"

        absolute_total_delta_angle = abs(total_delta_angle)

        if self.relative_displacement < 0.05:
            bucket = "Stationary"
            return bucket
        elif absolute_total_delta_angle < 15 and absolute_total_delta_angle > -15:
            bucket = "Straight"
            return bucket
        elif absolute_total_delta_angle <= 40 and direction == "Right":
            bucket = "Straight-Right"
            return bucket
        elif absolute_total_delta_angle <= 40 and direction == "Left":
            bucket = "Straight-Left"
            return bucket
        elif (
            absolute_total_delta_angle > 40
            and absolute_total_delta_angle <= 130
            and direction == "Right"
        ):
            bucket = "Right"
            return bucket
        elif (
            absolute_total_delta_angle > 40
            and absolute_total_delta_angle <= 130
            and direction == "Left"
        ):
            bucket = "Left"
            return bucket
        elif (
            absolute_total_delta_angle > 130
            and direction == "Right"
            and self.relative_displacement >= 0.10
        ):
            bucket = "Right"
            return bucket
        elif (
            absolute_total_delta_angle > 130
            and direction == "Left"
            and self.relative_displacement >= 0.10
        ):
            bucket = "Left"
            return bucket
        elif absolute_total_delta_angle > 130 and direction == "Right":
            bucket = "Right-U-Turn"
            return bucket
        elif absolute_total_delta_angle > 130 and direction == "Left":
            bucket = "Left-U-Turn"
            return bucket
        else:
            bucket = "Straight"
            return bucket

    def get_relative_displacement(self):
        total_displacement = self.get_total_displacement()
        _, _, width = self.scenario.get_viewport()

        relative_displacement = total_displacement / width
        return relative_displacement

    def get_total_displacement(self):
        """Calculates the total displacement of the vehicle with the given coordinates.

        Returns:
            str: Total displacement of the vehicle.
        """
        starting_point = (
            self.splined_coordinates["X"][0],
            self.splined_coordinates["Y"][0],
        )
        end_point = (
            self.splined_coordinates["X"].iloc[-1],
            self.splined_coordinates["Y"].iloc[-1],
        )

        displacement_vector = (
            end_point[0] - starting_point[0],
            end_point[1] - starting_point[1],
        )

        # Calculuating the magnitude of the displacement vector and returning it
        return math.sqrt(displacement_vector[0] ** 2 + displacement_vector[1] ** 2)

    @staticmethod
    def get_rotated_ego_coordinates_from_coordinates(
        coordinates: pd.DataFrame,
    ) -> Tuple[pd.DataFrame, float, float, float]:
        """Given a set of coordinates this method calculates the rotated coordinates of the trajectory and the x axis angle.

        Args:
            coordinates (pd.DataFrame): The coordinates of a trajectory.

        Returns:
            pd.DataFrame, float, float, float: rotated_coordinates, x_axix_angle, starting_point_x, starting_point_y
        """

        # Getting x axis angle
        first_x_coordinate = coordinates["X"][0]
        first_y_coordinate = coordinates["Y"][0]
        second_x_coordinate = coordinates["X"][1]
        second_y_coordinate = coordinates["Y"][1]

        first_move_vector = np.array(
            [
                second_x_coordinate - first_x_coordinate,
                second_y_coordinate - first_y_coordinate,
            ]
        )
        x_axis_vector = np.array([1, 0])

        dot = first_move_vector @ x_axis_vector
        det = (
            first_move_vector[0] * x_axis_vector[1]
            - first_move_vector[1] * x_axis_vector[0]
        )

        # dot = x1*x2 + y1*y2
        # det = x1*y2 - y1*x2
        x_axis_angle = -math.atan2(det, dot)

        # Getting ego coordinates

        first_x_coordinate = coordinates["X"][0]
        first_y_coordinate = coordinates["Y"][0]

        ego_coordinates = coordinates.copy()
        ego_coordinates["X"] = ego_coordinates["X"] - first_x_coordinate
        ego_coordinates["Y"] = ego_coordinates["Y"] - first_y_coordinate

        # Getting rotated coordinates
        rotated_coordinates = ego_coordinates.copy()
        for index, row in ego_coordinates.iterrows():
            rotated_x = (
                math.cos(x_axis_angle) * row["X"] - math.sin(x_axis_angle) * row["Y"]
            )
            rotated_y = (
                math.sin(x_axis_angle) * row["X"] + math.cos(x_axis_angle) * row["Y"]
            )
            rotated_coordinates.at[index, "X"] = rotated_x
            rotated_coordinates.at[index, "Y"] = rotated_y

        return rotated_coordinates, x_axis_angle, first_x_coordinate, first_y_coordinate

    @staticmethod
    def get_coordinates_from_rotated_ego_coordinates(
        rotated_ego_coordinates: pd.DataFrame,
        rotated_angle,
        original_starting_x,
        original_starting_y,
    ) -> pd.DataFrame:
        # Getting rotated coordinates
        unrotated_coordinates = rotated_ego_coordinates.copy()
        for index, row in rotated_ego_coordinates.iterrows():
            rotated_x = (
                math.cos(-rotated_angle) * row["X"]
                - math.sin(-rotated_angle) * row["Y"]
            )
            rotated_y = (
                math.sin(-rotated_angle) * row["X"]
                + math.cos(-rotated_angle) * row["Y"]
            )
            unrotated_coordinates.at[index, "X"] = rotated_x
            unrotated_coordinates.at[index, "Y"] = rotated_y

        non_ego_coordinates = unrotated_coordinates.copy()
        non_ego_coordinates["X"] = non_ego_coordinates["X"] + original_starting_x
        non_ego_coordinates["Y"] = non_ego_coordinates["Y"] + original_starting_y

        return non_ego_coordinates
