import numpy as np
import pandas as pd
import tensorflow as tf
from scipy import interpolate


class Trajectory:
    def __init__(self, decoded_example, specific_id):
        self.coordinates = self.get_coordinates(decoded_example, specific_id)
        self.splined_coordinates = self.get_spline_for_coordinates(self.coordinates)


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
            if np.sqrt((x[i] - x[i-1])**2 + (y[i] - y[i-1])**2) > threshold:
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
            tck, u = interpolate.splprep([filtered_x, filtered_y], s=120)

        
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