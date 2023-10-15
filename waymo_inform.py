import numpy as np
import tensorflow as tf
import pandas as pd
import math

def get_viewport(all_states, all_states_mask):
    """Gets the region containing the data.

    Args:
        all_states: states of agents as an array of shape [num_agents, num_steps,
        2].
        all_states_mask: binary mask of shape [num_agents, num_steps] for
        `all_states`.

    Returns:
        center_y: float. y coordinate for center of data.
        center_x: float. x coordinate for center of data.
        width: float. Width of data.
    """

    valid_states = all_states[all_states_mask]
    all_y = valid_states[..., 1]
    all_x = valid_states[..., 0]

    center_y = (np.max(all_y) + np.min(all_y)) / 2
    center_x = (np.max(all_x) + np.min(all_x)) / 2

    range_y = np.ptp(all_y)
    range_x = np.ptp(all_x)

    width = max(range_y, range_x)

    return center_y, center_x, width


def get_coordinates_one_step(states,
                        mask,
                        agent_ids=None,
                        specific_id:float=None,):
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


def get_coordinates(
        decoded_example,
        specific_id: float=None
    ):
    """Returns the coordinates of the vehicle identified by its
    specific_id and stores them as a CSV in the output folder.

    Args:
        decoded_example: Dictionary containing agent info about all modeled agents.
        specific_id: The idea for which to store the coordinates.
    """

    output_df = pd.DataFrame(columns=["X", "Y"])

    agent_ids = decoded_example['state/id'].numpy()

    # [num_agents, num_past_steps, 2] float32.
    past_states = tf.stack(
        [decoded_example['state/past/x'], decoded_example['state/past/y']],
        -1).numpy()
    past_states_mask = decoded_example['state/past/valid'].numpy() > 0.0

    # [num_agents, 1, 2] float32.
    current_states = tf.stack(
        [decoded_example['state/current/x'], decoded_example['state/current/y']],
        -1).numpy()
    current_states_mask = decoded_example['state/current/valid'].numpy() > 0.0

    # [num_agents, num_future_steps, 2] float32.
    future_states = tf.stack(
        [decoded_example['state/future/x'], decoded_example['state/future/y']],
        -1).numpy()
    future_states_mask = decoded_example['state/future/valid'].numpy() > 0.0

    _, num_past_steps, _ = past_states.shape
    num_future_steps = future_states.shape[1]

    # Generate images from past time steps.
    for _, (s, m) in enumerate(
        zip(
            np.split(past_states, num_past_steps, 1),
            np.split(past_states_mask, num_past_steps, 1))):
        coordinates_for_step = get_coordinates_one_step(s[:, 0], m[:, 0],
                                agent_ids=agent_ids, specific_id=specific_id)
        coordinates_for_step_df = pd.DataFrame([coordinates_for_step])
        output_df = pd.concat([output_df, coordinates_for_step_df], ignore_index=True)


    # Generate one image for the current time step.
    s = current_states
    m = current_states_mask

    coordinates_for_step = get_coordinates_one_step(s[:, 0], m[:, 0],
                                                    agent_ids=agent_ids,
                                                    specific_id=specific_id)
    coordinates_for_step_df = pd.DataFrame([coordinates_for_step])

    output_df = pd.concat([output_df, coordinates_for_step_df], ignore_index=True)


    # Generate images from future time steps.
    for _, (s, m) in enumerate(
        zip(
            np.split(future_states, num_future_steps, 1),
            np.split(future_states_mask, num_future_steps, 1))):
        coordinates_for_step = get_coordinates_one_step(s[:, 0],
                                                        m[:, 0],
                                                        agent_ids=agent_ids,
                                                        specific_id=specific_id)
        coordinates_for_step_df = pd.DataFrame([coordinates_for_step])
        output_df = pd.concat([output_df, coordinates_for_step_df], ignore_index=True)


    # Delete all rows where both X and Y are -1.0
    output_df = output_df[~((output_df["X"] == -1.0) & (output_df["Y"] == -1.0))]

    return output_df


def get_point_angle(point_one: pd.DataFrame, point_two: pd.DataFrame, reference_vector):
    """Calculates the angle between two points relative to a reference vector.

    Args:
        point_one (dict): The starting point with "X" and "Y" keys.
        point_two (dict): The end point with "X" and "Y" keys.
        reference_vector (tuple): The reference direction vector.
    """    
    # Calculate the direction vector for the segment
    segment_vector = (point_two["X"] - point_one["X"], point_two["Y"] - point_one["Y"])
    
    # Calculate the dot product and magnitudes of the vectors
    dot_product = segment_vector[0] * reference_vector[0] + segment_vector[1] * reference_vector[1]
    magnitude_segment = np.sqrt(segment_vector[0]**2 + segment_vector[1]**2)
    magnitude_reference = np.sqrt(reference_vector[0]**2 + reference_vector[1]**2)
    
    # Calculate the angle between the segment vector and the reference vector
    angle_radians = np.arccos(dot_product / (magnitude_segment * magnitude_reference))
    
    # Convert the angle from radians to degrees
    angle_degrees = angle_radians * (180 / math.pi)
    
    return angle_degrees


def get_total_displacement(coordinates: pd.DataFrame):
    """Calculates the total displacement of the vehicle with the given coordinates.

    Args:
        coordinates (pandas.dataframe): The coordinates of the vehicle for which
        to calculate the total displacement.
    Returns:
        str: Total displacement of the vehicle.
    """    
    starting_point = (coordinates["X"][0], coordinates["Y"][0])
    end_point = (coordinates["X"].iloc[-1], coordinates["Y"].iloc[-1])

    displacement_vector = (
        end_point[0] - starting_point[0], end_point[1] - starting_point[1])

    # Calculuating the magnitude of the displacement vector and returning it
    return math.sqrt(displacement_vector[0]**2 + displacement_vector[1]**2)


def get_relative_displacement(decoded_example, coordinates: pd.DataFrame):
    total_displacement = get_total_displacement(coordinates)
    _, _, width = get_viewport(
        get_all_states(decoded_example),
        get_all_states_mask(decoded_example))

    relative_displacement = total_displacement / width
    return relative_displacement


def get_all_states(decoded_example):

    past_states = tf.stack([decoded_example['state/past/x'],
                            decoded_example['state/past/y']], -1).numpy()
    
    current_states = tf.stack([decoded_example['state/current/x'],
                               decoded_example['state/current/y']], -1).numpy()
    
    future_states = tf.stack([decoded_example['state/future/x'],
                              decoded_example['state/future/y']], -1).numpy()
    
    all_states = np.concatenate([past_states, current_states, future_states], 1)
    return all_states


def get_all_states_mask(decoded_example):

    past_states_mask = decoded_example['state/past/valid'].numpy() > 0.0
    
    current_states_mask = decoded_example['state/current/valid'].numpy() > 0.0
    
    future_states_mask = decoded_example['state/future/valid'].numpy() > 0.0

    all_states_mask = np.concatenate([past_states_mask,
                                      current_states_mask,
                                      future_states_mask], 1)
    return all_states_mask


def get_total_trajectory_angle(coordinates: pd.DataFrame):
    """Returns the angle between the last direction vector and the first.

    Args:
        coordinates (pd.DataFrame): A dataframe containing the coordinates
                                    of the vehicle trajectory.
    """    
    # Calculate the direction vector of the first segment
    first_vector = (coordinates.iloc[1]["X"] - coordinates.iloc[0]["X"], 
                    coordinates.iloc[1]["Y"] - coordinates.iloc[0]["Y"])
    
    # Calculate the direction vector of the last segment
    last_vector = (coordinates.iloc[-1]["X"] - coordinates.iloc[-2]["X"], 
                   coordinates.iloc[-1]["Y"] - coordinates.iloc[-2]["Y"])
    
    # Compute the angle between the first and last direction vectors
    angle = get_point_angle(
        {"X": 0, "Y": 0}, {"X": last_vector[0], "Y": last_vector[1]}, first_vector)
    
    if get_gross_direction(coordinates) == "Right":
        angle = -angle
    
    return angle


def get_gross_direction(coordinates: pd.DataFrame):
    """Returns left, right, or straight depending on the direction of the trajectory.

    Args:
        coordinates (pd.DataFrame): The coordinates of the vehicle trajectory.
    """    
    starting_point = coordinates.iloc[0]
    ending_point = coordinates.iloc[-1]

    starting_X = starting_point["X"]
    starting_Y = starting_point["Y"]

    ending_X = ending_point["X"]
    ending_Y = ending_point["Y"]

    if (ending_X > starting_X and ending_Y < starting_Y):
        direction = "Right"
    elif (ending_X > starting_X and ending_Y > starting_Y):
        direction = "Left"
    elif (ending_X < starting_X and ending_Y > starting_Y):
        direction = "Right"
    elif (ending_X < starting_X and ending_Y < starting_Y):
        direction = "Left"
    else:
        direction = "Straight"

    return direction


def get_direction_of_vehicle(decoded_example, coordinates: pd.DataFrame):
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

    Args:
        coordinates (pandas.dataframe): The coordinates of the
                                        vehicle trajectory as a dataframe.

    Returns:
        str: Label of the bucket to which the vehicle trajectory was assigned.
    """

    relative_displacement = get_relative_displacement(decoded_example, coordinates)
    total_angle = get_total_trajectory_angle(coordinates)
    gross_direction = get_gross_direction(coordinates)

    if (relative_displacement < 0.01):
        direction = "Stationary"
    elif total_angle < 30 and gross_direction == "Right":
        direction = "Straight-Right"
    elif total_angle < 30 and gross_direction == "Left":
        direction = "Straight-Left"
    elif total_angle > 30 and total_angle <= 120 and gross_direction == "Right":
        direction = "Right"
    elif total_angle > 30 and total_angle <= 120 and gross_direction == "Left":
        direction = "Left"
    elif total_angle > 120 and gross_direction == "Right":
        direction = "Right-U-Turn"
    elif total_angle > 120 and gross_direction == "Left":
        direction = "Left-U-Turn"
    else:
        direction = "Straight"

    return direction


def get_vehicles_for_scenario(decoded_example):
    # All the vehicles in the scenario
    agent_ids = decoded_example['state/id'].numpy()

    # Filter out the -1 values (which are the vehicles that are not in the scene)
    filtered_ids = np.sort(agent_ids[agent_ids != -1])

    return filtered_ids


