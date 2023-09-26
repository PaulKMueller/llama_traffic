import numpy as np
import tensorflow as tf
import pandas as pd

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
                        specific_id=None,):
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
        specific_id=None
    ):
    """TODO: More accurate description
    Visualizes all agent predicted trajectories in a serie of images.

    Args:
        decoded_example: Dictionary containing agent info about all modeled agents.
        size_pixels: The size in pixels of the output image.

    Returns:
        T of [H, W, 3] uint8 np.arrays of the drawn matplotlib's figure canvas.
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


    return output_df