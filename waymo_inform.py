import numpy as np
import tensorflow as tf

def visualize_one_step(states,
                        mask,
                        roadgraph,
                        title,
                        center_y,
                        center_x,
                        width,
                        color_map,
                        with_ids,
                        agent_ids=None,
                        specific_id=None,
                        size_pixels=1000):
    """Generate visualization for a single step."""

    # Plot roadgraph.
    rg_pts = roadgraph[:, :2].T


    # If a specific ID is provided, filter the states,
    # masks, and colors to only include that ID.

    if specific_id is not None:
        n = 128  # For example, an array of size 5
        mask = np.full(n, False)
        index_of_id = np.where(agent_ids == float(specific_id))
        mask[index_of_id] = True

    masked_x = states[:, 0][mask]
    masked_y = states[:, 1][mask]
    colors = color_map[mask]

    # Plot agent current position.
    ax.scatter(
        masked_x,
        masked_y,
        marker='o',
        linewidths=3,
        color=colors,
    )

    print(f"X: {masked_x}\nY: {masked_y}")

    if with_ids:

        for x, y, agent_id in zip(
            # Iterate through the masked agent IDs
            masked_x, masked_y, agent_ids[mask]):
            # Plot the ID
            ax.text(x,
                    y,
                    str(agent_id),
                    color='black',
                    fontsize=20,
                    ha='center',
                    va='center')

    # Title.
    ax.set_title(title)

    # Set axes.  Should be at least 10m on a side and cover 160% of agents.
    size = max(10, width * 1.0)
    ax.axis([
        -size / 2 + center_x, size / 2 + center_x, -size / 2 + center_y,
        size / 2 + center_y
    ])
    ax.set_aspect('equal')

    image = fig_canvas_image(fig)
    plt.close(fig)
    return image


def visualize_all_agents_smooth(
        decoded_example,
        with_ids=False,
        specific_id=None,
        size_pixels=1000,
    ):
    """Visualizes all agent predicted trajectories in a serie of images.

    Args:
        decoded_example: Dictionary containing agent info about all modeled agents.
        size_pixels: The size in pixels of the output image.

    Returns:
        T of [H, W, 3] uint8 np.arrays of the drawn matplotlib's figure canvas.
    """

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

    # [num_points, 3] float32.
    roadgraph_xyz = decoded_example['roadgraph_samples/xyz'].numpy()

    num_agents, num_past_steps, _ = past_states.shape
    num_future_steps = future_states.shape[1]

    color_map = get_colormap(num_agents)

    # [num_agens, num_past_steps + 1 + num_future_steps, depth] float32.
    all_states = np.concatenate([past_states, current_states, future_states], 1)

    # [num_agens, num_past_steps + 1 + num_future_steps] float32.
    all_states_mask = np.concatenate(
        [past_states_mask, current_states_mask, future_states_mask], 1)

    center_y, center_x, width = get_viewport(all_states, all_states_mask)

    images = []

    # Generate images from past time steps.
    for i, (s, m) in enumerate(
        zip(
            np.split(past_states, num_past_steps, 1),
            np.split(past_states_mask, num_past_steps, 1))):
        im = visualize_one_step(s[:, 0], m[:, 0], roadgraph_xyz,
                                'past: %d' % (num_past_steps - i), center_y,
                                center_x, width, color_map, with_ids,
                                agent_ids, specific_id, size_pixels)
        images.append(im)

    # Generate one image for the current time step.
    s = current_states
    m = current_states_mask

    im = visualize_one_step(s[:, 0], m[:, 0], roadgraph_xyz, 'current', center_y,
                            center_x, width, color_map, with_ids,
                            agent_ids, specific_id, size_pixels)
    images.append(im)

    # Generate images from future time steps.
    for i, (s, m) in enumerate(
        zip(
            np.split(future_states, num_future_steps, 1),
            np.split(future_states_mask, num_future_steps, 1))):
        im = visualize_one_step(s[:, 0], m[:, 0], roadgraph_xyz,
                                'future: %d' % (i + 1), center_y, center_x, width,
                                color_map, with_ids, agent_ids, specific_id, size_pixels)
        images.append(im)

    return images