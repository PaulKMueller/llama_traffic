import time
import uuid

from matplotlib import cm
import matplotlib.animation as animation
import matplotlib.pyplot as plt

import numpy as np
from IPython.display import HTML
import itertools
import tensorflow as tf

def create_figure_and_axes(size_pixels):
    """Initializes a unique figure and axes for plotting."""
    fig, ax = plt.subplots(1, 1, num=uuid.uuid4())

    # Sets output image to pixel resolution.
    dpi = 100
    size_inches = size_pixels / dpi
    fig.set_size_inches([size_inches, size_inches])
    fig.set_dpi(dpi)
    fig.set_facecolor('white')
    ax.set_facecolor('white')
    ax.xaxis.label.set_color('black')
    ax.tick_params(axis='x', colors='black')
    ax.yaxis.label.set_color('black')
    ax.tick_params(axis='y', colors='black')
    fig.set_tight_layout(True)
    ax.grid(False)
    return fig, ax


def fig_canvas_image(fig):
    """Returns a [H, W, 3] uint8 np.array
    image from fig.canvas.tostring_rgb()."""
    # Just enough margin in the figure to display xticks and yticks.
    fig.subplots_adjust(
        left=0.08, bottom=0.08, right=0.98, top=0.98, wspace=0.0, hspace=0.0)
    fig.canvas.draw()
    data = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
    return data.reshape(fig.canvas.get_width_height()[::-1] + (3,))


def get_colormap(num_agents):
    """Compute a color map array of shape [num_agents, 4]."""
    colors = cm.get_cmap('jet', num_agents)
    colors = colors(range(num_agents))
    np.random.shuffle(colors)
    return colors


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

    # Create figure and axes.
    fig, ax = create_figure_and_axes(size_pixels=size_pixels)

    # Plot roadgraph.
    rg_pts = roadgraph[:, :2].T
    ax.plot(rg_pts[0, :], rg_pts[1, :], 'k.', alpha=1, ms=2)

    # If a specific ID is provided, filter the states,
    # masks, and colors to only include that ID.

    if specific_id is not None:
        n = 128
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


def create_animation(images):
    """ Creates a Matplotlib animation of the given images.

    Args:
        images: A list of numpy arrays representing the images.

    Returns:
        A matplotlib.animation.Animation.

    Usage:
        anim = create_animation(images)
        anim.save('/tmp/animation.avi')
        HTML(anim.to_html5_video())
    """

    plt.ioff()
    fig, ax = plt.subplots()
    dpi = 100
    size_inches = 1000 / dpi
    fig.set_size_inches([size_inches, size_inches])
    plt.ion()

    def animate_func(i):
        ax.imshow(images[i])
        ax.set_xticks([])
        ax.set_yticks([])
        ax.grid('off')

    anim = animation.FuncAnimation(
        fig, animate_func, frames=len(images) // 2, interval=100)
    plt.close(fig)
    return anim




def visualize_trajectory_one_step(ax, 
                                states,
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
                                size_pixels=1000,
                                alpha=1.0):
    """Generate visualization for a single step."""
    
    # Plot roadgraph.
    rg_pts = roadgraph[:, :2].T
    # Reduced alpha for roadgraph to keep it subtle
    ax.plot(rg_pts[0, :], rg_pts[1, :], 'k.', alpha=0.5, ms=2) 
    
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
        alpha=alpha,  # You can adjust the alpha depending on your preference.
    )
    
    if with_ids:
        for x, y, agent_id in zip(masked_x, masked_y, agent_ids[mask]):
            ax.text(x, y, str(agent_id),
                    color='black',
                    fontsize=20,
                    ha='center',
                    va='center')
    
    # Set axes.
    size = max(10, width * 1.0)
    ax.axis([
        -size / 2 + center_x, size / 2 + center_x, -size / 2 + center_y,
        size / 2 + center_y
    ])
    ax.set_aspect('equal')


def visualize_trajectory(decoded_example, with_ids=False,
                         specific_id=None, size_pixels=1000):
    """Visualizes all agent predicted trajectories in a single image.
    
    Args:
        decoded_example: Dictionary containing agent info about all modeled agents.
        size_pixels: The size in pixels of the output image.
    """
    
    agent_ids = decoded_example['state/id'].numpy()

    past_states = tf.stack([decoded_example['state/past/x'],
                            decoded_example['state/past/y']], -1).numpy()
    past_states_mask = decoded_example['state/past/valid'].numpy() > 0.0
    
    current_states = tf.stack([decoded_example['state/current/x'],
                               decoded_example['state/current/y']], -1).numpy()
    current_states_mask = decoded_example['state/current/valid'].numpy() > 0.0
    
    future_states = tf.stack([decoded_example['state/future/x'],
                              decoded_example['state/future/y']], -1).numpy()
    future_states_mask = decoded_example['state/future/valid'].numpy() > 0.0
    
    roadgraph_xyz = decoded_example['roadgraph_samples/xyz'].numpy()
    
    color_map = get_colormap(agent_ids.shape[0])
    
    all_states = np.concatenate([past_states, current_states, future_states], 1)
    all_states_mask = np.concatenate([past_states_mask,
                                      current_states_mask,
                                      future_states_mask], 1)

    center_y, center_x, width = get_viewport(all_states, all_states_mask)
    
    # Creating one figure and axis to visualize all steps on the same plot
    _, ax = create_figure_and_axes(size_pixels=size_pixels)
    
    # Visualize past states
    for s, m in zip(np.split(past_states, past_states.shape[1], 1), np.split(
        past_states_mask,
        past_states_mask.shape[1],
        1)):
        visualize_trajectory_one_step(ax, s[:, 0], m[:, 0], roadgraph_xyz,
                                      'past', center_y, center_x, width,
                                      color_map, with_ids, agent_ids,
                                      specific_id, size_pixels, alpha=0.5)
        
    # Visualize current state
    visualize_trajectory_one_step(ax, current_states[:, 0],
                                  current_states_mask[:, 0], roadgraph_xyz,
                                  'current', center_y, center_x, width, color_map,
                                  with_ids, agent_ids, specific_id,
                                  size_pixels, alpha=1.0)
    
    # Visualize future states
    for s, m in zip(np.split(future_states, future_states.shape[1], 1), np.split(
        future_states_mask,
        future_states_mask.shape[1],
        1)):
        visualize_trajectory_one_step(ax, s[:, 0], m[:, 0], roadgraph_xyz,
                                      'future', center_y, center_x, width,
                                      color_map, with_ids, agent_ids,
                                      specific_id, size_pixels, alpha=0.7)
    

    return plt

