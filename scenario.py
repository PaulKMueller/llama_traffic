import tensorflow as tf
import numpy as np
import uuid

from matplotlib import cm
import matplotlib.pyplot as plt
import matplotlib.animation as animation


class Scenario:
    def __init__(self, scenario_path):
        self.path = scenario_path
        self.data = self.init_waymo()
        self.name = self.path.split("/")[-1]

    def init_waymo(self):
        # If you use a custom conversion from Scenario to tf.Example, set the correct
        # number of map samples here. HINT: Was 30000 before
        num_map_samples = 30000

        # Example field definition
        roadgraph_features = {
            "roadgraph_samples/dir": tf.io.FixedLenFeature(
                [num_map_samples, 3], tf.float32, default_value=None
            ),
            "roadgraph_samples/id": tf.io.FixedLenFeature(
                [num_map_samples, 1], tf.int64, default_value=None
            ),
            "roadgraph_samples/type": tf.io.FixedLenFeature(
                [num_map_samples, 1], tf.int64, default_value=None
            ),
            "roadgraph_samples/valid": tf.io.FixedLenFeature(
                [num_map_samples, 1], tf.int64, default_value=None
            ),
            "roadgraph_samples/xyz": tf.io.FixedLenFeature(
                [num_map_samples, 3], tf.float32, default_value=None
            ),
        }
        # Features of other agents.
        state_features = {
            "state/id": tf.io.FixedLenFeature([128], tf.float32, default_value=None),
            "state/type": tf.io.FixedLenFeature([128], tf.float32, default_value=None),
            "state/is_sdc": tf.io.FixedLenFeature([128], tf.int64, default_value=None),
            "state/tracks_to_predict": tf.io.FixedLenFeature(
                [128], tf.int64, default_value=None
            ),
            "state/current/bbox_yaw": tf.io.FixedLenFeature(
                [128, 1], tf.float32, default_value=None
            ),
            "state/current/height": tf.io.FixedLenFeature(
                [128, 1], tf.float32, default_value=None
            ),
            "state/current/length": tf.io.FixedLenFeature(
                [128, 1], tf.float32, default_value=None
            ),
            "state/current/timestamp_micros": tf.io.FixedLenFeature(
                [128, 1], tf.int64, default_value=None
            ),
            "state/current/valid": tf.io.FixedLenFeature(
                [128, 1], tf.int64, default_value=None
            ),
            "state/current/vel_yaw": tf.io.FixedLenFeature(
                [128, 1], tf.float32, default_value=None
            ),
            "state/current/velocity_x": tf.io.FixedLenFeature(
                [128, 1], tf.float32, default_value=None
            ),
            "state/current/velocity_y": tf.io.FixedLenFeature(
                [128, 1], tf.float32, default_value=None
            ),
            "state/current/width": tf.io.FixedLenFeature(
                [128, 1], tf.float32, default_value=None
            ),
            "state/current/x": tf.io.FixedLenFeature(
                [128, 1], tf.float32, default_value=None
            ),
            "state/current/y": tf.io.FixedLenFeature(
                [128, 1], tf.float32, default_value=None
            ),
            "state/current/z": tf.io.FixedLenFeature(
                [128, 1], tf.float32, default_value=None
            ),
            "state/future/bbox_yaw": tf.io.FixedLenFeature(
                [128, 80], tf.float32, default_value=None
            ),
            "state/future/height": tf.io.FixedLenFeature(
                [128, 80], tf.float32, default_value=None
            ),
            "state/future/length": tf.io.FixedLenFeature(
                [128, 80], tf.float32, default_value=None
            ),
            "state/future/timestamp_micros": tf.io.FixedLenFeature(
                [128, 80], tf.int64, default_value=None
            ),
            "state/future/valid": tf.io.FixedLenFeature(
                [128, 80], tf.int64, default_value=None
            ),
            "state/future/vel_yaw": tf.io.FixedLenFeature(
                [128, 80], tf.float32, default_value=None
            ),
            "state/future/velocity_x": tf.io.FixedLenFeature(
                [128, 80], tf.float32, default_value=None
            ),
            "state/future/velocity_y": tf.io.FixedLenFeature(
                [128, 80], tf.float32, default_value=None
            ),
            "state/future/width": tf.io.FixedLenFeature(
                [128, 80], tf.float32, default_value=None
            ),
            "state/future/x": tf.io.FixedLenFeature(
                [128, 80], tf.float32, default_value=None
            ),
            "state/future/y": tf.io.FixedLenFeature(
                [128, 80], tf.float32, default_value=None
            ),
            "state/future/z": tf.io.FixedLenFeature(
                [128, 80], tf.float32, default_value=None
            ),
            "state/past/bbox_yaw": tf.io.FixedLenFeature(
                [128, 10], tf.float32, default_value=None
            ),
            "state/past/height": tf.io.FixedLenFeature(
                [128, 10], tf.float32, default_value=None
            ),
            "state/past/length": tf.io.FixedLenFeature(
                [128, 10], tf.float32, default_value=None
            ),
            "state/past/timestamp_micros": tf.io.FixedLenFeature(
                [128, 10], tf.int64, default_value=None
            ),
            "state/past/valid": tf.io.FixedLenFeature(
                [128, 10], tf.int64, default_value=None
            ),
            "state/past/vel_yaw": tf.io.FixedLenFeature(
                [128, 10], tf.float32, default_value=None
            ),
            "state/past/velocity_x": tf.io.FixedLenFeature(
                [128, 10], tf.float32, default_value=None
            ),
            "state/past/velocity_y": tf.io.FixedLenFeature(
                [128, 10], tf.float32, default_value=None
            ),
            "state/past/width": tf.io.FixedLenFeature(
                [128, 10], tf.float32, default_value=None
            ),
            "state/past/x": tf.io.FixedLenFeature(
                [128, 10], tf.float32, default_value=None
            ),
            "state/past/y": tf.io.FixedLenFeature(
                [128, 10], tf.float32, default_value=None
            ),
            "state/past/z": tf.io.FixedLenFeature(
                [128, 10], tf.float32, default_value=None
            ),
        }

        traffic_light_features = {
            "traffic_light_state/current/state": tf.io.FixedLenFeature(
                [1, 16], tf.int64, default_value=None
            ),
            "traffic_light_state/current/valid": tf.io.FixedLenFeature(
                [1, 16], tf.int64, default_value=None
            ),
            "traffic_light_state/current/x": tf.io.FixedLenFeature(
                [1, 16], tf.float32, default_value=None
            ),
            "traffic_light_state/current/y": tf.io.FixedLenFeature(
                [1, 16], tf.float32, default_value=None
            ),
            "traffic_light_state/current/z": tf.io.FixedLenFeature(
                [1, 16], tf.float32, default_value=None
            ),
            "traffic_light_state/past/state": tf.io.FixedLenFeature(
                [10, 16], tf.int64, default_value=None
            ),
            "traffic_light_state/past/valid": tf.io.FixedLenFeature(
                [10, 16], tf.int64, default_value=None
            ),
            "traffic_light_state/past/x": tf.io.FixedLenFeature(
                [10, 16], tf.float32, default_value=None
            ),
            "traffic_light_state/past/y": tf.io.FixedLenFeature(
                [10, 16], tf.float32, default_value=None
            ),
            "traffic_light_state/past/z": tf.io.FixedLenFeature(
                [10, 16], tf.float32, default_value=None
            ),
        }

        features_description = {}
        features_description.update(roadgraph_features)
        features_description.update(state_features)
        features_description.update(traffic_light_features)
        dataset = tf.data.TFRecordDataset(self.path, compression_type="")
        data = next(dataset.as_numpy_iterator())
        parsed = tf.io.parse_single_example(data, features_description)
        return parsed

    def visualize_all_agents_smooth(
        self,
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

        agent_ids = self.data["state/id"].numpy()

        # [num_agents, num_past_steps, 2] float32.
        past_states = tf.stack(
            [self.data["state/past/x"], self.data["state/past/y"]], -1
        ).numpy()
        past_states_mask = self.data["state/past/valid"].numpy() > 0.0

        # [num_agents, 1, 2] float32.
        current_states = tf.stack(
            [self.data["state/current/x"], self.data["state/current/y"]], -1
        ).numpy()
        current_states_mask = self.data["state/current/valid"].numpy() > 0.0

        # [num_agents, num_future_steps, 2] float32.
        future_states = tf.stack(
            [self.data["state/future/x"], self.data["state/future/y"]], -1
        ).numpy()
        future_states_mask = self.data["state/future/valid"].numpy() > 0.0

        # [num_points, 3] float32.
        roadgraph_xyz = self.data["roadgraph_samples/xyz"].numpy()

        num_agents, num_past_steps, _ = past_states.shape
        num_future_steps = future_states.shape[1]

        color_map = self.get_colormap(num_agents)

        # [num_agens, num_past_steps + 1 + num_future_steps, depth] float32.
        all_states = np.concatenate([past_states, current_states, future_states], 1)

        # [num_agens, num_past_steps + 1 + num_future_steps] float32.
        all_states_mask = np.concatenate(
            [past_states_mask, current_states_mask, future_states_mask], 1
        )

        center_y, center_x, width = self.get_viewport()

        images = []

        # Generate images from past time steps.
        for i, (s, m) in enumerate(
            zip(
                np.split(past_states, num_past_steps, 1),
                np.split(past_states_mask, num_past_steps, 1),
            )
        ):
            im = self.visualize_one_step(
                s[:, 0],
                m[:, 0],
                roadgraph_xyz,
                "past: %d" % (num_past_steps - i),
                center_y,
                center_x,
                width,
                color_map,
                with_ids,
                agent_ids,
                specific_id,
                size_pixels,
            )
            images.append(im)

        # Generate one image for the current time step.
        s = current_states
        m = current_states_mask

        im = self.visualize_one_step(
            s[:, 0],
            m[:, 0],
            roadgraph_xyz,
            "current",
            center_y,
            center_x,
            width,
            color_map,
            with_ids,
            agent_ids,
            specific_id,
            size_pixels,
        )
        images.append(im)

        # Generate images from future time steps.
        for i, (s, m) in enumerate(
            zip(
                np.split(future_states, num_future_steps, 1),
                np.split(future_states_mask, num_future_steps, 1),
            )
        ):
            im = self.visualize_one_step(
                s[:, 0],
                m[:, 0],
                roadgraph_xyz,
                "future: %d" % (i + 1),
                center_y,
                center_x,
                width,
                color_map,
                with_ids,
                agent_ids,
                specific_id,
                size_pixels,
            )
            images.append(im)

        return images

    def visualize_one_step(
        self,
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
    ):
        """Generate visualization for a single step."""

        # Create figure and axes.
        fig, ax = self.create_figure_and_axes(size_pixels)

        # Plot roadgraph.
        rg_pts = roadgraph[:, :2].T
        ax.plot(rg_pts[0, :], rg_pts[1, :], "k.", alpha=1, ms=2)

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
            marker="o",
            linewidths=3,
            color=colors,
        )

        if with_ids:
            for x, y, agent_id in zip(
                # Iterate through the masked agent IDs
                masked_x,
                masked_y,
                agent_ids[mask],
            ):
                # Plot the ID
                ax.text(
                    x,
                    y,
                    str(agent_id),
                    color="black",
                    fontsize=20,
                    ha="center",
                    va="center",
                )

        # Title.
        ax.set_title(title)

        # Set axes.  Should be at least 10m on a side and cover 160% of agents.
        size = max(10, width * 1.0)
        ax.axis(
            [
                -size / 2 + center_x,
                size / 2 + center_x,
                -size / 2 + center_y,
                size / 2 + center_y,
            ]
        )
        ax.set_aspect("equal")

        image = self.fig_canvas_image(fig)
        plt.close(fig)
        return image

    @staticmethod
    def get_colormap(num_agents):
        """Compute a color map array of shape [num_agents, 4]."""
        colors = cm.get_cmap("jet", num_agents)
        colors = colors(range(num_agents))
        np.random.shuffle(colors)
        return colors

    @staticmethod
    def fig_canvas_image(fig):
        """Returns a [H, W, 3] uint8 np.array
        image from fig.canvas.tostring_rgb()."""
        # Just enough margin in the figure to display xticks and yticks.
        fig.subplots_adjust(
            left=0.08, bottom=0.08, right=0.98, top=0.98, wspace=0.0, hspace=0.0
        )
        fig.canvas.draw()
        data = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
        return data.reshape(fig.canvas.get_width_height()[::-1] + (3,))

    def get_viewport(self):
        """Gets the region containing the data.

        Returns:
            center_y: float. y coordinate for center of data.
            center_x: float. x coordinate for center of data.
            width: float. Width of data.
        """

        # [num_agents, num_past_steps, 2] float32.
        past_states = tf.stack(
            [self.data["state/past/x"], self.data["state/past/y"]], -1
        ).numpy()
        past_states_mask = self.data["state/past/valid"].numpy() > 0.0

        # [num_agents, 1, 2] float32.
        current_states = tf.stack(
            [self.data["state/current/x"], self.data["state/current/y"]], -1
        ).numpy()
        current_states_mask = self.data["state/current/valid"].numpy() > 0.0

        # [num_agents, num_future_steps, 2] float32.
        future_states = tf.stack(
            [self.data["state/future/x"], self.data["state/future/y"]], -1
        ).numpy()
        future_states_mask = self.data["state/future/valid"].numpy() > 0.0

        # [num_agens, num_past_steps + 1 + num_future_steps, depth] float32.
        all_states = np.concatenate([past_states, current_states, future_states], 1)

        # [num_agens, num_past_steps + 1 + num_future_steps] float32.
        all_states_mask = np.concatenate(
            [past_states_mask, current_states_mask, future_states_mask], 1
        )

        valid_states = all_states[all_states_mask]
        all_y = valid_states[..., 1]
        all_x = valid_states[..., 0]

        center_y = (np.max(all_y) + np.min(all_y)) / 2
        center_x = (np.max(all_x) + np.min(all_x)) / 2

        range_y = np.ptp(all_y)
        range_x = np.ptp(all_x)

        width = max(range_y, range_x)

        return center_y, center_x, width

    def get_animation(self, with_ids=False):
        images = self.visualize_all_agents_smooth(with_ids=with_ids)
        print(len(images))
        anim = self.create_animation(images[::5])
        return anim

    @staticmethod
    def create_animation(images):
        """Creates a Matplotlib animation of the given images.

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
            ax.grid("off")

        anim = animation.FuncAnimation(
            fig, animate_func, frames=len(images) // 2, interval=100
        )
        plt.close(fig)
        return anim

    def create_figure_and_axes(self, size_pixels):
        """Initializes a unique figure and axes for plotting."""
        fig, ax = plt.subplots(1, 1, num=uuid.uuid4())

        # Sets output image to pixel resolution.
        dpi = 100
        size_inches = size_pixels / dpi
        fig.set_size_inches([size_inches, size_inches])
        fig.set_dpi(dpi)
        fig.set_facecolor("white")
        ax.set_facecolor("white")
        ax.xaxis.label.set_color("black")
        ax.tick_params(axis="x", colors="black")
        ax.yaxis.label.set_color("black")
        ax.tick_params(axis="y", colors="black")
        fig.set_tight_layout(True)
        ax.grid(False)
        return fig, ax

    def visualize_coordinates(self, coordinates, size_pixels=1000):
        plot = self.visualize_map()
        print(type(coordinates))
        # Plot coordinates on map
        for _, coordinate in coordinates.iterrows():
            plot.plot(coordinate["X"], coordinate["Y"], "ro", markersize=5)

        return plot

    def visualize_map(self, with_ids=False, specific_id=None, size_pixels=1000):
        """Visualizes all agent predicted trajectories in a single image.

        Args:
            decoded_example: Dictionary containing agent info about all modeled agents.
            size_pixels: The size in pixels of the output image.
        """

        agent_ids = self.data["state/id"].numpy()

        past_states = tf.stack(
            [self.data["state/past/x"], self.data["state/past/y"]], -1
        ).numpy()
        past_states_mask = self.data["state/past/valid"].numpy() > 0.0

        current_states = tf.stack(
            [self.data["state/current/x"], self.data["state/current/y"]], -1
        ).numpy()
        current_states_mask = self.data["state/current/valid"].numpy() > 0.0

        future_states = tf.stack(
            [self.data["state/future/x"], self.data["state/future/y"]], -1
        ).numpy()
        future_states_mask = self.data["state/future/valid"].numpy() > 0.0

        roadgraph_xyz = self.data["roadgraph_samples/xyz"].numpy()

        color_map = self.get_colormap(agent_ids.shape[0])

        center_y, center_x, width = self.get_viewport()

        # Creating one figure and axis to visualize all steps on the same plot
        _, ax = self.create_figure_and_axes(size_pixels=size_pixels)

        # Visualize current state
        self.visualize_map_one_step(
            ax=ax,
            states=current_states[:, 0],
            mask=current_states_mask[:, 0],
            roadgraph=roadgraph_xyz,
            center_y=center_y,
            center_x=center_x,
            width=width,
            color_map=color_map,
            with_ids=with_ids,
            agent_ids=agent_ids,
            alpha=1.0,
        )

        return plt

    @staticmethod
    def visualize_map_one_step(
        ax,
        states,
        mask,
        roadgraph,
        center_y,
        center_x,
        width,
        color_map,
        with_ids,
        agent_ids=None,
        alpha=1.0,
    ):
        """Generate visualization for a single step."""

        # Plot roadgraph.
        rg_pts = roadgraph[:, :2].T
        # Reduced alpha for roadgraph to keep it subtle
        ax.plot(rg_pts[0, :], rg_pts[1, :], "k.", alpha=0.5, ms=2)

        n = 128  # For example, an array of size 5
        mask = np.full(n, False)

        masked_x = states[:, 0][mask]
        masked_y = states[:, 1][mask]
        colors = color_map[mask]

        # Plot agent current position.
        ax.scatter(
            masked_x,
            masked_y,
            marker="o",
            linewidths=3,
            color=colors,
            alpha=alpha,  # You can adjust the alpha depending on your preference.
        )

        if with_ids:
            for x, y, agent_id in zip(masked_x, masked_y, agent_ids[mask]):
                ax.text(
                    x,
                    y,
                    str(agent_id),
                    color="black",
                    fontsize=20,
                    ha="center",
                    va="center",
                )

        # Set axes.
        size = max(10, width * 1.0)
        ax.axis(
            [
                -size / 2 + center_x,
                size / 2 + center_x,
                -size / 2 + center_y,
                size / 2 + center_y,
            ]
        )
        ax.set_aspect("equal")

    def visualize_trajectory(self, with_ids=False, specific_id=None, size_pixels=1000):
        """Visualizes all agent predicted trajectories in a single image.

        Args:
            decoded_example: Dictionary containing agent info about all modeled agents.
            size_pixels: The size in pixels of the output image.
        """

        agent_ids = self.data["state/id"].numpy()

        past_states = tf.stack(
            [self.data["state/past/x"], self.data["state/past/y"]], -1
        ).numpy()
        past_states_mask = self.data["state/past/valid"].numpy() > 0.0

        current_states = tf.stack(
            [self.data["state/current/x"], self.data["state/current/y"]], -1
        ).numpy()
        current_states_mask = self.data["state/current/valid"].numpy() > 0.0

        future_states = tf.stack(
            [self.data["state/future/x"], self.data["state/future/y"]], -1
        ).numpy()
        future_states_mask = self.data["state/future/valid"].numpy() > 0.0

        roadgraph_xyz = self.data["roadgraph_samples/xyz"].numpy()

        color_map = self.get_colormap(agent_ids.shape[0])

        all_states = np.concatenate([past_states, current_states, future_states], 1)
        all_states_mask = np.concatenate(
            [past_states_mask, current_states_mask, future_states_mask], 1
        )

        center_y, center_x, width = self.get_viewport()

        # Creating one figure and axis to visualize all steps on the same plot
        _, ax = self.create_figure_and_axes(size_pixels=size_pixels)

        # Visualize past states
        for s, m in zip(
            np.split(past_states, past_states.shape[1], 1),
            np.split(past_states_mask, past_states_mask.shape[1], 1),
        ):
            self.visualize_trajectory_one_step(
                ax,
                s[:, 0],
                m[:, 0],
                roadgraph_xyz,
                "past",
                center_y,
                center_x,
                width,
                color_map,
                with_ids,
                agent_ids,
                specific_id,
                size_pixels,
                alpha=0.5,
            )

        # Visualize current state
        self.visualize_trajectory_one_step(
            ax,
            current_states[:, 0],
            current_states_mask[:, 0],
            roadgraph_xyz,
            "current",
            center_y,
            center_x,
            width,
            color_map,
            with_ids,
            agent_ids,
            specific_id,
            size_pixels,
            alpha=1.0,
        )

        # Visualize future states
        for s, m in zip(
            np.split(future_states, future_states.shape[1], 1),
            np.split(future_states_mask, future_states_mask.shape[1], 1),
        ):
            self.visualize_trajectory_one_step(
                ax,
                s[:, 0],
                m[:, 0],
                roadgraph_xyz,
                "future",
                center_y,
                center_x,
                width,
                color_map,
                with_ids,
                agent_ids,
                specific_id,
                size_pixels,
                alpha=0.7,
            )

        return plt

    @staticmethod
    def visualize_trajectory_one_step(
        ax,
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
        alpha=1.0,
    ):
        """Generate visualization for a single step."""

        # Plot roadgraph.
        rg_pts = roadgraph[:, :2].T
        # Reduced alpha for roadgraph to keep it subtle
        ax.plot(rg_pts[0, :], rg_pts[1, :], "k.", alpha=0.5, ms=2)

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
            marker="o",
            linewidths=3,
            color=colors,
            alpha=alpha,  # You can adjust the alpha depending on your preference.
        )

        if with_ids:
            for x, y, agent_id in zip(masked_x, masked_y, agent_ids[mask]):
                ax.text(
                    x,
                    y,
                    str(agent_id),
                    color="black",
                    fontsize=20,
                    ha="center",
                    va="center",
                )

        # Set axes.
        size = max(10, width * 1.0)
        ax.axis(
            [
                -size / 2 + center_x,
                size / 2 + center_x,
                -size / 2 + center_y,
                size / 2 + center_y,
            ]
        )
        ax.set_aspect("equal")
