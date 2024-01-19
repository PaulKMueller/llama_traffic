import math
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.pyplot import figure
from mpl_toolkits.mplot3d.art3d import Poly3DCollection


class NpzTrajectory:
    # print(f"Object ID: {data['object_id']}")
    # print(f"Yaw: {data['yaw']}")
    # print(f"Shift: {data['shift']}")
    # print(f"GT Marginal: {data['_gt_marginal']}")
    # print(f"GT Joint: {data['gt_joint']}")
    # print(f"Shape of GT Joint: {data['gt_joint'].shape}")
    # print(f"Future Val Marginal: {data['future_val_marginal']}")
    # print(f"Future Val Joint: {data['future_val_joint']}")
    # print(f"Scenario ID: {data['scenario_id']}")
    # print(f"Self Type: {data['self_type']}")
    # print(f"Vector Data: {data['vector_data']}")

    def __init__(self, path):
        self.path = path
        self.init_data()

    def init_data(self):
        with np.load(self.path) as data:
            self.object_id = data["object_id"]
            self.raster = data["raster"]
            self.yaw = data["yaw"]
            self.shift = data["shift"]
            self._gt_marginal = data["_gt_marginal"]
            self.gt_marginal = data["gt_marginal"]
            self.future_val_marginal = data["future_val_marginal"]
            self.gt_joint = data["gt_joint"]
            self.scenario_id = data["scenario_id"]
            self.type = data["self_type"]
            self.vector_data = data["vector_data"]
            self.coordinates = self.get_parsed_coordinates()
            self.direction = self.get_direction_of_vehicle()
            self.movement_vectors = self.get_movement_vectors()

    def get_parsed_coordinates(self):
        x = self._gt_marginal[:, 0]
        y = self._gt_marginal[:, 1]
        coordinates = pd.DataFrame({"X": x, "Y": y})
        return coordinates

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

    def get_movement_vectors(self):
        vectors = []
        x = self._gt_marginal[:, 0]
        y = self._gt_marginal[:, 1]
        for i in range(len(x) - 2):
            current_vector = [x[i] - x[i + 1], y[i] - y[i + 1]]
            vectors.append(current_vector)

        return vectors

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
            angle = self.angle_between(current_vector, previous_vector)

            direction = self.get_gross_direction_for_three_points(
                coordinates.iloc[i - 1], coordinates.iloc[i], coordinates.iloc[i + 1]
            )

            if direction == "Right":
                angle = -angle

            delta_angles.append(angle)

        return delta_angles

    def unit_vector(self, vector):
        """Returns the unit vector of the vector."""
        return vector / np.linalg.norm(vector)

    def angle_between(self, v1, v2):
        """Returns the angle in radians between vectors 'v1' and 'v2'::"""
        v1_u = self.unit_vector(v1)
        v2_u = self.unit_vector(v2)
        return math.degrees(np.arccos(np.clip(np.dot(v1_u, v2_u), -1.0, 1.0)))

    def get_sum_of_delta_angles(self) -> float:
        """Returns the sum of the angles between each segment in the trajectory.

        Args:
            coordinates (pd.DataFrame): A dataframe containing the coordinates
                                        of the vehicle trajectory.
        """
        delta_angles = self.get_delta_angles(self.coordinates)
        filtered_delta_angles = self.remove_outlier_angles(delta_angles)
        return sum(filtered_delta_angles)

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

        if self.get_relative_displacement() < 0.03:
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
            and self.get_relative_displacement() >= 0.10
        ):
            bucket = "Right"
            return bucket
        elif (
            absolute_total_delta_angle > 130
            and direction == "Left"
            and self.get_relative_displacement() >= 0.10
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

    def get_relative_displacement(self):
        total_displacement = self.get_total_displacement()
        # _, _, width = self.scenario.get_viewport()

        relative_displacement = total_displacement  # / width
        return relative_displacement

    def get_total_displacement(self):
        """Calculates the total displacement of the vehicle with the given coordinates.

        Returns:
            str: Total displacement of the vehicle.
        """
        starting_point = (
            self.coordinates["X"][0],
            self.coordinates["Y"][0],
        )
        end_point = (
            self.coordinates["X"].iloc[-1],
            self.coordinates["Y"].iloc[-1],
        )

        displacement_vector = (
            end_point[0] - starting_point[0],
            end_point[1] - starting_point[1],
        )

        # Calculuating the magnitude of the displacement vector and returning it
        return math.sqrt(displacement_vector[0] ** 2 + displacement_vector[1] ** 2)

    def plot_marginal_predictions_3d(
        self,
        vector_data,
        predictions=None,
        confidences=None,
        is_available=None,
        gt_marginal=None,
        plot_subsampling_rate=2,
        prediction_subsampling_rate=5,
        prediction_horizon=50,
        x_range=(-50, 50),
        y_range=(-50, 50),
        dpi=80,
    ):
        ax = plt.figure(figsize=(15, 15), dpi=dpi).add_subplot(projection="3d")
        V = vector_data
        X, idx = V[:, :44], V[:, 44].flatten()

        car = np.array(
            [
                (-2.25, -1, 0),  # left bottom front
                (-2.25, 1, 0),  # left bottom back
                (2.25, -1, 0),  # right bottom front
                (-2.25, -1, 1.5),  # left top front -> height
            ]
        )

        pedestrian = np.array(
            [
                (-0.3, -0.3, 0),  # left bottom front
                (-0.3, 0.3, 0),  # left bottom back
                (0.3, -0.3, 0),  # right bottom front
                (-0.3, -0.3, 2),  # left top front -> height
            ]
        )

        cyclist = np.array(
            [
                (-1, -0.3, 0),  # left bottom front
                (-1, 0.3, 0),  # left bottom back
                (1, -0.3, 0),  # right bottom front
                (-1, -0.3, 2),  # left top front -> height
            ]
        )

        for i in np.unique(idx):
            _X = X[
                (idx == i)
                & (X[:, 0] < x_range[1])
                & (X[:, 1] < y_range[1])
                & (X[:, 0] > x_range[0])
                & (X[:, 1] > y_range[0])
            ]
            if _X[:, 8].sum() > 0:
                if _X[-1, 0] == 0 and _X[-1, 1] == 0:
                    plt.plot(_X[:, 0], _X[:, 1], 0, linewidth=4, color="blue")
                    plt.plot(_X[-1, 0], _X[-1, 1], 0, "o", markersize=10, color="blue")

                bbox = self.rotate_bbox_zxis(car, _X[-1, 4])
                bbox = self.shift_cuboid(_X[-1, 0], _X[-1, 1], bbox)

                if _X[-1, 2]:  # speed to determine dynamic or static
                    self.add_cube(bbox, ax, color="tab:blue", alpha=0.5)
                else:
                    self.add_cube(bbox, ax, color="tab:grey", alpha=0.5)
            elif _X[:, 9].sum() > 0:
                if _X[-1, 0] == 0 and _X[-1, 1] == 0:
                    plt.plot(_X[:, 0], _X[:, 1], 0, linewidth=4, color="orange")
                    plt.plot(
                        _X[-1, 0], _X[-1, 1], 0, "o", markersize=10, color="orange"
                    )
                bbox = self.rotate_bbox_zxis(pedestrian, _X[-1, 4])
                bbox = self.shift_cuboid(_X[-1, 0], _X[-1, 1], bbox)
                self.add_cube(bbox, ax, color="tab:orange", alpha=0.5)
            elif _X[:, 10].sum() > 0:
                if _X[-1, 0] == 0 and _X[-1, 1] == 0:
                    plt.plot(_X[:, 0], _X[:, 1], 0, linewidth=4, color="green")
                    plt.plot(_X[-1, 0], _X[-1, 1], 0, "o", markersize=10, color="green")
                bbox = self.rotate_bbox_zxis(cyclist, _X[-1, 4])
                bbox = self.shift_cuboid(_X[-1, 0], _X[-1, 1], bbox)
                self.add_cube(bbox, ax, color="tab:green", alpha=0.5)
            elif _X[:, 13:16].sum() > 0:  # Traffic lanes
                plt.plot(_X[:, 0], _X[:, 1], 0, color="black")
            elif _X[:, 16].sum() > 0:  # Bike lanes
                plt.plot(_X[:, 0], _X[:, 1], 0, color="tab:red")
            elif _X[:, 18:26].sum() > 0:  # Road lines
                plt.plot(_X[:, 0], _X[:, 1], 0, "--", color="white")
            elif _X[:, 26:29].sum() > 0:  # Road edges
                plt.plot(_X[:, 0], _X[:, 1], 0, linewidth=2, color="white")

        ax.set_zlim(bottom=0, top=5)
        ax.set_aspect("equal")
        ax.set_axis_off()
        ax.set_facecolor("tab:grey")

        is_available = is_available[
            prediction_subsampling_rate
            - 1 : prediction_horizon : prediction_subsampling_rate
        ]
        gt_marginal = gt_marginal[
            prediction_subsampling_rate
            - 1 : prediction_horizon : prediction_subsampling_rate
        ]

        confids_scaled = self.sigmoid(confidences)
        colors = plt.cm.viridis(confidences * 4)

        for pred_id in np.argsort(confidences):
            confid = confidences[pred_id]
            label = f"Pred {pred_id}, confid: {confid:.2f}" if False else ""
            confid_scaled = confids_scaled[pred_id]
            plt.plot(
                np.concatenate(
                    (
                        np.array([[0.0, 0.0]]),
                        predictions[pred_id][is_available > 0][::plot_subsampling_rate],
                    )
                )[:, 0],
                np.concatenate(
                    (
                        np.array([[0.0, 0.0]]),
                        predictions[pred_id][is_available > 0][::plot_subsampling_rate],
                    )
                )[:, 1],
                "-o",
                color=colors[pred_id],
                label=label,
                linewidth=3,  # linewidth,
                markersize=10,  # linewidth+3,
            )

        plt.plot(
            np.concatenate(
                (
                    np.array([0.0]),
                    gt_marginal[is_available > 0][:, 0][::plot_subsampling_rate],
                )
            ),
            np.concatenate(
                (
                    np.array([0.0]),
                    gt_marginal[is_available > 0][:, 1][::plot_subsampling_rate],
                )
            ),
            "--o",
            color="tab:cyan",
            label=label,
            linewidth=4,  # 4
            markersize=10,
        )

        return plt

    def plot_trajectory(self, filename="output/3D_trajectory_plot"):
        predictions = np.zeros(self.future_val_marginal.shape)
        prediction_dummy = np.zeros((6, 10, 2))

        print(predictions.shape)
        plot = self.plot_marginal_predictions_3d(
            vector_data=self.vector_data,
            is_available=self.future_val_marginal,
            gt_marginal=self.gt_marginal,
            predictions=prediction_dummy,
            confidences=np.zeros((6,)),
            x_range=(-50, 100),
            y_range=(-50, 100),
            # gt_marginal=npz_trajectory.gt_marginal,
        )
        plot.savefig(filename)

    def add_cube(self, cube_definition, ax, color="b", edgecolor="k", alpha=0.2):
        cube_definition_array = [np.array(list(item)) for item in cube_definition]

        points = []
        points += cube_definition_array
        vectors = [
            cube_definition_array[1] - cube_definition_array[0],
            cube_definition_array[2] - cube_definition_array[0],
            cube_definition_array[3] - cube_definition_array[0],
        ]

        points += [cube_definition_array[0] + vectors[0] + vectors[1]]
        points += [cube_definition_array[0] + vectors[0] + vectors[2]]
        points += [cube_definition_array[0] + vectors[1] + vectors[2]]
        points += [cube_definition_array[0] + vectors[0] + vectors[1] + vectors[2]]

        points = np.array(points)

        edges = [
            [points[0], points[3], points[5], points[1]],
            [points[1], points[5], points[7], points[4]],
            [points[4], points[2], points[6], points[7]],
            [points[2], points[6], points[3], points[0]],
            [points[0], points[2], points[4], points[1]],
            [points[3], points[6], points[7], points[5]],
        ]

        faces = Poly3DCollection(
            edges, linewidths=1, edgecolors=edgecolor, facecolors=color, alpha=alpha
        )

        ax.add_collection3d(faces)
        # Plot the points themselves to force the scaling of the axes
        ax.scatter(points[:, 0], points[:, 1], points[:, 2], s=0)

    def shift_cuboid(self, x_shift, y_shift, cuboid):
        cuboid = np.copy(cuboid)
        cuboid[:, 0] += x_shift
        cuboid[:, 1] += y_shift

        return cuboid

    def rotate_point_zaxis(self, p, angle):
        rot_matrix = np.array(
            [
                [np.cos(np.deg2rad(angle)), -np.sin(np.deg2rad(angle)), 0],
                [np.sin(np.deg2rad(angle)), np.cos(np.deg2rad(angle)), 0],
                [0, 0, 1],
            ]
        )
        return np.matmul(p, rot_matrix)

    def rotate_bbox_zxis(self, bbox, angle):
        bbox = np.copy(bbox)
        _bbox = []
        angle = np.rad2deg(-angle)
        for point in bbox:
            _bbox.append(self.rotate_point_zaxis(point, angle))

        return np.array(_bbox)

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def legend_without_duplicate_labels(
        figure, fontsize=20, ncols=1, loc="upper left", **kwargs
    ):
        handles, labels = plt.gca().get_legend_handles_labels()
        by_label = dict(zip(labels, handles))
        figure.legend(
            by_label.values(),
            by_label.keys(),
            loc=loc,
            ncols=ncols,
            fontsize=fontsize,
            facecolor="white",
            framealpha=1,
            **kwargs,
        )

    def plot_marginal_predictions(
        self,
        predictions,
        confidences,
        vector_data,
        gt_marginal,
        is_available,
        x_range,
        y_range,
        add_env_legend,
        add_pred_legend,
        plot_predictions=True,
        plot_top1=False,
        dark_mode=False,
        fontsize=20,
        fontsize_legend=20,
        prediction_horizon=50,
        prediction_subsampling_rate=5,
        plot_subsampling_rate=2,
        colors_predictions=[
            "tab:purple",
            "tab:brown",
            "#39737c",
            "tab:pink",
            "tab:olive",
            "tab:orange",
        ],
    ):
        figure(figsize=(15, 15), dpi=80)
        vectors, idx = vector_data[:, :44], vector_data[:, 44]

        gt_marginal = gt_marginal[
            prediction_subsampling_rate
            - 1 : prediction_horizon : prediction_subsampling_rate
        ]
        is_available = is_available[
            prediction_subsampling_rate
            - 1 : prediction_horizon : prediction_subsampling_rate
        ]

        for i in np.unique(idx):
            _vectors = vectors[idx == i]
            if _vectors[:, 26:29].sum() > 0:
                label = "Road edges" if add_env_legend else ""
                plt.plot(
                    _vectors[:, 0],
                    _vectors[:, 1],
                    color="grey",
                    linewidth=4,
                    label=label,
                )
            elif _vectors[:, 13:16].sum() > 0:
                label = "Lane centerlines" if add_env_legend else ""
                color = "white" if dark_mode else "black"
                plt.plot(
                    _vectors[:, 0],
                    _vectors[:, 1],
                    color=color,
                    linewidth=2,
                    label=label,
                )
            elif _vectors[:, 16].sum() > 0:
                label = "Bike lane centerlines" if add_env_legend else ""
                plt.plot(
                    _vectors[:, 0],
                    _vectors[:, 1],
                    "-",
                    color="tab:red",
                    linewidth=2,
                    label=label,
                )
            elif _vectors[:, 18:26].sum() > 0:
                label = "Road lines" if add_env_legend else ""
                plt.plot(
                    _vectors[:, 0],
                    _vectors[:, 1],
                    "--",
                    color="grey",
                    linewidth=2,
                    label=label,
                )
            elif _vectors[:, 30:33].sum() > 0:
                label = "Misc. markings" if add_env_legend else ""
                plt.plot(
                    _vectors[:, 0],
                    _vectors[:, 1],
                    color="grey",
                    linewidth=2,
                    label=label,
                )

        if plot_top1:
            pred_id = np.argsort(confidences)[-1]
            confid = confidences[pred_id]
            plt.plot(
                np.concatenate(
                    (
                        np.array([[0.0, 0.0]]),
                        predictions[pred_id][is_available > 0][::plot_subsampling_rate],
                    )
                )[:, 0],
                np.concatenate(
                    (
                        np.array([[0.0, 0.0]]),
                        predictions[pred_id][is_available > 0][::plot_subsampling_rate],
                    )
                )[:, 1],
                "-o",
                color="tab:orange",
                label=f"Top 1, confid: {confid:.2f}",
                linewidth=4,
                markersize=10,
            )
        elif plot_predictions:
            for pred_id, color in zip(np.argsort(confidences), colors_predictions):
                confid = confidences[pred_id]
                label = (
                    f"Pred {pred_id}, confid: {confid:.2f}" if add_pred_legend else ""
                )
                plt.plot(
                    np.concatenate(
                        (
                            np.array([[0.0, 0.0]]),
                            predictions[pred_id][is_available > 0][
                                ::plot_subsampling_rate
                            ],
                        )
                    )[:, 0],
                    np.concatenate(
                        (
                            np.array([[0.0, 0.0]]),
                            predictions[pred_id][is_available > 0][
                                ::plot_subsampling_rate
                            ],
                        )
                    )[:, 1],
                    "-o",
                    color=color,
                    label=label,
                    linewidth=4,
                    markersize=10,
                )

        label = "Ground truth" if add_pred_legend else ""
        plt.plot(
            np.concatenate(
                (
                    np.array([0.0]),
                    gt_marginal[is_available > 0][:, 0][::plot_subsampling_rate],
                )
            ),
            np.concatenate(
                (
                    np.array([0.0]),
                    gt_marginal[is_available > 0][:, 1][::plot_subsampling_rate],
                )
            ),
            "--o",
            color="tab:cyan",
            label=label,
            linewidth=4,
            markersize=10,
        )

        # 2nd loop to plot agents on top
        for i in np.unique(idx):
            _vectors = vectors[idx == i]
            if _vectors[:, 8].sum() > 0:
                label = "Vehicles" if add_env_legend else ""
                if len(_vectors) >= 7:
                    plt.plot(
                        _vectors[0:2, 0],
                        _vectors[0:2, 1],
                        linewidth=15,
                        color="tab:blue",
                        alpha=0.1,
                    )
                    plt.plot(
                        _vectors[2:4, 0],
                        _vectors[2:4, 1],
                        linewidth=15,
                        color="tab:blue",
                        alpha=0.2,
                    )
                    plt.plot(
                        _vectors[3:7, 0],
                        _vectors[3:7, 1],
                        linewidth=15,
                        color="tab:blue",
                        alpha=0.5,
                    )
                    plt.plot(
                        _vectors[6:, 0],
                        _vectors[6:, 1],
                        linewidth=15,
                        color="tab:blue",
                        alpha=1.0,
                        label=label,
                    )
                elif len(_vectors) >= 5:
                    plt.plot(
                        _vectors[0:3, 0],
                        _vectors[0:3, 1],
                        linewidth=15,
                        color="tab:blue",
                        alpha=0.2,
                    )
                    plt.plot(
                        _vectors[2:5, 0],
                        _vectors[2:5, 1],
                        linewidth=15,
                        color="tab:blue",
                        alpha=0.5,
                    )
                    plt.plot(
                        _vectors[4:6, 0],
                        _vectors[4:6, 1],
                        linewidth=15,
                        color="tab:blue",
                        alpha=1.0,
                    )
                elif len(_vectors) == 1:
                    plt.plot(
                        _vectors[0, 0],
                        _vectors[0, 1],
                        "s",
                        markersize=8,
                        color="tab:blue",
                        alpha=1,
                    )
                else:
                    plt.plot(
                        _vectors[0:-1, 0],
                        _vectors[0:-1, 1],
                        linewidth=15,
                        color="tab:blue",
                        alpha=1,
                    )
            elif _vectors[:, 9].sum() > 0:
                label = "Pedestrians" if add_env_legend else ""
                plt.plot(
                    _vectors[0:-1, 0],
                    _vectors[0:-1, 1],
                    linewidth=4,
                    color="tab:red",
                    alpha=0.6,
                )
                plt.plot(
                    _vectors[-1, 0],
                    _vectors[-1, 1],
                    "o",
                    markersize=8,
                    color="tab:red",
                    alpha=1.0,
                    label=label,
                )
            elif _vectors[:, 10].sum() > 0:
                label = "Cyclists" if add_env_legend else ""
                plt.plot(
                    _vectors[0:-1, 0],
                    _vectors[0:-1, 1],
                    "-",
                    linewidth=4,
                    color="tab:green",
                    alpha=0.6,
                )
                plt.plot(
                    _vectors[-1, 0],
                    _vectors[-1, 1],
                    "D",
                    markersize=10,
                    color="tab:green",
                    alpha=1.0,
                    label=label,
                )

        plt.xlim([x_range[0], x_range[1]])
        plt.ylim([y_range[0], y_range[1]])
        plt.xticks(fontsize=fontsize)
        plt.yticks(fontsize=fontsize)

        if add_env_legend or add_pred_legend:
            self.legend_without_duplicate_labels(
                plt,
                fontsize=fontsize_legend,
                loc="upper left",
                bbox_to_anchor=(-0.65, 1),
            )
