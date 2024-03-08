import math
import pandas as pd
import numpy as np


class TrafficLane:
    def __init__(self, x: np.array, y: np.array):
        self.x = x
        self.y = y
        self.coordinates = np.column_stack((x, y))

    def min_distance_to_point(self, point: tuple):
        return min(
            [np.linalg.norm(point - coordinate) for coordinate in self.coordinates]
        )

    # def get_delta_angles(self):
    #     """Returns the angle between each segment in the trajectory.
    #     Args:
    #         coordinates (np.array): A numpy array containing the coordinates
    #                                     of the vehicle trajectory.
    #     """
    #     delta_angles = []

    #     for i in range(1, len(self.coordinates) - 1):
    #         # Calculate the direction vector of the current segment
    #         current_vector = np.array(
    #             (
    #                 self.coordinates[i + 1][0] - self.coordinates[i][0],
    #                 self.coordinates[i + 1][1] - self.coordinates[i][1],
    #             )
    #         )

    #         # Calculate the direction vector of the previous segment
    #         previous_vector = np.array(
    #             (
    #                 self.coordinates[i][0] - self.coordinates[i - 1][0],
    #                 self.coordinates[i][1] - self.coordinates[i - 1][1],
    #             )
    #         )

    #         # Compute the angle between the current and previous direction vectors
    #         angle = self.angle_between(current_vector, previous_vector)

    #         direction = self.get_gross_direction_for_three_points(
    #             self.coordinates[i - 1], self.coordinates[i], self.coordinates[i + 1]
    #         )

    #         if direction == "Right":
    #             angle = -angle

    #         delta_angles.append(angle)

    #     return delta_angles

    def get_delta_angles(self):
        # Vectorized implementation of delta angles calculation
        deltas = np.diff(self.coordinates, axis=0)
        angles = np.arctan2(deltas[:, 1], deltas[:, 0])
        delta_angles = np.diff(angles)
        delta_angles_degrees = np.degrees(delta_angles)
        filtered_angles = delta_angles_degrees[np.abs(delta_angles_degrees) <= 20]

        return filtered_angles

    def get_cumulative_delta_angle(self):
        return sum(self.get_delta_angles())

    def unit_vector(self, vector):
        """Returns the unit vector of the vector."""
        return vector / np.linalg.norm(vector)

    def angle_between(self, v1, v2):
        """Returns the angle in radians between vectors 'v1' and 'v2'::"""
        v1_u = self.unit_vector(v1)
        v2_u = self.unit_vector(v2)
        return math.degrees(np.arccos(np.clip(np.dot(v1_u, v2_u), -1.0, 1.0)))

    @staticmethod
    def get_gross_direction_for_three_points(
        start: np.array, intermediate: np.array, end: np.array
    ):
        """Returns left, right, or straight depending on the direction of the trajectory.
        Args:
            start (pd.DataFrame): The coordinates of the starting point.
            intermediate (pd.DataFrame): The coordinates of the intermediate point.
            end (pd.DataFrame): The coordinates of the ending point.
        """
        # Calculate vectors
        vector1 = intermediate - start
        vector2 = end - intermediate

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

    def get_min_dist_to_other_lane(self, lane):
        min_dist = 1000000
        for point in self.coordinates:
            for other_point in lane.coordinates:
                dist = np.linalg.norm(point - other_point)
                if dist < min_dist:
                    min_dist = dist

        return min_dist
