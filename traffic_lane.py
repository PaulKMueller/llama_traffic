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
