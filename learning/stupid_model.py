import numpy as np


class StupidModel:
    def __init__(self):
        pass

    def predict(self, x, y):
        # Return a numpy array with 101 points, all the same
        x_cords = np.ones(101) * x
        y_cords = np.array([i for i in range(101)])
        return x_cords, y_cords
