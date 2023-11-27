import numpy as np


class StupidModel:
    def __init__(self):
        pass

    def predict(self, x, y):
        # Return a numpy array with 101 points, all the same
        x_cords = np.ones(101) * x
        y_cords = np.ones(101) * y
        increment = 1
        for y in y_cords:
            y += increment
            increment += 3
        return list(zip(np.ones(101) * x, np.ones(101) * y))
