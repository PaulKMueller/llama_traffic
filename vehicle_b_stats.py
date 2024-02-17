import json
import numpy as np


with open("output/labeled_scenarios_vehicle_b.json") as data:
    data = json.load(data)

arrays = [np.array(value) for value in data.values()]

print(np.sum(arrays, axis=0))
