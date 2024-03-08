import sys
import numpy as np
import json

# with open("output/turnaround_combined.json") as data:
#     data = json.load(data)

# keys = list(data.keys())


# count = 0
# for key in keys:
#     count += data[key]

# print(count)
# print(468108 - count)

# output = {}

# with open("output/intersection_combined_parsed.json") as turnarounds:
#     turnaround_data = json.load(turnarounds)

# with open("output/with_turnaround.json") as input:
#     data = json.load(input)

# for key, value in data.items():
#     item = data[key]
#     item.append(turnaround_data[key])
#     output[key] = item

with open("output/scenario_features.json") as data_file:
    data = np.array(list(json.load(data_file).values()))
    print(data.shape)
    np.set_printoptions(threshold=sys.maxsize)
    print(data.sum(axis=0) / 468108)
