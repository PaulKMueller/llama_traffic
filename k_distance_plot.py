import math
import json
import matplotlib.pyplot as plt
from tqdm import tqdm

def calculate_kn_distance(data,k):
    kn_distance = []
    for i in tqdm(range(len(data))):
        eucl_dist = []
        for j in range(len(data)):
            eucl_dist.append(
                math.sqrt(
                    ((data[i,0] - data[j,0]) ** 2) +
                    ((data[i,1] - data[j,1]) ** 2)))

        eucl_dist.sort()
        kn_distance.append(eucl_dist[k])

    return kn_distance

with open("datasets/encoder_output_vehicle_a_mse.json") as encoder_output:
    data = json.load(encoder_output)
    data = data.values()

eps_dist = calculate_kn_distance(data, 2048)
plt.hist(eps_dist,bins=30)
plt.ylabel('n')
plt.xlabel('Epsilon distance')

# calculate_kn_distance(data, 2024)

