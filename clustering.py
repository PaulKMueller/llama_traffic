from sklearn.cluster import DBSCAN, OPTICS
import numpy as np
import json

with open("datasets/encoder_output_vehicle_a_mse.json") as data:
    data = json.load(data).values()
clustering = DBSCAN(eps=3).fit(data)

# with open("output/clustering.txt", "w") as cluster:
#     cluster.write(clustering.labels_)

n_clusters = len(set(clustering.labels_)) - (1 if -1 in clustering.labels_ else 0)
print(n_clusters)
