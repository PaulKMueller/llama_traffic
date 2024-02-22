import json
import numpy as np

# Load the data from JSON file
with open("datasets/encoder_output_vehicle_a_mse.json", "r") as f:
    data = json.load(f)

# Assuming the data is a list of lists
data_points = list(data.values())
data_matrix = np.array(data)

from sklearn.decomposition import PCA

# Initialize PCA, we'll reduce to 2 dimensions for easy visualization
pca = PCA(n_components=2)

# Apply PCA to the data
reduced_data = pca.fit(data_matrix)

import matplotlib.pyplot as plt

# Plot the reduced data
plt.scatter(reduced_data[:, 0], reduced_data[:, 1])
plt.xlabel("Principal Component 1")
plt.ylabel("Principal Component 2")
plt.title("PCA of High-Dimensional Data")
plt.savefig("output/pca.png")
