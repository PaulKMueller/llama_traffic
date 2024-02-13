import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import json

# Sample data: Replace this with your actual dataset
# Assuming each row is a sample and each column is a feature

# It's often a good idea to scale the data before performing PCA
with open("datasets/encoder_output_vehicle_a_mse.json") as data:
    data = json.load(data).values()

# Initialize PCA, setting the number of components to 3
pca = PCA(n_components=3)

# Fit PCA on the scaled dataset and transform the data
X_pca = pca.fit_transform(data)

# Plotting the 3D points
fig = plt.figure()
ax = fig.add_subplot(111, projection="3d")

# Extracting the three dimensions
x, y, z = X_pca[:, 0], X_pca[:, 1], X_pca[:, 2]

# Scatter plot
ax.scatter(x, y, z, c="r", marker="o")

# Adding labels for clarity
ax.set_xlabel("PC1")
ax.set_ylabel("PC2")
ax.set_zlabel("PC3")

plt.savefig("output/pca_test.png")
