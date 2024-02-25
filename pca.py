# import json
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D

# # Load the data from JSON file
# with open("datasets/encoder_output_vehicle_a_mse.json", "r") as f:
#     data = json.load(f)

# # Assuming the data is a list of lists

# fig = plt.figure(figsize=(12, 12))
# ax = fig.add_subplot(projection="3d")
# data_points = list(data.values())
# data_matrix = np.array(data_points)

from sklearn.decomposition import PCA

# # Initialize PCA, we'll reduce to 2 dimensions for easy visualization
pca = PCA(n_components=3)

# # Apply PCA to the data
# reduced_data = pca.fit_transform(data_matrix)


# # Plot the reduced data
# ax.scatter(reduced_data[:, 0], reduced_data[:, 1], reduced_data[:, 2])
# # ax.xlabel("Principal Component 1")
# # ax.ylabel("Principal Component 2")
# # ax.title("PCA of High-Dimensional Data")
# plt.savefig("output/pca.png")


# import json
# import matplotlib.pyplot as plt
# import numpy as np
# from sklearn.decomposition import PCA

# # Load the data from JSON file
# with open("datasets/encoder_output_vehicle_a_mse.json", "r") as f:
#     data = json.load(f)

# # Assuming the data is a list of lists
# fig = plt.figure(figsize=(12, 12))
# ax = fig.add_subplot(projection="3d")
# data_points = list(data.values())
# data_matrix = np.array(data_points)

# # Initialize PCA, we'll reduce to 3 dimensions
# pca = PCA(n_components=3)

# # Apply PCA to the data
# reduced_data = pca.fit_transform(data_matrix)

# # Plot the reduced data with colors based on the z-axis values
# # Use a colormap to specify the color scheme, e.g., 'viridis', 'jet', etc.
# scatter = ax.scatter(
#     reduced_data[:, 0],
#     reduced_data[:, 1],
#     reduced_data[:, 2],
#     c=reduced_data[:, 2],
#     cmap="viridis",
# )

# # Adding a color bar to understand the mapping of colors
# plt.colorbar(scatter, ax=ax, label="Height (Z-axis value)")

# # Uncomment and correct these lines if you want to set labels and titles
# # ax.set_xlabel("Principal Component 1")
# # ax.set_ylabel("Principal Component 2")
# # ax.set_zlabel("Principal Component 3")
# # ax.set_title("PCA of High-Dimensional Data")

# plt.savefig("output/pca_colored.png")


# import json
# import matplotlib.pyplot as plt
# import numpy as np
# from sklearn.decomposition import PCA

# # Load the data from JSON file


# def get_matplotlib_color(number):
#     """
#     Returns a Matplotlib color based on the input number.

#     Parameters:
#     - number: int, a number between 0 and 7 inclusive.

#     Returns:
#     - str, a named Matplotlib color.
#     """
#     colors = ["blue", "green", "red", "cyan", "magenta", "yellow", "black", "white"]
#     if 0 <= number <= 7:
#         return colors[number]
#     else:
#         raise ValueError("Input number must be between 0 and 7.")


data_points = np.load("datasets/encoder_output_a_mse.npy")

# # Assuming the data is a list of lists
# fig = plt.figure(figsize=(12, 12))
# ax = fig.add_subplot(projection="3d")
# data_points = list(data.values())
data_matrix = np.array(data_points)

# # Initialize PCA, we'll reduce to 3 dimensions
pca = PCA(n_components=3)

# # Apply PCA to the data
reduced_data = pca.fit_transform(data_matrix)
# reduced_data = np.zeros(8)

# # Plot the initial state of the reduced data
# scatter = ax.scatter(
#     reduced_data[:, 0],
#     reduced_data[:, 1],
#     reduced_data[:, 2],
#     c=reduced_data[:, 2],
#     cmap="viridis",
# )

# # Set labels and title
# ax.set_xlabel("Principal Component 1")
# ax.set_ylabel("Principal Component 2")
# ax.set_zlabel("Principal Component 3")
# ax.set_title("PCA of High-Dimensional Data")

# # Create a directory to save the frames
import os

# frames_dir = "pca_frames"
# os.makedirs(frames_dir, exist_ok=True)

# # Save each frame
# for angle in range(0, 360, 2):
#     print(angle)
#     ax.view_init(elev=10.0, azim=angle)
#     filename = f"{frames_dir}/frame_{angle:03d}.png"
#     plt.savefig(filename)

# print("All frames saved.")


import json
import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA


labels = [
    "Left",
    "Right",
    "Stationary",
    "Straight",
    "Straight-Left",
    "Straight-Right",
    "Right-U-Turn",
    "Left-U-Turn",
]
colors = [
    "orange",  # Left
    "blue",  # Right
    "grey",  # Stationary
    "black",  # Straight
    "yellow",  # Straight-Left
    "lightblue",  # Straight-Right
    "purple",  # Right-U-Turn
    "red",  # Left-U-Turn (replacing white with purple for better visibility)
]


# Step 1: Define the color selection function
def get_matplotlib_color(number):
    colors = [
        "orange",  # Left
        "blue",  # Right
        "grey",  # Stationary
        "black",  # Straight
        "yellow",  # Straight-Left
        "lightblue",  # Straight-Right
        "purple",  # Right-U-Turn
        "red",  # Left-U-Turn (replacing white with purple for better visibility)
    ]
    if 0 <= number <= 7:
        return colors[number]
    else:
        raise ValueError("Input number must be between 0 and 7.")


# Assuming `number_list` is your list of numbers corresponding to the points
number_list = np.load("datasets/raw_direction_labels.npy")
print(number_list[:5])
# number_list = [your list of numbers here]  # Replace this with your actual list of numbers

# Generate a list of colors for each point
point_colors = [get_matplotlib_color(number) for number in number_list]

# Assuming `reduced_data` contains your PCA-transformed data
fig = plt.figure(figsize=(12, 12))
ax = fig.add_subplot(projection="3d")

# Step 3: Plot using the generated colors
scatter = ax.scatter(
    reduced_data[:, 0], reduced_data[:, 1], reduced_data[:, 2], c=point_colors
)

# Set labels and title (optional, for better understanding of the plot)
ax.set_xlabel("Principal Component 1")
ax.set_ylabel("Principal Component 2")
ax.set_zlabel("Principal Component 3")
ax.set_title("PCA of High-Dimensional Data")

# plt.savefig("output/clustered.png")

legend_handles = [
    Line2D([0], [0], color=color, marker="o", linestyle="", markersize=10, label=label)
    for color, label in zip(colors, labels)
]
plt.legend(handles=legend_handles, title="Trajectory Bucket")

frames_dir = "pca_frames_cluster"
os.makedirs(frames_dir, exist_ok=True)

# Save each frame
for angle in range(0, 360, 2):
    print(angle)
    ax.view_init(elev=10.0, azim=angle)
    filename = f"{frames_dir}/frame_{angle:03d}.png"
    plt.savefig(filename)

print("All frames saved.")
