from PIL import Image
import os

# Path to the folder containing images
folder_path = "pca_frames_cluster"

# Retrieve file names of images in the folder
image_files = [
    f for f in os.listdir(folder_path) if os.path.isfile(os.path.join(folder_path, f))
]

# Sort files if necessary; depends on your naming convention
# This is important for the order of images in the GIF
image_files.sort()

# Open images and save them in a list
images = [Image.open(os.path.join(folder_path, f)) for f in image_files]

# Save the first image with .save() and append the rest with .append()
# Adjust the 'duration' for frame display time (in milliseconds)
images[0].save(
    "output_cluster_colored.gif",
    save_all=True,
    append_images=images[1:],
    optimize=False,
    duration=100,
    loop=0,
)

print("GIF created successfully!")
