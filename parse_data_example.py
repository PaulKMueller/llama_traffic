import numpy as np
from glob import glob
from red_motion.data_utils.visualize import plot_marginal_predictions_3d

# load data
samples = sorted(glob("/content/red-motion/src/red_motion/data_utils/demo_files/*.npz"))
data = [np.load(sample) for sample in samples]

# make sure you have added red_motion
# https://github.com/KIT-MRT/red-motion
# git submodule add https://github.com/KIT-MRT/red-motion (?)

idx = 1

plot_marginal_predictions_3d(
    data[idx]["vector_data"],
    predictions=None,
    x_range=(-20, 50),
    y_range=(-20, 50),
    confidences=predictions[idx]["confidences"],
    is_available=data[idx]["future_val_marginal"],
    gt_marginal=data[idx]["gt_marginal"],
)