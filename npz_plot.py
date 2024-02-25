from npz_trajectory import NpzTrajectory
import sys
import yaml

with open("config.yml") as config:
    npz_directory = yaml.safe_load(config)["npz_dataset"]

arguments = sys.argv

trajectory = NpzTrajectory(npz_directory + arguments[1])
trajectory.plot_scenario()
