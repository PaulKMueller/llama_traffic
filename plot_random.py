import yaml

from npz_utils import list_vehicle_files_absolute
import random
from npz_trajectory import NpzTrajectory
from turn_around import has_turnaround


with open("config.yml") as config:
    config = yaml.safe_load(config)
    data_directory = config["npz_dataset"]
vehicles_file_paths = list_vehicle_files_absolute(data_directory)

while True:
    chosen_vehicle = random.choice(vehicles_file_paths)
    npz_trajectory = NpzTrajectory(chosen_vehicle)
    if has_turnaround(npz_trajectory):
        # npz_trajectory.plot_trajectory()
        npz_trajectory.plot_scenario()
        print(chosen_vehicle)
        break
