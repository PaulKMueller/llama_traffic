from parking_lot import has_parking_lot_refined
import json
from npz_trajectory import NpzTrajectory
from tqdm import tqdm
from npz_utils import list_vehicle_files_absolute

# directory_path = (
#     "/storage_local/fzi_datasets_tmp/waymo_open_motion_dataset/unzipped/train-2e6/"
# )

# output_data = {}

# # with open("output/labeled_scenarios_vehicle_a.json") as file:
# #     data = json.load(file)
# #     data_keys = list(data.keys())
# data_keys = list_vehicle_files_absolute()
# for i in tqdm(range(len(data_keys))):
#     key = data_keys[i]
#     # data[key].append(has_parking_lot(NpzTrajectory(directory_path + key)))
#     output_data[key] = 1 if has_parking_lot_refined(NpzTrajectory(key)) else 0

# with open("output/parking_lot_vehicle_a.json", "w") as output:
#     json.dump(output_data, output, indent=4)

import json

with open("output/parking_lot_data.json", "r") as parking_file:
    parking_lot_data = json.load(parking_file)
parking_keys = list(parking_lot_data.keys())


has_lot = 0
for key in parking_keys:
    has_lot += parking_lot_data[key]

print(has_lot)

# with open("output/labeled_scenarios_vehicle_a.json", "r") as labeled_scenarios:
#     labeled_scenarios_data = json.load(labeled_scenarios)


# output = {}
# for key in parking_keys:
#     combined_list = labeled_scenarios_data[key] + [parking_lot_data[key]]
#     output[key] = combined_list

# with open("output/with_parking_lot.json", "w") as output_file:
#     json.dump(output, output_file, indent=4)
