import json

with open("output/parking_lot_vehicle_a.json", "r") as parking_file:
    parking_lot_data = json.load(parking_file)
parking_keys = list(parking_lot_data.keys())

output = {}

for key in parking_keys:
    new_key = key.split("/")[-1]
    output[new_key] = parking_lot_data[key]

with open("output/parking_lot_data.json", "w") as output_data:
    json.dump(output, output_data, indent=4)
