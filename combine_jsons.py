import json

with open("output/turnaround_vehicle_a_1.json") as data:
    one = json.load(data)
with open("output/turnaround_vehicle_a_2.json") as data:
    two = json.load(data)
with open("output/turnaround_vehicle_a_3.json") as data:
    three = json.load(data)
with open("output/turnaround_vehicle_a_4.json") as data:
    four = json.load(data)

combined_data = {}
combined_data.update(one)
combined_data.update(two)
combined_data.update(three)
combined_data.update(four)

with open("output/turnaround_combined.json", "w") as output:
    json.dump(combined_data, output, indent=4)
