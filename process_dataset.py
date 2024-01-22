with open("output/direction_labeled_npz_vehicle_a_new.json", "r") as file:
    text = file.read()
    new_text = text.replace("'", '"').replace("(", "[").replace(")", "]")

with open("output/processed.json", "w") as new_file:
    new_file.write(new_text)
