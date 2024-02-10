import json

input_file_path = 'datasets/direction_labeled_npz_vehicle_a.json'
output_file_path = 'datasets/direction_labeled_npz_vehicle_a_modified.json'

try:
    with open(input_file_path, 'r', encoding='utf-8') as file:
        data = json.load(file)
    
    # Process the keys
    modified_data = {key.split('/')[-1]: value for key, value in data.items()}
    
    # Save the modified JSON data
    with open(output_file_path, 'w', encoding='utf-8') as file:
        json.dump(modified_data, file, indent=4)
    
    print("JSON keys have been successfully modified.")
except json.JSONDecodeError as e:
    print(f"An error occurred while parsing the JSON file: {e}")
except Exception as e:
    print(f"An unexpected error occurred: {e}")