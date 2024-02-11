def remove_last_comma(file_path):
    # Read the file content
    with open(file_path, "r") as file:
        content = file.read()

    # Find the last comma in the content
    last_comma_index = content.rfind(",")

    # If a comma is found, remove it
    if last_comma_index != -1:
        updated_content = content[:last_comma_index] + content[last_comma_index + 1 :]

        # Write the updated content back to the file
        with open(file_path, "w") as file:
            file.write(updated_content)
        print("Last comma removed successfully.")
    else:
        print("No comma found in the file.")


# Replace 'yourfile.txt' with the actual path to your file
file_path = "datasets/direction_labeled_npz_vehicle_a.json"
remove_last_comma(file_path)
