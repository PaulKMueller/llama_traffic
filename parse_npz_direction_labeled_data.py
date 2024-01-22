def insert_newline_after_comma(file_path):
    try:
        with open(file_path, "r") as file:
            content = file.read()

        # Replace every comma with a comma followed by a newline
        modified_content = content.replace(",", ",\n")

        return modified_content

    except FileNotFoundError:
        return "File not found."
    except Exception as e:
        return f"An error occurred: {e}"


# Usage example:
file_path = "/home/pmueller/llama_traffic/output/direction_labeled_npz_vehicle_b.json"
with open("output/test.json", "w") as file:
    content = insert_newline_after_comma(file_path)
    file.write(content)
