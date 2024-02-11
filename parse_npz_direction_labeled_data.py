file_path = "/home/pmueller/llama_traffic/output/test_2.json"

try:
    with open(file_path, "r") as file:
        content = file.readlines()

    for line in content:
        modified_line = line.replace("(", "[")
        with open("output/test_3.json", "a") as file:
            file.write(modified_line)

    # Replace every comma with a comma followed by a newline
    modified_content = content.replace("'", '"')

except FileNotFoundError:
    print("File not found.")
except Exception as e:
    print(f"An error occurred: {e}")
