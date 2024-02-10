filepath = "datasets/direction_labeled_npz_vehicle_a.json"
counter_files = 0
counter_ends = 0
end_brackets = 0
begin_brackets = 0
with open(filepath, "r") as file:
    # data = file.readlines()
    # print(len(data))
    content = file.read()
#     # print(len(content))
#     content = content[:15372398] + '"' + content[15372398] + '"' + content[15372399:]
#     print(content[15372397])
#     print(content[15372398])
#     print(content[15372399])
#     print(content[15372400])
#     print(content[15372401])
#     print(content[15372402])
#     print(content[15372403])
#     print(content[15372404])
#     print(content[15372405])
#     print(content[15372406])
#     print(content[15372407])
#     print(content[15372408])
#     print(content[15372409])
#     print(content[15372410])
#     print(content[15372411])
#     print(content[15372412])
#     print(content[15372413])
#     print(content[15372414])
#     print(content[15372415])
#     print(content[15372416])
#     print(content[15372417])
#     print(content[15372418])
# with open(filepath, "w") as file:
#     file.write(content)

    # print(content[0][15372200:])
    print(content.count('{'))
    print(content.count('}'))
    print(content.count('['))
    print(content.count(']'))

    # for index, line in enumerate(data):
    #     if "]" in line:
    #         counter_files += 1
    #     if "]," in line:
    #         counter_ends += 1
    #     if '{{' in line:
    #         begin_brackets += 1
    #         # print(index)
    #     if "}}" in line:
    #     print("Test")
    #     if index == 3059:
    #         print(line)
    #         end_brackets += 1
    # print(data[-1])
    # print(data[1])
    # print(data[-3])
# print(counter_files)
# print(counter_ends)
# print(end_brackets)
# print(begin_brackets)

# Define the filename and the line number to be modified
# filename = 'datasets/encoder_output_vehicle_a_cos.json'
# line_number_to_modify = 468108 # Assuming you want to modify the 4th line

# # Read the file
# with open(filename, 'r') as file:
#     lines = file.readlines()

# # Modify the specified line if it exists
# if 0 < line_number_to_modify <= len(lines):
#     lines[line_number_to_modify - 1] = lines[line_number_to_modify - 1].replace('{', '').replace('}', '')

# # Write the changes back to the file
# with open(filename, 'w') as file:
#     file.writelines(lines)
