import random
import os

SCENARIO_LABEL_LIST = [
    "vehicle",
    "pedestrian",
    "cyclist",
    "freeway",
    "surface_street",
    "bike_lane",
    "stop_sign",
    "crosswalk",
    "speed_bump",
    "driveway",
]

SCENARIO_FEATURES = [
    "vehicle",
    "pedestrian",
    "cyclist",
    "freeway",
    "surface_street",
    "bike_lane",
    "stop_sign",
    "crosswalk",
    "driveway",
    "parking_lot",
    "turnaround",
    "intersection",
]


def list_vehicle_files_relative(
    directory="/storage_local/fzi_datasets_tmp/waymo_open_motion_dataset/unzipped/train-2e6/",
):
    """
    Listet alle Dateien in einem angegebenen Verzeichnis auf, die mit 'vehicle' beginnen.

    Args:
    directory (str): Der Pfad zum Verzeichnis, in dem gesucht werden soll.

    Returns:
    list: Eine Liste von Dateinamen, die mit 'vehicle' beginnen.
    """
    vehicle_files = []
    for filename in os.listdir(directory):
        if filename.startswith("vehicle_a"):
            vehicle_files.append(filename)
    return vehicle_files


def get_random_npz_trajectory():
    return random.choice(list_vehicle_files_absolute())


def list_vehicle_files_absolute(
    directory="/storage_local/fzi_datasets_tmp/waymo_open_motion_dataset/unzipped/train-2e6/",
):
    """
    Listet alle Dateien in einem angegebenen Verzeichnis auf, die mit 'vehicle' beginnen und gibt ihre absoluten Pfade zurück.

    Args:
    directory (str): Der Pfad zum Verzeichnis, in dem gesucht werden soll.

    Returns:
    list: Eine Liste von absoluten Pfaden zu Dateien, die mit 'vehicle' beginnen.
    """
    vehicle_files = []
    counter = 0
    for filename in os.listdir(directory):
        # if counter == 50000:
        #     break
        if filename.startswith("vehicle_a"):
            absolute_path = os.path.abspath(os.path.join(directory, filename))
            vehicle_files.append(absolute_path)
            counter += 1
    return vehicle_files


def one_hot_encode_trajectory(input_string, vocabulary):
    # Split the input string into words
    words = input_string.split(" ")

    # Initialize the one-hot-encoded vector with zeros
    one_hot_encoded_vector = [0] * len(vocabulary)

    # Create a dictionary for faster lookup
    word_to_index = {word: i for i, word in enumerate(vocabulary)}

    # Set the corresponding index to 1 for each word in the input that is in the vocabulary
    for word in words:
        if word in word_to_index:
            one_hot_encoded_vector[word_to_index[word]] = 1

    return one_hot_encoded_vector


def decode_one_hot_vector(one_hot_encoded_vector, vocabulary=SCENARIO_FEATURES):
    # Initialize an empty list to hold words that are present (indicated by a 1 in the vector)
    words_present = []

    # Iterate over the one-hot-encoded vector
    for i, value in enumerate(one_hot_encoded_vector):
        # If the value at the current position is 1, append the corresponding word from the vocabulary
        if value == 1:
            words_present.append(vocabulary[i])

    # Join the words present into a single string, separated by spaces
    decoded_string = " ".join(words_present)

    return decoded_string
