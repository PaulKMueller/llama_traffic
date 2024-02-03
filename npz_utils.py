import os


def list_vehicle_files_relative(directory="datasets/npz_test_data/train-2e6"):
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


def list_vehicle_files_absolute(
    directory="/storage_local/fzi_datasets_tmp/waymo_open_motion_dataset/unzipped/train-2e6/vehicle_d_13657_00002_4856147881.npz",
):
    """
    Listet alle Dateien in einem angegebenen Verzeichnis auf, die mit 'vehicle' beginnen und gibt ihre absoluten Pfade zurück.

    Args:
    directory (str): Der Pfad zum Verzeichnis, in dem gesucht werden soll.

    Returns:
    list: Eine Liste von absoluten Pfaden zu Dateien, die mit 'vehicle' beginnen.
    """
    vehicle_files = []
    maximum = 50000
    for filename in os.listdir(directory):
        if maximum == 0:
            return vehicle_files
        if filename.startswith("vehicle_b"):
            absolute_path = os.path.abspath(os.path.join(directory, filename))
            vehicle_files.append(absolute_path)
            maximum -= 1
    return vehicle_files
