import os


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
    for filename in os.listdir(directory):
        if filename.startswith("vehicle_a"):
            absolute_path = os.path.abspath(os.path.join(directory, filename))
            vehicle_files.append(absolute_path)
    return vehicle_files
