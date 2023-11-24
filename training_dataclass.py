import json


class TrainingData:
    def __init__(self, data_path):
        self.data_path = data_path
        self.data = self.load_data(data_path)

    def __str__(self):
        return self.data_path

    def get_size(self):
        """Returns the number of objects in the TrainingData."""
        return len(self.data)

    @staticmethod
    def load_data(filepath):
        """Loads trajectory data from a JSON file.

        Args:
            filepath (str): The path to the JSON file containing the trajectory data.

        Returns:
            dict: The loaded trajectory data.
        """
        with open(filepath, "r") as json_file:
            data_dict = json.load(json_file)
        return data_dict
