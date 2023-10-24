import os
import numpy as np
from scipy import interpolate
import pandas as pd

def get_scenario_list():
        """Returns an array of strings with all the paths to the available scenarios.
        """
        try:
            folder_path = '/mrtstorage/datasets/tmp/waymo_open_motion_v_1_2_0/uncompressed/tf_example/training'
            all_entries = os.listdir(folder_path)
            
            # Filter the list, keeping only files (not directories)
            files = [entry for entry in all_entries if os.path.isfile(
                os.path.join(folder_path, entry))]
            
            return files
    
        except FileNotFoundError:
            return f"Error: The folder {folder_path} does not exist."
        except PermissionError:
            return f"Error: Permission denied for accessing {folder_path}."
        except Exception as e:  
            return f"Error: An unexpected error occurred - {str(e)}."
        

def get_scenario_id(scenario_name):
    """Returns the scenario id for the given scenario name.

    Args:
        scenario_name (str): The name of the scenario
    """
    scenario_list = get_scenario_list()

    scenario_index = scenario_list.index(scenario_name)
    return scenario_index


def get_spline_for_coordinates(coordinates):
    """Returns the splined coordinates for the given trajectory coordinates.

    Args:
        coordinates (pd.DataFrame): The coordinates of a vehicle represented as a DataFrame
    """     
    # Get the x and y coordinates
    x = coordinates["X"]
    y = coordinates["Y"]
    
    # Get the spline for the x and y coordinates
    tck, u = interpolate.splprep([x, y], s=5)
    unew = np.arange(0, 1.01, 0.01)
    spline = interpolate.splev(unew, tck)

    result = pd.DataFrame({"X": spline[0], "Y": spline[1]})
    
    return result