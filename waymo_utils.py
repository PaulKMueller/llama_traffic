import os


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