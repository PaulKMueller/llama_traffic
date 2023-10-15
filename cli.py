import cmd
import argparse

from datetime import datetime

from waymo_inform import (get_coordinates,
                          get_direction_of_vehicle,
                          get_total_trajectory_angle,
                          get_relative_displacement,
                          get_vehicles_for_scenario)

from waymo_visualize import (visualize_all_agents_smooth,
                              create_animation,
                              visualize_trajectory)
from waymo_initialize import init_waymo

from waymo_utils import get_scenario_list


class SimpleShell(cmd.Cmd):
    prompt = '(waymo_cli) '
    waymo_scenario = {}
    scenario_loaded = False


    def arg_parser(self):
        # Initializing the available flags for the different commands.
        parser = argparse.ArgumentParser()
        parser.add_argument('-n', '--name', default='World')
        parser.add_argument('-p', '--path')
        parser.add_argument('--ids', action='store_true')
        parser.add_argument('--index')
        parser.add_argument('--example', '-e', action='store_true')
        return parser


    def do_greet(self, arg):
        """Prints a greeting to the user whose name is provided as an argument.
        The format of the command is: greet --name <NAME>
        or: greet -n <NAME>

        Args:
            arg (str): The name of the user to greet.
        """        
        try:
            parsed = self.arg_parser().parse_args(arg.split())
            print(f"Hello, {parsed.name}!")
        except SystemExit:
            pass  # argparse calls sys.exit(), catch the exception to prevent shell exit


    def do_load_scenario(self, arg):
        """Loads the scenario from the given path.
        The format of the command is: load_scenario <PATH>
        or: load_scenario --example
        or: load_scenario -e

        Args:
            arg (str): The path to the scenario that should be loaded.
            Alternatively, the flag --example or -e can be used to load
            a pre-defined example scenario.
        """        

        # For testing purposes you can use the following paths
        # /mrtstorage/datasets/tmp/waymo_open_motion_v_1_2_0/uncompressed/tf_example/training/training_tfexample.tfrecord-00499-of-01000

        parser = self.arg_parser()
        args = parser.parse_args(arg.split())

        # Check for empty arguments (no path provided)
        if (arg == ""):
            print("\nYou have provided no path for the scenario you want to load."
                  "\nPlease provide a path!\n")
            return
        elif (args.example):
            self.waymo_scenario = init_waymo(
                "/mrtstorage/datasets/tmp/waymo_open_motion_v_1_2_0/uncompressed/tf_example/training/training_tfexample.tfrecord-00499-of-01000")
            self.scenario_loaded = True
            print("Successfully initialized the example scenario!")
            return
        else:
            filename = arg.split()[0]
            self.waymo_scenario = init_waymo(filename)
            self.scenario_loaded = True
            print("Successfully initialized the given scenario!")

    def do_print_current_raw_scenario(self, arg):
        """Prints the current scenario that has been loaded in its decoded form.
        This function is for debugging purposes only.

        Args:
            arg (str): No arguments are required.
        """        
        print(self.waymo_scenario)


    def do_plot_scenario(self, arg):
        """Plots the scenario that has previously been
        loaded with the 'load_scenario' command.

        Args:
            arg (str): No arguments are required.
        """        

        parser = self.arg_parser()
        args = parser.parse_args(arg.split())

        if args.index is not None:
            scenario_name = get_scenario_list()[int(args.index)]
            print(scenario_name)
            scenario = init_waymo(('/mrtstorage/datasets/tmp/waymo_open_motion_v_1_2_0'
                                   '/uncompressed/tf_example/training/') + 
                                   scenario_name)
        elif self.scenario_loaded:
                scenario = self.waymo_scenario
        else:
            print(("\nNo scenario has been initialized yet!"
                        " \nPlease use 'load_scenario'"
                        " to load a scenario before calling"
                         " the 'plot_scenario' command.\n"))
            return

        if args.ids:
            print("Plotting scenario with agent ids...")
            images = visualize_all_agents_smooth(
                scenario, with_ids=True)
        else:
            print("Plotting scenario without agent ids...")
            images = visualize_all_agents_smooth(
                scenario, with_ids=False)


        anim = create_animation(images[::5])
        timestamp = datetime.now()
        anim.save(f'/home/pmueller/llama_traffic/output/{timestamp}.mp4',
                    writer='ffmpeg', fps=10)
        print(("Successfully created animation in"
                f" /home/pmueller/llama_traffic/output/{timestamp}.mp4!"))


    def do_list_scenarios(self, arg):
        """Lists all available scenarios in the training folder.
        See:
        /mrtstorage/datasets/tmp/waymo_open_motion_v_1_2_0/uncompressed/tf_example/training

        Args:
            arg (str): No arguments are required.
        """        
        scenarios = get_scenario_list()
        for file in scenarios:
            print(file)


    def do_plot_vehicle(self, arg):
        """Creates a mp4 animation of the trajectory of
        the given vehicle for the loaded scenario.
        Format should be: plot_vehicle <ID>
        Pleas make sure, that you have loaded a scenario before.

        Args:
            arg (str): the vehicle ID for which to plot the trajectory.
        """        

        # Checking if a scenario has been loaded already.
        if not self.scenario_loaded:
            print(("\nNo scenario has been initialized yet!"
                  " \nPlease use 'load_scenario' to load a s"
                  "cenario before calling the 'plot_scenario' command.\n"))
            return

        # Check for empty arguments (no ID provided)
        if (arg == ""):
            print(("\nYou have provided no ID for the vehicle "
                    "that you want to plot.\nPlease provide a path!\n"))
            return

        vehicle_id = arg.split()[0]
        print(f"Plotting vehicle with the ID: {vehicle_id}...")
        images = visualize_all_agents_smooth(
                decoded_example=self.waymo_scenario,
                with_ids=False,
                specific_id=vehicle_id)
        anim = create_animation(images[::5])
        timestamp = datetime.now()
        anim.save(f'/home/pmueller/llama_traffic/output/{timestamp}.mp4',
                      writer='ffmpeg', fps=10)
        print("Successfully created animation in "
              f"/home/pmueller/llama_traffic/output/{timestamp}.mp4!")

    
    def do_plot_trajectory(self, arg):
        """Saves a trajectory (represented as a line) for the given vehicle.
        Format should be: get_trajectory <ID>
        Pleas make sure, that you have loaded a scenario before.

        Args:
            arg (string): The vehicle ID for which to plot the trajectory.
        """        

        # Checking if a scenario has been loaded already.
        if not self.scenario_loaded:
            print(("\nNo scenario has been initialized yet! \nPlease use 'load_scenario'"
                  " to load a scenario before calling the 'plot_scenario' command.\n"))
            return
        
        # Check for empty arguments (no ID provided)
        if (arg == ""):
            print(("\nYou have provided no ID for the vehicle "
                    "whose trajectory you want to get.\nPlease provide a path!\n"))
            return

        vehicle_id = arg.split()[0]
        print(f"Plotting trajectory for vehicle {vehicle_id}...")
        timestamp = datetime.now()
        trajectory = visualize_trajectory(decoded_example=self.waymo_scenario,
                                          specific_id=vehicle_id)
        trajectory.savefig(f"/home/pmueller/llama_traffic/output/{timestamp}.png")


    def do_plot_all_trajectories(self, arg):
        """Saves a trajectory (represented as a line) for all vehicles in the scenario.
        Format should be: plot_all_trajectories
        Please make sure, that you have loaded a scenario before.

        Args:
            arg (str): No arguments are required.
        """        

        # Checking if a scenario has been loaded already.
        if not self.scenario_loaded:
            print(("\nNo scenario has been initialized yet! \nPlease use 'load_scenario'"
                  " to load a scenario before calling the 'plot_scenario' command.\n"))
            return

        print("Plotting trajectories for all vehicles...")
        vehicle_ids = get_vehicles_for_scenario(self.waymo_scenario)

        for vehicle_id in vehicle_ids:
            trajectory = visualize_trajectory(decoded_example=self.waymo_scenario,
                                              specific_id=vehicle_id)
            trajectory.savefig(f"/home/pmueller/llama_traffic/vehicle_trajectories/{vehicle_id}.png")

    
    def do_get_coordinates(self, arg):
        """Saves the coordinates of the given vehicle as a csv file.
        Format should be: get_coordinates <ID>
        Pleas make sure, that you have loaded a scenario before.

        Args:
            arg (str): The vehicle ID for which to get the coordinates.
        """        
        # Checking if a scenario has been loaded already.
        if not self.scenario_loaded:
            print(("\nNo scenario has been initialized yet! \nPlease use 'load_scenario'"
                  " to load a scenario before calling the 'plot_scenario' command.\n"))
            return
        
        # Check for empty arguments (no ID provided)
        if (arg == ""):
            print(("\nYou have provided no ID for the vehicle "
                    "whose trajectory you want to get.\nPlease provide a path!\n"))
            return
        
        vehicle_id = arg.split()[0]
        coordinates = get_coordinates(decoded_example = self.waymo_scenario,
                                      specific_id = vehicle_id)

        timestamp = datetime.now()
        print(coordinates.head())
        coordinates.to_csv(f"/home/pmueller/llama_traffic/output/{timestamp}.scsv")


    def do_get_direction(self, arg):
        """Returns the direction of the given vehicle.
        The direction in this case is defined as one of eight buckets:
            - Straight
            - Straight-Left
            - Straight-Right
            - Left
            - Right
            - Left U-Turn
            - Right U-Turn
            - Stationary

        Format should be: get_direction <ID>
        Pleas make sure, that you have loaded a scenario before.

        Args:
            arg (str): The vehicle ID for which to get the direction bucket.
        """        
        
        # Check for empty arguments (no vehicle ID provided)
        if (arg == ""):
            print(("\nYou have provided no ID for the vehicle "
                    "whose trajectory you want to get.\nPlease provide a path!\n"))
            return
        
        vehicle_id = arg.split()[0]
        coordinates = get_coordinates(decoded_example = self.waymo_scenario,
                                      specific_id = vehicle_id)
        
        print(f"\n{get_direction_of_vehicle(self.waymo_scenario, coordinates)}!\n")


    def do_get_displacement(self, arg):
        """Calculates the total displacement of the vehicle with the given ID
        and prints it.
        Format should be: get_displacement <ID>
        Pleas make sure, that you have loaded a scenario before.

        Args:
            arg (str): Vehicle ID for which to calculate the displacement.
        """        

        # Check for empty arguments (no coordinates provided)
        if (arg == ""):
            print(("\nYou have provided no ID for the vehicle "
                    "whose trajectory you want to get.\nPlease provide a path!\n"))
            return
        
        vehicle_id = arg.split()[0]
        displacement = get_relative_displacement(
            decoded_example=self.waymo_scenario,
            coordinates = get_coordinates(
                decoded_example = self.waymo_scenario,
                specific_id = vehicle_id))
        
        print(f'{round(displacement*100, 2)} %')



    def do_filter_trajectories(self, arg):
        """Filters the trajectories in the loaded scenario in left, right and straight.

        Args:
            arg (str): Arguments for the command.

        Returns:
            str: The path to the folder containing 
            one folder for each direction bucket (see get_direction).
        """
        # TODO: Implement this function.


    def do_get_vehicles_in_loaded_scenario(self, arg):
        """Prints the IDs of all vehicles in the loaded scenario.

        Args:
            arg (str): No arguments are required.
        """        

        # Checking if a scenario has been loaded already.
        if not self.scenario_loaded:
            print(("\nNo scenario has been initialized yet!"
                  " \nPlease use 'load_scenario' to load a s"
                  "cenario before calling the 'plot_scenario' command.\n"))
            return

        filtered_ids = get_vehicles_for_scenario(self.waymo_scenario)
        
        print(*filtered_ids, sep = "\n")
        
        

    
    def do_get_angle_for_vehicle(self, arg):
        """Returns the aggregated delta angles of the trajectory of the given vehicle.
        """

        # Check for empty arguments (no coordinates provided)
        if (arg == ""):
            print(("\nYou have provided no ID for the vehicle "
                    "whose trajectory you want to get.\nPlease provide a path!\n"))
            return
        
        vehicle_id = arg.split()[0]
        coordinates = get_coordinates(decoded_example = self.waymo_scenario,
                                      specific_id = vehicle_id)
        angle = get_total_trajectory_angle(coordinates)
        
        print(f"The total heading change is:{angle} degrees!")


    # Basic command to exit the shell
    def do_exit(self, arg):
        'Exit the shell: EXIT'
        print("Exiting the shell...")
        return True


    do_EOF = do_exit

if __name__ == '__main__':
    SimpleShell().cmdloop()
