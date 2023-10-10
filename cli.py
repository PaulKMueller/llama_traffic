import cmd
import argparse

from datetime import datetime

from waymo_inform import (get_coordinates,
                          get_direction_of_vehicle,
                          get_total_trajectory_angle)

from waymo_visualize import (visualize_all_agents_smooth,
                              create_animation,
                              visualize_trajectory)
from waymo_initialize import init_waymo

from waymo_utils import get_scenario_list


class SimpleShell(cmd.Cmd):
    prompt = '(waymo_cli) '
    waymo_dataset = {}
    scenario_loaded = False


    def arg_parser(self):
        # Initializing the available flags for the different commands.
        parser = argparse.ArgumentParser()
        parser.add_argument('-n', '--name', default='World')
        parser.add_argument('-p', '--path')
        parser.add_argument('--ids', action='store_true')
        parser.add_argument('--index')
        return parser


    def do_greet(self, arg):
        'Greet someone. Flags: --name=NAME'
        try:
            parsed = self.arg_parser().parse_args(arg.split())
            print(f"Hello, {parsed.name}!")
        except SystemExit:
            pass  # argparse calls sys.exit(), catch the exception to prevent shell exit


    def do_load_scenario(self, arg):
        """Initialize the Waymo Open Motion Dataset Scenario for the given path."""

        # For testing purposes you can use the following paths
        # /mrtstorage/datasets/tmp/waymo_open_motion_v_1_2_0/uncompressed/tf_example/training/training_tfexample.tfrecord-00499-of-01000

        # Check for empty arguments (no path provided)
        if (arg == ""):
            print("\nYou have provided no path for the scenario you want to load."
                  "\nPlease provide a path!\n")
            return
        
        filename = arg.split()[0]
        self.waymo_dataset = init_waymo(filename)
        self.scenario_loaded = True
        print("Successfully initialized the given scenario!")

    def do_print_current_raw_scenario(self, arg):
        print(self.waymo_dataset)


    def do_plot_scenario(self, arg):
        """Plots the scenarion that has been loaded with 'load_scenario'."""

        parser = self.arg_parser()
        args = parser.parse_args(arg.split())

        if args.index is not None:
            scenario_name = get_scenario_list()[int(args.index)]
            print(scenario_name)
            scenario = init_waymo('/mrtstorage/datasets/tmp/waymo_open_motion_v_1_2_0/uncompressed/tf_example/training/' + scenario_name)
        elif self.scenario_loaded:
                scenario = self.waymo_dataset
        else:
            print(("\nNo scenario has been initialized yet!"
                        " \nPlease use 'load_scenario'"
                        " to load a scenario before calling the 'plot_scenario' command.\n"))
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
        scenarios = get_scenario_list()
        for file in scenarios:
            print(file)


    def do_plot_vehicle(self, arg):
        """Plots the vehicle with the given ID"""

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
                decoded_example=self.waymo_dataset,
                with_ids=False,
                specific_id=vehicle_id)
        anim = create_animation(images[::5])
        timestamp = datetime.now()
        anim.save(f'/home/pmueller/llama_traffic/output/{timestamp}.mp4',
                      writer='ffmpeg', fps=10)
        print("Successfully created animation in "
              f"/home/pmueller/llama_traffic/output/{timestamp}.mp4!")

    
    def do_get_trajectory(self, arg):
        """Saves a trajectory (represented as a line) for the given vehicle.

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
        trajectory = visualize_trajectory(decoded_example=self.waymo_dataset,
                                          specific_id=vehicle_id)
        trajectory.savefig(f"/home/pmueller/llama_traffic/output/{timestamp}.png")

    
    def do_get_coordinates(self, arg):
        """Prints the coordinates for the given vehicle 
        at each moment in time in the loaded scenario."""
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
        coordinates = get_coordinates(decoded_example = self.waymo_dataset,
                                      specific_id = vehicle_id)

        timestamp = datetime.now()
        print(coordinates.head())
        coordinates.to_csv(f"/home/pmueller/llama_traffic/output/{timestamp}.scsv")


    def do_get_direction(self, arg):
        
        # Check for empty arguments (no vehicle ID provided)
        if (arg == ""):
            print(("\nYou have provided no ID for the vehicle "
                    "whose trajectory you want to get.\nPlease provide a path!\n"))
            return
        
        vehicle_id = arg.split()[0]
        coordinates = get_coordinates(decoded_example = self.waymo_dataset,
                                      specific_id = vehicle_id)
        
        print(f"{get_direction_of_vehicle(coordinates)}!")


    def do_filter_trajectories(self, arg):
        """Filters the trajectories in the loaded scenario in left, right and straight.

        Args:
            arg (str): Arguments for the command.

        Returns:
            str: The path to the folder containing 
            the folders "Left", "Right" and "Straight".
        """        
        # TODO: Implement this function.

    
    def do_get_angle_for_vehicle(self, arg):
        """Returns the aggregated delta angles of the trajectory of the given vehicle.
        """

        # Check for empty arguments (no coordinates provided)
        if (arg == ""):
            print(("\nYou have provided no ID for the vehicle "
                    "whose trajectory you want to get.\nPlease provide a path!\n"))
            return
        
        vehicle_id = arg.split()[0]
        coordinates = get_coordinates(decoded_example = self.waymo_dataset,
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
