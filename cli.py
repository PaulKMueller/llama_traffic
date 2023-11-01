import cmd
import argparse
import os

import pandas as pd

from datetime import datetime

from waymo_inform import (get_coordinates,
                          get_direction_of_vehicle,
                          get_total_trajectory_angle,
                          get_relative_displacement,
                          get_vehicles_for_scenario,
                          get_delta_angles,
                          get_sum_of_delta_angles)

from waymo_visualize import (visualize_all_agents_smooth,
                              create_animation,
                              visualize_trajectory,
                              visualize_coordinates,
                              visualize_map)
from waymo_initialize import init_waymo

from waymo_utils import get_scenario_list, get_spline_for_coordinates, get_scenario_id

from trajectory_encoder import get_trajectory_embedding

from bert_encoder import get_bert_embedding


class SimpleShell(cmd.Cmd):
    prompt = '(waymo_cli) '
    waymo_scenario = {}
    scenario_loaded = False
    scenario_name = ""


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


    def do_plot_map(self, arg):
        """Plots the map for the loaded scenario.
        """        
        # Checking if a scenario has been loaded already.
        if not self.scenario_loaded:
            print(("\nNo scenario has been initialized yet!"
                  " \nPlease use 'load_scenario' to load a s"
                  "cenario before calling the 'plot_scenario' command.\n"))
            return
        
        image = visualize_map(self.waymo_scenario)
        image.savefig(f"/home/pmueller/llama_traffic/output/roadgraph_{get_scenario_id(self.scenario_name)}.png")


    def do_visualize_trajectory(self, arg):
        """Plots the trajectory for the given vehicle ID.
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


        coordinates = get_coordinates(decoded_example = self.waymo_scenario,
                                        specific_id = arg)
        trajectory = visualize_coordinates(self.waymo_scenario, coordinates)

        trajectory.savefig(f"/home/pmueller/llama_traffic/output/{vehicle_id}.png")




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

        args = arg.split()

        # Check for empty arguments (no path provided)
        if (arg == ""):
            print("\nYou have provided no path for the scenario you want to load."
                  "\nPlease provide a path!\n")
            return
        
        elif (args[0] == "-e" or args[0] == "--example"):
            self.scenario_name = "training_tfexample.tfrecord-00499-of-01000"
            self.waymo_scenario = init_waymo(
                "/mrtstorage/datasets/tmp/waymo_open_motion_v_1_2_0/uncompressed/tf_example/training/training_tfexample.tfrecord-00499-of-01000")
            self.scenario_loaded = True
            print("Successfully initialized the example scenario!")
            return
        elif (args[0] == "-p" or args[0] == "--path"):
            filename = arg.split()[0]
            print(f"Das Szenario wird geladen: {filename}")
            self.waymo_scenario = init_waymo(args[1])
            self.scenario_name = filename
            self.scenario_loaded = True
            print("\nSuccessfully initialized the given scenario!\n")
            return
        elif (args[0] == "-i" or args[0] == "--index"):
            scenario_name = get_scenario_list()[int(args[1])]
            print(f"The scenario {scenario_name} is being loaded...")
            self.waymo_scenario = init_waymo(('/mrtstorage/datasets/tmp/waymo_open_motion_v_1_2_0'
                                   '/uncompressed/tf_example/training/') + 
                                   scenario_name)
            self.scenario_name = scenario_name
            self.scenario_loaded = True
            print("\nSuccessfully initialized the given scenario!\n")
            return
        else:
            try:
                self.waymo_scenario = init_waymo(args[0])
                self.scenario_loaded = True
                print("\nSuccessfully initialized the given scenario!\n")
                return
            except:
                print("\nYou have provided no path for the scenario you want to load. Please use the -p flag to indicate the path.\n")

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
            print("\nPlotting scenario with agent ids...")
            images = visualize_all_agents_smooth(
                scenario, with_ids=True)
        else:
            print("\nPlotting scenario without agent ids...")
            images = visualize_all_agents_smooth(
                scenario, with_ids=False)


        anim = create_animation(images[::5])
        timestamp = datetime.now()
        anim.save(f'/home/pmueller/llama_traffic/output/{timestamp}.mp4',
                    writer='ffmpeg', fps=10)
        print(("Successfully created animation in"
                f" /home/pmueller/llama_traffic/output/{timestamp}.mp4!\n"))


    def do_list_scenarios(self, arg):
        """Lists all available scenarios in the training folder.
        See:
        /mrtstorage/datasets/tmp/waymo_open_motion_v_1_2_0/uncompressed/tf_example/training

        Args:
            arg (str): No arguments are required.
        """        
        scenarios = get_scenario_list()
        print("\n")
        counter = 0
        for file in scenarios:
            print(f"{counter}: /mrtstorage/datasets/tmp/waymo_open_motion_v_1_2_0/uncompressed/tf_example/training/{file}")
            counter += 1

        print("\n")


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
        print(f"\nPlotting vehicle with the ID: {vehicle_id}...")
        images = visualize_all_agents_smooth(
                decoded_example=self.waymo_scenario,
                with_ids=False,
                specific_id=vehicle_id)
        anim = create_animation(images[::5])
        timestamp = datetime.now()
        anim.save(f'/home/pmueller/llama_traffic/output/{timestamp}.mp4',
                      writer='ffmpeg', fps=10)
        print(("Successfully created animation in "
              f"/home/pmueller/llama_traffic/output/{timestamp}.mp4!\n"))

    
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
        print(f"\nPlotting trajectory for vehicle {vehicle_id}...")
        timestamp = datetime.now()
        trajectory = visualize_trajectory(decoded_example=self.waymo_scenario,
                                          specific_id=vehicle_id)
        # trajectory.savefig(f"/home/pmueller/llama_traffic/output/{timestamp}.png")
        sum_of_delta_angles = get_sum_of_delta_angles(
            get_coordinates(self.waymo_scenario, vehicle_id))
        trajectory.savefig(f"/home/pmueller/llama_traffic/output/{vehicle_id}_{sum_of_delta_angles}.png")
        print(("Successfully created trajectory plot in "
              f"/home/pmueller/llama_traffic/output/{timestamp}.png"))


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

        print("\nPlotting trajectories for all vehicles...")
        vehicle_ids = get_vehicles_for_scenario(self.waymo_scenario)


        for vehicle_id in vehicle_ids:
            print(f"Plotting trajectory for vehicle {vehicle_id}...")
            trajectory = visualize_trajectory(decoded_example=self.waymo_scenario,
                                              specific_id=vehicle_id)
            sum_of_delta_angles = get_sum_of_delta_angles(
                get_coordinates(self.waymo_scenario, vehicle_id))
            trajectory.savefig(f"/home/pmueller/llama_traffic/output/{vehicle_id}_{sum_of_delta_angles}.png")

        print(("Plotting complete.\n"
              "You can find the plots in /home/pmueller/llama_traffic/output/"))

    
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
        coordinates.to_csv(f"/home/pmueller/llama_traffic/output/{timestamp}.scsv")

        print((f"\nThe coordinates of vehicle {vehicle_id} "
              f"have been saved to /home/pmueller/llama_traffic/"
              f"output/{timestamp}.scsv\n"))


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
        
        # Take every 5th coordinate to reduce the number of coordinates
        # and therefore the number of calculations
        
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
        
        print(f'\nThe relative displacement is: {round(displacement*100, 2)} %\n')



    def do_filter_trajectories(self, arg):
        """Filters the trajectories in the loaded scenario in left, right and straight.

        Args:
            arg (str): Arguments for the command.

        Returns:
            str: The path to the folder containing 
            one folder for each direction bucket (see get_direction).
        """

        # Checking if a scenario has been loaded already.
        if not self.scenario_loaded:
            print(("\nNo scenario has been initialized yet!"
                  " \nPlease use 'load_scenario' to load a s"
                  "cenario before calling the 'plot_scenario' command.\n"))
            return

        vehicle_ids = get_vehicles_for_scenario(self.waymo_scenario)

        for vehicle_id in vehicle_ids:
            print(f"Vehicle ID: {vehicle_id}")
            direction = get_direction_of_vehicle(
                self.waymo_scenario,
                get_coordinates(self.waymo_scenario, vehicle_id))


            # \nAngle: {delta_angle_sum}\nDirection: {direction}"
            
            trajectory = visualize_trajectory(decoded_example=self.waymo_scenario,
                                              specific_id=vehicle_id)
            
            trajectory.savefig(f"/home/pmueller/llama_traffic/{direction}/{vehicle_id}.png")


    def do_plot_spline(self, arg):
        """Plots the spline for the given vehicle ID.

        Args:
            arg (str): The vehicle ID for which to plot the spline.
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

        spline = get_spline_for_coordinates(coordinates)
        spline_plot = visualize_coordinates(self.waymo_scenario, spline)
        spline_plot.savefig(f"/home/pmueller/llama_traffic/output/{vehicle_id}_spline.png")


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
        
        print("\n")
        print(*filtered_ids, sep = "\n")
        print("\n")


    def do_get_delta_angles_for_vehicle(self, arg):
        """Returns the delta angles of the trajectory of the given vehicle.
        """

        # Check for empty arguments (no vehicle ID provided)
        if (arg == ""):
            print(("\nYou have provided no ID for the vehicle "
                    "whose trajectory you want to get.\nPlease provide a path!\n"))
            return
        
        vehicle_id = arg.split()[0]
        coordinates = get_coordinates(decoded_example = self.waymo_scenario,
                                      specific_id = vehicle_id)
        angles = get_delta_angles(coordinates)

        output_df = pd.DataFrame(angles, columns = ["Delta Angle"])
        output_df.to_csv(f"/home/pmueller/llama_traffic/output/{vehicle_id}_delta_angles.csv")
        
        print(f"The total heading change is: {angles} degrees!")
        # print(sum(angles))
        
        

    
    def do_total_delta_angle_for_vehicle(self, arg):
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
        angle = get_sum_of_delta_angles(coordinates)
        
        print(f"The total heading change is: {angle} degrees!")


    def do_get_spline(self, arg):
        """Prints the spline for the given vehicle ID.
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

        spline = get_spline_for_coordinates(coordinates)
        print(spline)
        print(len(spline))


    
    def do_get_total_angle_for_vehicle(self, arg):
        """Returns the total angle of the trajectory of the given vehicle.
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
        
        print(f"The total heading change is: {angle} degrees!")


    def do_clear_buckets(self, arg):
        """Clears the buckets for the different directions.

        Args:
            arg (str): No arguments are required.
        """

        print("\nClearing the buckets...")
        for direction in [
            "Left",
            "Right",
            "Straight",
            "Stationary",
            "Right-U-Turn",
            "Left-U-Turn",
            "Straight-Right",
            "Straight-Left"]:
            for file in os.listdir(f"/home/pmueller/llama_traffic/{direction}"):
                os.remove(f"/home/pmueller/llama_traffic/{direction}/{file}")
        print("Successfully cleared the buckets!\n")    


    def do_clear_output_folder(self, arg):
        """Clears the standard output folder.

        Args:
            arg (str): No arguments are required.
        """       

        for file in os.listdir("/home/pmueller/llama_traffic/output"):
            os.remove(f"/home/pmueller/llama_traffic/output/{file}")

        print("\nSuccessfully cleared the output folder!\n")


    def do_get_trajectory_embedding(self, arg):
        """Returns an embedding of the vehicle which's ID has been provided.

        Args:
            arg (str): The vehicle ID for which to get the trajectory embedding.
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
        

        print("\nCalculating the trajectory embedding...")

        vehicle_id = arg.split()[0]
        coordinates = get_coordinates(decoded_example = self.waymo_scenario,
                                      specific_id = vehicle_id)
        trajectory_embedding = get_trajectory_embedding(coordinates)
        print(trajectory_embedding)


    def do_get_bert_embedding(self, arg):
        """Returns an embedding of the given string generated by BERT.

        Args:
            arg (str): _description_
        """
        # Check for empty arguments (no string to encode)
        if (arg == ""):
            print(("\nYou have provided no string to encode.\nPlease provide a string!\n"))
            return
        
        print("\nCalculating the BERT embedding...")
        embedding = get_bert_embedding(arg)
        print(embedding)
        



    # Basic command to exit the shell
    def do_exit(self, arg):
        'Exit the shell: EXIT'
        print("\nExiting the shell...\n")
        return True


    do_EOF = do_exit

if __name__ == '__main__':
    SimpleShell().cmdloop()
