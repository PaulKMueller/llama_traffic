import cmd
import argparse
import os
import yaml

import pandas as pd

from datetime import datetime

from training_dataclass import TrainingData
from trajectory import Trajectory

import numpy as np
import random

from waymo_inform import (
    get_coordinates,
    get_direction_of_vehicle,
    get_total_trajectory_angle,
    get_relative_displacement,
    get_vehicles_for_scenario,
    get_delta_angles,
    get_sum_of_delta_angles,
    get_labeled_trajectories_for_scenario,
    get_zipped_labeled_trajectories_for_all_scenarios_json,
)

from trajectory_generator import (
    create_neural_network,
    infer_with_neural_network,
    train_neural_network,
    create_transformer_model,
    train_transformer_network,
)

from multi_head_attention import get_positional_encoding

from waymo_visualize import (
    visualize_all_agents_smooth,
    create_animation,
    visualize_trajectory,
    visualize_coordinates,
    visualize_map,
    visualize_raw_coordinates_without_scenario,
)
from waymo_initialize import init_waymo

from waymo_utils import (
    get_scenario_list,
    get_spline_for_coordinates,
    get_scenario_index,
)

from trajectory_encoder import get_trajectory_embedding

from bert_encoder import (
    get_bert_embedding,
    get_reduced_bucket_embeddings,
    init_bucket_embeddings,
)

from trajectory_classifier import train_classifier


class SimpleShell(cmd.Cmd):
    prompt = "(waymo_cli) "
    waymo_scenario = {}
    scenario_loaded = False
    scenario_name = ""

    with open("config.yaml", "r") as file:
        config = yaml.safe_load(file)
        scenario_data_folder = config["scenario_data_folder"]

    def arg_parser(self):
        # Initializing the available flags for the different commands.
        parser = argparse.ArgumentParser()
        parser.add_argument("-n", "--name", default="World")
        parser.add_argument("-p", "--path")
        parser.add_argument("--ids", action="store_true")
        parser.add_argument("--index")
        parser.add_argument("--example", "-e", action="store_true")
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
        """Plots the map for the loaded scenario."""

        # Load config file
        with open("config.yaml", "r") as file:
            config = yaml.safe_load(file)
            output_folder = config["output_folder"]

        # Checking if a scenario has been loaded already.
        if not self.scenario_loaded:
            print(
                (
                    "\nNo scenario has been initialized yet!"
                    " \nPlease use 'load_scenario' to load a s"
                    "cenario before calling the 'plot_scenario' command.\n"
                )
            )
            return

        image = visualize_map(self.waymo_scenario)
        image.savefig(
            f"{output_folder}roadgraph_{get_scenario_index(self.scenario_name)}.png"
        )

    def do_visualize_trajectory(self, arg):
        """Plots the trajectory for the given vehicle ID."""

        # Load config file
        with open("config.yaml", "r") as file:
            config = yaml.safe_load(file)
            output_folder = config["output_folder"]

        # Checking if a scenario has been loaded already.
        if not self.scenario_loaded:
            print(
                (
                    "\nNo scenario has been initialized yet!"
                    " \nPlease use 'load_scenario' to load a s"
                    "cenario before calling the 'plot_scenario' command.\n"
                )
            )
            return

        # Check for empty arguments (no ID provided)
        if arg == "":
            print(
                (
                    "\nYou have provided no ID for the vehicle "
                    "that you want to plot.\nPlease provide a path!\n"
                )
            )
            return

        vehicle_id = arg.split()[0]

        coordinates = get_coordinates(
            decoded_example=self.waymo_scenario, specific_id=arg
        )
        trajectory = visualize_coordinates(self.waymo_scenario, coordinates)

        trajectory.savefig(f"{output_folder}{vehicle_id}.png")

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

        # Load config file
        with open("config.yaml", "r") as file:
            config = yaml.safe_load(file)
            scenario_data_folder = config["scenario_data_folder"]
            example_scenario_path = config["example_scenario_path"]

        args = arg.split()

        # Check for empty arguments (no path provided)
        if arg == "":
            print(
                "\nYou have provided no path for the scenario you want to load."
                "\nPlease provide a path!\n"
            )
            return

        elif args[0] == "-e" or args[0] == "--example":
            self.scenario_name = "training_tfexample.tfrecord-00499-of-01000"
            self.waymo_scenario = init_waymo(
                example_scenario_path
            )
            self.scenario_loaded = True
            print("Successfully initialized the example scenario!")
            return
        elif args[0] == "-p" or args[0] == "--path":
            filename = arg.split()[0]
            print(f"Das Szenario wird geladen: {filename}")
            self.waymo_scenario = init_waymo(args[1])
            self.scenario_name = filename
            self.scenario_loaded = True
            print("\nSuccessfully initialized the given scenario!\n")
            return
        elif args[0] == "-i" or args[0] == "--index":
            scenario_name = get_scenario_list()[int(args[1])]
            print(f"The scenario {scenario_name} is being loaded...")
            self.waymo_scenario = init_waymo(scenario_data_folder + scenario_name)
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
            except Exception:
                print(
                    "\nYou have provided no path for the scenario you want to load. Please use the -p flag to indicate the path.\n"
                )

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

        # Load config file
        with open("config.yaml", "r") as file:
            config = yaml.safe_load(file)
            scenario_data_folder = config["scenario_data_folder"]
            output_folder = config["output_folder"]

        parser = self.arg_parser()
        args = parser.parse_args(arg.split())

        if args.index is not None:
            scenario_name = get_scenario_list()[int(args.index)]
            print(scenario_name)
            scenario = init_waymo(scenario_data_folder + scenario_name)
        elif self.scenario_loaded:
            scenario = self.waymo_scenario
        else:
            print(
                (
                    "\nNo scenario has been initialized yet!"
                    " \nPlease use 'load_scenario'"
                    " to load a scenario before calling"
                    " the 'plot_scenario' command.\n"
                )
            )
            return

        if args.ids:
            print("\nPlotting scenario with agent ids...")
            images = visualize_all_agents_smooth(scenario, with_ids=True)
        else:
            print("\nPlotting scenario without agent ids...")
            images = visualize_all_agents_smooth(scenario, with_ids=False)

        scenario_name = get_scenario_index(self.scenario_name)
        anim = create_animation(images[::5])
        anim.save(
            f"/home/pmueller/llama_traffic/output/{scenario_name}.mp4",
            writer="ffmpeg",
            fps=10,
        )
        print(
            f"Successfully created animation in {output_folder}{scenario_name}.mp4!\n"
        )

    def do_list_scenarios(self, arg):
        """Lists all available scenarios in the training folder.
        See: config.yaml under "scenario_data_folder".

        Args:
            arg (str): No arguments are required.
        """

        # Load config file
        with open("config.yaml", "r") as file:
            config = yaml.safe_load(file)
            scenario_data_folder = config["scenario_data_folder"]

        scenarios = get_scenario_list()
        print("\n")
        counter = 0
        for file in scenarios:
            print(f"{counter}: {scenario_data_folder}{file}")
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

        # Load config file
        with open("config.yaml", "r") as file:
            config = yaml.safe_load(file)
            output_folder = config["output_folder"]

        # Checking if a scenario has been loaded already.
        if not self.scenario_loaded:
            print(
                (
                    "\nNo scenario has been initialized yet!"
                    " \nPlease use 'load_scenario' to load a s"
                    "cenario before calling the 'plot_scenario' command.\n"
                )
            )
            return

        # Check for empty arguments (no ID provided)
        if arg == "":
            print(
                (
                    "\nYou have provided no ID for the vehicle "
                    "that you want to plot.\nPlease provide a path!\n"
                )
            )
            return

        vehicle_id = arg.split()[0]
        print(f"\nPlotting vehicle with the ID: {vehicle_id}...")
        images = visualize_all_agents_smooth(
            decoded_example=self.waymo_scenario, with_ids=False, specific_id=vehicle_id
        )
        anim = create_animation(images[::5])
        timestamp = datetime.now()
        anim.save(f"{output_folder}{timestamp}.mp4", writer="ffmpeg", fps=10)
        print(
            ("Successfully created animation in " f"{output_folder}{timestamp}.mp4!\n")
        )

    def do_plot_trajectory(self, arg):
        """Saves a trajectory (represented as a line) for the given vehicle.
        Format should be: get_trajectory <ID>
        Pleas make sure, that you have loaded a scenario before.

        Args:
            arg (string): The vehicle ID for which to plot the trajectory.
        """

        # Load config file
        with open("config.yaml", "r") as file:
            config = yaml.safe_load(file)
            output_folder = config["output_folder"]

        # Checking if a scenario has been loaded already.
        if not self.scenario_loaded:
            print(
                (
                    "\nNo scenario has been initialized yet! \nPlease use 'load_scenario'"
                    " to load a scenario before calling the 'plot_scenario' command.\n"
                )
            )
            return

        # Check for empty arguments (no ID provided)
        if arg == "":
            print(
                (
                    "\nYou have provided no ID for the vehicle "
                    "whose trajectory you want to get.\nPlease provide a path!\n"
                )
            )
            return

        vehicle_id = arg.split()[0]
        print(f"\nPlotting trajectory for vehicle {vehicle_id}...")
        timestamp = datetime.now()
        trajectory = visualize_trajectory(
            decoded_example=self.waymo_scenario, specific_id=vehicle_id
        )
        # trajectory.savefig(f"/home/pmueller/llama_traffic/output/{timestamp}.png")
        sum_of_delta_angles = get_sum_of_delta_angles(
            get_coordinates(self.waymo_scenario, vehicle_id)
        )
        trajectory.savefig(f"{output_folder}{vehicle_id}_{sum_of_delta_angles}.png")
        print(
            (
                "Successfully created trajectory plot in "
                f"{output_folder}{timestamp}.png"
            )
        )

    def do_plot_raw_coordinates_without_scenario(self, arg):
        """Saves a trajectory (represented as a line) for the given vehicle.
        Format should be: get_trajectory <ID>
        Pleas make sure, that you have loaded a scenario before.

        Args:
            arg (string): The vehicle ID for which to plot the trajectory.
        """

        # Load config file
        with open("config.yaml", "r") as file:
            config = yaml.safe_load(file)
            output_folder = config["output_folder"]

        # Checking if a scenario has been loaded already.
        if not self.scenario_loaded:
            print(
                (
                    "\nNo scenario has been initialized yet! \nPlease use 'load_scenario'"
                    " to load a scenario before calling the 'plot_scenario' command.\n"
                )
            )
            return

        # Check for empty arguments (no ID provided)
        if arg == "":
            print(
                (
                    "\nYou have provided no ID for the vehicle "
                    "whose trajectory you want to get.\nPlease provide a path!\n"
                )
            )
            return

        vehicle_id = arg.split()[0]
        print(f"\nPlotting trajectory for vehicle {vehicle_id}...")
        coordinates = get_coordinates(
            decoded_example=self.waymo_scenario, specific_id=vehicle_id
        )
        trajectory = visualize_raw_coordinates_without_scenario(coordinates)
        trajectory.savefig(f"{output_folder}raw_trajectory_{vehicle_id}.png")

        print(
            ("Successfully created trajectory plot in "),
            f"{output_folder}raw_trajectory_{vehicle_id}.png",
        )

    def do_plot_all_trajectories(self, arg):
        """Saves a trajectory (represented as a line) for all vehicles in the scenario.
        Format should be: plot_all_trajectories
        Please make sure, that you have loaded a scenario before.

        Args:
            arg (str): No arguments are required.
        """

        # Load config file
        with open("config.yaml", "r") as file:
            config = yaml.safe_load(file)
            output_folder = config["output_folder"]

        # Checking if a scenario has been loaded already.
        if not self.scenario_loaded:
            print(
                (
                    "\nNo scenario has been initialized yet! \nPlease use 'load_scenario'"
                    " to load a scenario before calling the 'plot_scenario' command.\n"
                )
            )
            return

        print("\nPlotting trajectories for all vehicles...")
        vehicle_ids = get_vehicles_for_scenario(self.waymo_scenario)

        for vehicle_id in vehicle_ids:
            print(f"Plotting trajectory for vehicle {vehicle_id}...")
            trajectory = visualize_trajectory(
                decoded_example=self.waymo_scenario, specific_id=vehicle_id
            )
            sum_of_delta_angles = get_sum_of_delta_angles(
                get_coordinates(self.waymo_scenario, vehicle_id)
            )
            trajectory.savefig(f"{output_folder}{vehicle_id}_{sum_of_delta_angles}.png")

        print(("Plotting complete.\n" f"You can find the plots in {output_folder}"))

    def do_get_coordinates(self, arg):
        """Saves the coordinates of the given vehicle as a csv file.
        Format should be: get_coordinates <ID>
        Pleas make sure, that you have loaded a scenario before.

        Args:
            arg (str): The vehicle ID for which to get the coordinates.
        """

        # Load config file
        with open("config.yaml", "r") as file:
            config = yaml.safe_load(file)
            output_folder = config["output_folder"]

        # Checking if a scenario has been loaded already.
        if not self.scenario_loaded:
            print(
                (
                    "\nNo scenario has been initialized yet! \nPlease use 'load_scenario'"
                    " to load a scenario before calling the 'plot_scenario' command.\n"
                )
            )
            return

        # Check for empty arguments (no ID provided)
        if arg == "":
            print(
                (
                    "\nYou have provided no ID for the vehicle "
                    "whose trajectory you want to get.\nPlease provide a path!\n"
                )
            )
            return

        vehicle_id = arg.split()[0]
        coordinates = get_coordinates(
            decoded_example=self.waymo_scenario, specific_id=vehicle_id
        )

        coordinates.to_csv(f"{output_folder}coordinates_for_{vehicle_id}.scsv")

        print(
            (
                f"\nThe coordinates of vehicle {vehicle_id} "
                f"have been saved to {output_folder}coordinates_for_{vehicle_id}.scsv\n"
            )
        )

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
        if arg == "":
            print(
                (
                    "\nYou have provided no ID for the vehicle "
                    "whose trajectory you want to get.\nPlease provide a path!\n"
                )
            )
            return

        vehicle_id = arg.split()[0]
        coordinates = get_coordinates(
            decoded_example=self.waymo_scenario, specific_id=vehicle_id
        )

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
        if arg == "":
            print(
                (
                    "\nYou have provided no ID for the vehicle "
                    "whose trajectory you want to get.\nPlease provide a path!\n"
                )
            )
            return

        vehicle_id = arg.split()[0]
        displacement = get_relative_displacement(
            decoded_example=self.waymo_scenario,
            coordinates=get_coordinates(
                decoded_example=self.waymo_scenario, specific_id=vehicle_id
            ),
        )

        print(f"\nThe relative displacement is: {round(displacement*100, 2)} %\n")

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
            print(
                (
                    "\nNo scenario has been initialized yet!"
                    " \nPlease use 'load_scenario' to load a s"
                    "cenario before calling the 'plot_scenario' command.\n"
                )
            )
            return

        vehicle_ids = get_vehicles_for_scenario(self.waymo_scenario)

        for vehicle_id in vehicle_ids:
            direction = get_direction_of_vehicle(
                self.waymo_scenario, get_coordinates(self.waymo_scenario, vehicle_id)
            )

            # \nAngle: {delta_angle_sum}\nDirection: {direction}"

            trajectory = visualize_trajectory(
                decoded_example=self.waymo_scenario, specific_id=vehicle_id
            )

            trajectory.savefig(
                f"/home/pmueller/llama_traffic/{direction}/{vehicle_id}.png"
            )

    def do_plot_spline(self, arg):
        """Plots the spline for the given vehicle ID.

        Args:
            arg (str): The vehicle ID for which to plot the spline.
        """

        # Load config file
        with open("config.yaml", "r") as file:
            config = yaml.safe_load(file)
            output_folder = config["output_folder"]

        # Checking if a scenario has been loaded already.
        if not self.scenario_loaded:
            print(
                (
                    "\nNo scenario has been initialized yet! \nPlease use 'load_scenario'"
                    " to load a scenario before calling the 'plot_scenario' command.\n"
                )
            )
            return

        # Check for empty arguments (no ID provided)
        if arg == "":
            print(
                (
                    "\nYou have provided no ID for the vehicle "
                    "whose trajectory you want to get.\nPlease provide a path!\n"
                )
            )
            return

        vehicle_id = arg.split()[0]
        coordinates = get_coordinates(
            decoded_example=self.waymo_scenario, specific_id=vehicle_id
        )

        spline = get_spline_for_coordinates(coordinates)
        spline_plot = visualize_coordinates(self.waymo_scenario, spline)
        spline_plot.savefig(f"{output_folder}{vehicle_id}_spline.png")

    def do_get_vehicles_in_loaded_scenario(self, arg):
        """Prints the IDs of all vehicles in the loaded scenario.

        Args:
            arg (str): No arguments are required.
        """

        # Checking if a scenario has been loaded already.
        if not self.scenario_loaded:
            print(
                (
                    "\nNo scenario has been initialized yet!"
                    " \nPlease use 'load_scenario' to load a s"
                    "cenario before calling the 'plot_scenario' command.\n"
                )
            )
            return

        filtered_ids = get_vehicles_for_scenario(self.waymo_scenario)

        print("\n")
        print(*filtered_ids, sep="\n")
        print("\n")

    def do_get_delta_angles_for_vehicle(self, arg):
        """Returns the delta angles of the trajectory of the given vehicle."""

        # Load config file
        with open("config.yaml", "r") as file:
            config = yaml.safe_load(file)
            output_folder = config["output_folder"]

        # Check for empty arguments (no vehicle ID provided)
        if arg == "":
            print(
                (
                    "\nYou have provided no ID for the vehicle "
                    "whose trajectory you want to get.\nPlease provide a path!\n"
                )
            )
            return

        vehicle_id = arg.split()[0]
        coordinates = get_coordinates(
            decoded_example=self.waymo_scenario, specific_id=vehicle_id
        )
        angles = get_delta_angles(coordinates)

        output_df = pd.DataFrame(angles, columns=["Delta Angle"])
        output_df.to_csv(f"{output_folder}{vehicle_id}_delta_angles.csv")

        print(f"The total heading change is: {angles} degrees!")
        # print(sum(angles))

    def do_total_delta_angle_for_vehicle(self, arg):
        """Returns the aggregated delta angles of the trajectory of the given vehicle."""

        # Check for empty arguments (no coordinates provided)
        if arg == "":
            print(
                (
                    "\nYou have provided no ID for the vehicle "
                    "whose trajectory you want to get.\nPlease provide a path!\n"
                )
            )
            return

        vehicle_id = arg.split()[0]
        coordinates = get_coordinates(
            decoded_example=self.waymo_scenario, specific_id=vehicle_id
        )
        angle = get_sum_of_delta_angles(coordinates)

        print(f"The total heading change is: {angle} degrees!")

    def do_get_spline(self, arg):
        """Prints the spline for the given vehicle ID."""

        # Checking if a scenario has been loaded already.
        if not self.scenario_loaded:
            print(
                (
                    "\nNo scenario has been initialized yet! \nPlease use 'load_scenario'"
                    " to load a scenario before calling the 'plot_scenario' command.\n"
                )
            )
            return

        # Check for empty arguments (no ID provided)
        if arg == "":
            print(
                (
                    "\nYou have provided no ID for the vehicle "
                    "whose trajectory you want to get.\nPlease provide a path!\n"
                )
            )
            return

        vehicle_id = arg.split()[0]

        trajectory = Trajectory(self.waymo_scenario, vehicle_id)
        spline = trajectory.splined_coordinates

        print(spline)

    def do_get_total_angle_for_vehicle(self, arg):
        """Returns the total angle of the trajectory of the given vehicle."""

        # Check for empty arguments (no coordinates provided)
        if arg == "":
            print(
                (
                    "\nYou have provided no ID for the vehicle "
                    "whose trajectory you want to get.\nPlease provide a path!\n"
                )
            )
            return

        vehicle_id = arg.split()[0]
        coordinates = get_coordinates(
            decoded_example=self.waymo_scenario, specific_id=vehicle_id
        )
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
            "Straight-Left",
        ]:
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

    def do_get_scenario_labeled_trajectories(self, arg):
        """Returns a dictionary with the vehicle IDs of the loaded scenario as
        keys and the corresponding trajectories (as numpy arrays of X and Y coordinates) and labels as values.

        Args:
            arg (str): No arguments are required.
        """

        # Load config file
        with open("config.yaml", "r") as file:
            config = yaml.safe_load(file)
            output_folder = config["output_folder"]

        # Checking if a scenario has been loaded already.
        if not self.scenario_loaded:
            print(
                (
                    "\nNo scenario has been initialized yet! \nPlease use 'load_scenario'"
                    " to load a scenario before calling the 'plot_scenario' command.\n"
                )
            )
            return

        print("\nGetting the labeled trajectories...")
        labeled_trajectories = get_labeled_trajectories_for_scenario(
            self.waymo_scenario, self.scenario_name
        )

        # Save the labeled trajectories to a txt file in the output folder
        with open(
            f"{output_folder}{get_scenario_index(self.scenario_name)}_labeled_trajectories.txt",
            "w",
        ) as file:
            file.write(str(labeled_trajectories))

        print("Successfully got the labeled trajectories!\n")

    def do_get_zipped_data(self, arg):
        """Returns a dictionary with the scenario IDs as keys and the corresponding
        labeled trajectories for each vehicle as values.
        'Labeled' in this context refers to the direction buckets that the trajectories
        are sorted into.

        Args:
            arg (str): No arguments are required.
        """

        print("\nGetting the labeled trajectories for all scenarios...")
        get_zipped_labeled_trajectories_for_all_scenarios_json()

        print("Successfully got the labeled trajectories for all scenarios!\n")

    def do_get_trajectory_embedding(self, arg):
        """Returns an embedding of the vehicle which's ID has been provided.

        Args:
            arg (str): The vehicle ID for which to get the trajectory embedding.
        """

        # Checking if a scenario has been loaded already.
        if not self.scenario_loaded:
            print(
                (
                    "\nNo scenario has been initialized yet! \nPlease use 'load_scenario'"
                    " to load a scenario before calling the 'plot_scenario' command.\n"
                )
            )
            return

        # Check for empty arguments (no ID provided)
        if arg == "":
            print(
                (
                    "\nYou have provided no ID for the vehicle "
                    "whose trajectory you want to get.\nPlease provide a path!\n"
                )
            )
            return

        print("\nCalculating the trajectory embedding...")

        vehicle_id = arg.split()[0]
        coordinates = get_coordinates(
            decoded_example=self.waymo_scenario, specific_id=vehicle_id
        )
        trajectory_embedding = get_trajectory_embedding(coordinates)
        print(trajectory_embedding)

    def do_get_bert_embedding(self, arg):
        """Returns an embedding of the given string generated by BERT.

        Args:
            arg (str): _description_
        """
        # Check for empty arguments (no string to encode)
        if arg == "":
            print(
                ("\nYou have provided no string to encode.\nPlease provide a string!\n")
            )
            return

        print("\nCalculating the BERT embedding...")
        embedding = get_bert_embedding(arg)
        print(embedding)
        print(len(embedding[0]))

    def do_get_reduced_bucket_embeddings(self, arg):
        """Returns a BERT embedding for the given word which's dimensionality
        has been reduced to 101 dimensions using PCA.

        Args:
            arg (str): _description_
        """

        reduced_embeddings = get_reduced_bucket_embeddings()
        print(reduced_embeddings)

    def do_get_scenario_index(self, arg):
        """Returns the ID of the loaded scenario.

        Args:
            arg (str): No arguments are required.
        """

        print(
            f"\nThe ID of the loaded scenario is: {get_scenario_index(self.scenario_name)}\n"
        )

    def do_test_trajectory_bucketing(self, arg):
        """Generates the buckets for 20 random trajectories from random scenarios.
        Plot them and save them in the corresponding folders.
        Store the plotted trajectories with the corresponding trajectory bucket,
        the scenario index and the vehicle ID as the file name.

        <bucket_scenarioindex_vehicleid.png>

        Args:
            arg (str): No arguments are required.
        """

        print("\nTesting the trajectory bucketing...")
        scenarios = get_scenario_list()
        number_of_scenarios = len(scenarios)

        # Load config file
        with open("config.yaml", "r") as file:
            config = yaml.safe_load(file)
            scenario_data_folder = config["scenario_data_folder"]
            output_folder = config["output_folder"]

        for i in range(20):
            # Get random scenario
            random_scenario_index = random.randint(0, number_of_scenarios)
            random_scenario = init_waymo(
                scenario_data_folder + scenarios[random_scenario_index]
            )

            # Get random vehicle ID
            vehicles_for_scenario = get_vehicles_for_scenario(random_scenario)
            number_of_vehicles_in_scenario = len(vehicles_for_scenario)
            random_vehicle_id = vehicles_for_scenario[
                random.randint(0, number_of_vehicles_in_scenario - 1)
            ]

            # Get trajectory coordinates
            trajectory_coordinates = get_coordinates(random_scenario, random_vehicle_id)

            # Plot the vehicle trajectory
            visualized_trajectory = visualize_trajectory(
                random_scenario, specific_id=random_vehicle_id
            )

            # Get direction of the chosen vehicle in the chosen scenario
            direction = get_direction_of_vehicle(
                random_scenario, get_coordinates(random_scenario, random_vehicle_id)
            )

            # Safe trajectory with the defined name convention
            visualized_trajectory.savefig(
                f"{output_folder}{direction}_{random_scenario_index}_{random_vehicle_id}.png"
            )

            # Safe trajectory coordinates to csv with same naming convention as trajectory visualization
            trajectory_coordinates.to_csv(
                f"{output_folder}{direction}_{random_scenario_index}_{random_vehicle_id}.scsv"
            )

        print("Successfully prepared the trajectory bucket data for training!\n")

    def do_training_data_length(self, arg):
        """Returns the length of the labeled training data.

        Args:
            arg (str): No arguments required.
        """

        training_data = TrainingData(
            "/home/pmueller/llama_traffic/datasets/zipped_labeled_trajectories.json"
        )
        print(training_data.get_size())

    def do_classification(self, arg):
        """Trains a classification model and tests for its accuracy.

        Args:
            arg (str): No arguments required.
        """
        train_classifier()

    def do_create_neural_network(self, arg):
        """Creates a neural network for the purpose of trajectory prediction.

        Args:
            arg (str): No arguments required.
        """

        create_neural_network()

    def do_infer_with_neural_network(self, arg):
        """Infer with the neural network.

        Args:
            arg (str): Bucket for which to predict embedding.
        """

        # Load config file
        with open("config.yaml", "r") as file:
            config = yaml.safe_load(file)
            output_folder = config["output_folder"]

        # Check for empty arguments (no bucket provided)
        if arg == "":
            print(
                (
                    "\nYou have provided no bucket for which to predict the embedding.\nPlease provide a bucket!\n"
                )
            )
            return

        bucket = arg.split()[0]

        input_data = get_bert_embedding(bucket)
        prediction = infer_with_neural_network(input_data)

        # First 101 dimensions are the predicted x coordinates, the last 101 dimensions are the predicted y coordinates
        x_coords = prediction[0][:101]
        y_coords = prediction[0][101:]

        # Store the coordinates in coordinate dictionary
        coordinates = {"X": x_coords, "Y": y_coords}

        visualize_raw_coordinates_without_scenario(coordinates)
        trajectory = visualize_raw_coordinates_without_scenario(coordinates)
        trajectory.savefig(f"{output_folder}predicted_trajectory.png")

    def do_train_neural_network(self, arg):
        """Train the neural network.

        Args:
            arg (str): No arguments required.
        """

        train_neural_network()

    def do_init_bucket_embeddings(self, arg):
        """Initialize the bucket embeddings.

        Args:
            arg (str): No arguments required.
        """

        print("\nInitializing the bucket embeddings...")
        init_bucket_embeddings()
        print("Successfully initialized the bucket embeddings!\n")

    def do_plot_predicted_trajectory(self, arg):
        """Plot the predicted trajectory for the given bucket.

        Args:
            arg (str): No arguments required.
        """

        # Load config file
        with open("config.yaml", "r") as file:
            config = yaml.safe_load(file)
            output_folder = config["output_folder"]

        # Load predicted trajectory JSON from file
        with open(f"{output_folder}predicted_trajectory.json", "r") as file:
            prediction = file.read()

        # Parse loaded data to numpy array
        prediction = prediction.replace("[", "")
        prediction = prediction.replace("]", "")
        prediction = prediction.replace("\n", "")
        prediction = prediction.split(",")
        prediction = [float(i) for i in prediction]
        prediction = np.array(prediction)

        # Parse loaded data (first 101 dimensions are the predicted x coordinates, the last 101 dimensions are the predicted y coordinates)
        print(prediction.shape)
        x_coordinates = prediction[:101]
        y_coordinates = prediction[101:]
        complete_coordinates = {"X": x_coordinates, "Y": y_coordinates}

        trajectory = visualize_raw_coordinates_without_scenario(complete_coordinates)
        trajectory.savefig(f"{output_folder}predicted_trajectory.png")

        print(
            ("Successfully created trajectory plot in "),
            f"{output_folder}predicted_trajectory.png",
        )

    def do_get_positional_encoding(self, arg):
        """Get the positional encoding for the given bucket.

        Args:
            arg (str): vehicle ID
        """

        # Load config file
        with open("config.yaml", "r") as file:
            config = yaml.safe_load(file)
            output_folder = config["output_folder"]

        # Checking if a scenario has been loaded already.
        if not self.scenario_loaded:
            print(
                (
                    "\nNo scenario has been initialized yet! \nPlease use 'load_scenario'"
                    " to load a scenario before calling the 'plot_scenario' command.\n"
                )
            )
            return

        # Check for empty arguments (no ID provided)
        if arg == "":
            print(
                (
                    "\nYou have provided no ID for the vehicle "
                    "whose trajectory you want to get.\nPlease provide a path!\n"
                )
            )
            return

        vehicle_id = arg.split()[0]

        trajectory = Trajectory(
            decoded_example=self.waymo_scenario, specific_id=vehicle_id
        )

        reshaped_coordinates = np.array(
            [[x, y] for x, y in zip(trajectory.x_coordinates, trajectory.y_coordinates)]
        )

        positional_encoding = get_positional_encoding(reshaped_coordinates)
        print(positional_encoding)
        # Store positional encoding in file
        with open(f"{output_folder}positional_encoding.txt", "w") as file:
            file.write(str(positional_encoding))

    def do_create_transformer_model(self, arg):
        """Create a transformer model.

        Args:
            arg (str): No arguments required.
        """

        print("\nCreating the transformer model...")
        create_transformer_model()
        print("Successfully created the transformer model!\n")

    def do_train_transformer_network(self, arg):
        """Train the transformer network.

        Args:
            arg (str): No arguments required.
        """

        train_transformer_network()

    def do_store_raw_scenario(self, arg):
        """Store the raw scenario in a JSON file.

        Args:
            arg (str): No arguments required.
        """

        # Checking if a scenario has been loaded already.
        if not self.scenario_loaded:
            print(
                (
                    "\nNo scenario has been initialized yet! \nPlease use 'load_scenario'"
                    " to load a scenario before calling the 'plot_scenario' command.\n"
                )
            )
            return

        # Store the raw scenario in a JSON file
        with open(
            "/home/pmueller/llama_traffic/datasets/raw_scenario.json", "w"
        ) as file:
            file.write(str(self.waymo_scenario))

    # Basic command to exit the shell
    def do_exit(self, arg):
        "Exit the shell: EXIT"
        print("\nExiting the shell...\n")
        return True

    do_EOF = do_exit


if __name__ == "__main__":
    SimpleShell().cmdloop()
