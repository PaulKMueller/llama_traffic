import cmd
import argparse
import os
from typing import Tuple
import yaml
import json
import math
import keras
import io

from tqdm import tqdm

import seaborn as sns

import time

from PIL import Image

import wandb
from wandb.keras import WandbMetricsLogger


import tensorflow as tf
from tensorflow.keras.losses import MeanSquaredError

import matplotlib.pyplot as plt
import matplotlib.image as mpimg

from npz_utils import list_vehicle_files_absolute

import pandas as pd

from datetime import datetime

# from training_dataclass import TrainingData
from trajectory import Trajectory

import numpy as np
import random

from npz_trajectory import NpzTrajectory

# from llama_test import get_llama_embeddingK

from cohere_encoder import get_cohere_encoding

from uae_explore import get_uae_encoding

# from voyage_explore import get_voyage_encoding

# from training_dataclass import TrainingData

import torch

from learning.rnns import train_lstm_neural_network

from waymo_inform import (
    create_labeled_ego_trajectories,
    plot_trajectory_by_id,
    get_trajectories_for_text_input,
    visualize_raw_coordinates_without_scenario,
    get_vehicles_for_scenario,
    create_labeled_trajectories_for_scenario,
    create_zipped_labeled_trajectories_for_all_scenarios_json,
    get_labeled_trajectories_for_all_scenarios_json,
)

from learning.trajectory_generator import (
    infer_with_simple_neural_network,
    train_simple_neural_network,
    infer_with_neural_network,
    train_neural_network,
)

from learning.transformer_encoder import (
    positional_encoding,
    PositionalEmbedding,
    GlobalSelfAttention,
    FeedForward,
    EncoderLayer,
    Encoder,
    CrossAttention,
    CausalSelfAttention,
    DecoderLayer,
    Decoder,
    Transformer,
    CustomSchedule,
    train_transformer,
)

from learning.rnns import train_rnn_neural_network

# from learning.multi_head_attention import get_positional_encoding

from waymo_utils import (
    get_scenario_list,
    get_scenario_index,
)

from bert_encoder import get_bert_embedding, init_bucket_embeddings, test_bert_encoding

from learning.trajectory_classifier import train_classifier

from scenario import Scenario


class SimpleShell(cmd.Cmd):
    prompt = "(waymo_cli) "
    loaded_scenario = None
    loaded_trajectory = None
    loaded_npz_trajectory = None

    with open("config.yml", "r") as file:
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

    def do_greet(self, arg: str):
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

    def do_plot_map(self, arg: str):
        """Plots the map for the loaded scenario."""

        # Load config file
        with open("config.yml", "r") as file:
            config = yaml.safe_load(file)
            output_folder = config["output_folder"]

        # Checking if a scenario has been loaded already.
        if not self.scenario_loaded():
            return

        image = self.loaded_scenario.visualize_map()
        image.savefig(
            f"{output_folder}roadgraph_{get_scenario_index(self.loaded_scenario.name)}.png"
        )

    def do_visualize_trajectory(self, arg: str):
        """Plots the trajectory for the given vehicle ID."""

        # Load config file
        with open("config.yml", "r") as file:
            config = yaml.safe_load(file)
            output_folder = config["output_folder"]

        # Checking if a scenario has been loaded already.
        if not self.scenario_loaded():
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
        trajectory = Trajectory(self.loaded_scenario, vehicle_id)

        trajectory_plot = self.loaded_scenario.visualize_coordinates(
            trajectory.coordinates, title=f"Trajectory {vehicle_id}"
        )

        trajectory_plot.savefig(f"{output_folder}{vehicle_id}.png")

    def do_store_full_raw_scenario(self, arg: str):
        # TODO: Check for loaded scenario
        dataset = tf.data.TFRecordDataset(
            self.loaded_scenario.path, compression_type=""
        )
        data = next(dataset.as_numpy_iterator())
        with open("output/raw_scenario.txt", "w") as file:
            file.write(str(data))

    def do_load_scenario(self, arg: str):
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
        with open("config.yml", "r") as file:
            config = yaml.safe_load(file)
            scenario_data_folder = config["scenario_data_folder"]
            example_scenario_path = config["example_scenario_path"]

        args = arg.split()

        # Check for empty arguments (no path provided)
        print("The scenario is being loaded...")
        if arg == "":
            print(
                "\nYou have provided no path for the scenario you want to load."
                "\nPlease provide a path!\n"
            )
            return

        elif args[0] == "-e" or args[0] == "--example":
            self.loaded_scenario = Scenario(example_scenario_path)
            print("Successfully initialized the example scenario!")
            return
        elif args[0] == "-p" or args[0] == "--path":
            filename = args[1]
            self.loaded_scenario = Scenario(scenario_path=filename)
            print("\nSuccessfully initialized the given scenario!\n")
            return
        elif args[0] == "-i" or args[0] == "--index":
            scenario_name = get_scenario_list()[int(args[1])]
            self.loaded_scenario = Scenario(
                scenario_path=scenario_data_folder + scenario_name
            )
            print("\nSuccessfully initialized the given scenario!\n")
            return
        else:
            print(
                """Invalid input, please try again. 
                Use -p to specify the scenario path, -i to specify the scenario index
                or - e to load the example scenario chosen in your config.yml."""
            )

    def do_test_npz_bucketing(self, arg: str):
        npz_files = list_vehicle_files_absolute()
        output = {}

        for file in tqdm(npz_files):
            npz_trajectory = NpzTrajectory(file)
            output[file] = npz_trajectory.direction

        with open("output/npz_bucketing_test.json", "w") as file:
            json.dump(output, file)

    def do_load_npz_trajectory(self, arg: str):
        # vehicle is type 1, pedestrian type 2
        npz_trajectory = NpzTrajectory(
            "/mrtstorage/datasets/tmp/waymo_open_motion_processed/train-2e6/vehicle_a_67582_00003_5913311279.npz"
        )
        # npz_trajectory = NpzTrajectory(
        #     "/home/pmueller/llama_traffic/datasets/npz_test_data/train-2e6/pedestrian_c_77063_00004_7456769354.npz"
        # )

        self.loaded_npz_trajectory = npz_trajectory

        # print(npz_trajectory._gt_marginal)
        x = npz_trajectory._gt_marginal[:, 0]
        y = npz_trajectory._gt_marginal[:, 1]
        # print(x)
        # print(y)
        coordinates = pd.DataFrame({"X": x, "Y": y})
        npz_plot = npz_trajectory.visualize_raw_coordinates_without_scenario(
            coordinates
        )
        # npz_plot.savefig("output/test_npz_plot.png")
        # print(npz_trajectory.get_delta_angles(npz_trajectory.coordinates))
        # print(npz_trajectory.movement_vectors)
        # print(npz_trajectory.get_sum_of_delta_angles())

        # plot = npz_trajectory.plot_marginal_predictions_3d(npz_trajectory.vector_data)

        predictions = np.zeros(npz_trajectory.future_val_marginal.shape)
        prediction_dummy = np.zeros((6, 10, 2))

        # print(type(prediction_dummy))
        # print(npz_trajectory.future_val_marginal.shape)
        # print(npz_trajectory.future_val_marginal)

        # print(predictions.shape)
        # plot = npz_trajectory.plot_marginal_predictions_3d(
        #     vector_data=npz_trajectory.vector_data,
        #     is_available=npz_trajectory.future_val_marginal,
        #     gt_marginal=npz_trajectory.gt_marginal,
        #     predictions=prediction_dummy,
        #     confidences=np.zeros((6,)),
        #     # gt_marginal=npz_trajectory.gt_marginal,
        # )

        npz_trajectory.plot_trajectory()

        print(npz_trajectory.direction)

    def do_create_direction_labeled_npz_dataset(self, arg: str):
        with open("config.yml") as config:
            config = yaml.safe_load(config)
            npz_directory = config["npz_dataset"]
        print("Config read!")
        # output = {}

        trajectory_paths = list_vehicle_files_absolute(npz_directory)
        print("Trajectories listed!")
        with open("output/direction_labeled_npz.json", "w") as file:
            file.write("{")
        for i in tqdm(range(len(trajectory_paths))):
            path = trajectory_paths[i]
            npz_trajectory = NpzTrajectory(path)
            coordinates = list(
                zip(npz_trajectory.coordinates["X"], npz_trajectory.coordinates["Y"])
            )
            local_dict = {
                "Coordinates": coordinates,
                "Direction": npz_trajectory.direction,
            }
            # output[path] = local_dict
            with open("output/direction_labeled_npz.json", "a") as file:
                file.write(f'"{path}": {local_dict},\n')
        with open("output/direction_labeled_npz.json", "a") as file:
            file.write("}")

    def do_get_u_turn_candidates(self, arg: str):
        with open("config.yml") as config:
            config = yaml.safe_load(config)
            npz_directory = config["npz_dataset"]

        output = {}

        trajectory_paths = list_vehicle_files_absolute(npz_directory)
        for i in tqdm(range(len(trajectory_paths))):
            path = trajectory_paths[i]
            npz_trajectory = NpzTrajectory(path)
            delta_angle = npz_trajectory.get_sum_of_delta_angles()
            rel_displacement = npz_trajectory.get_relative_displacement()
            if delta_angle > 130:
                with open("output/potential_u_turns.txt", "a") as file:
                    file.write(
                        f"Path: {path}, Relative Displacement: {rel_displacement}\n"
                    )

    def do_load_trajectory(self, arg: str):
        """Loads the trajectory specified by the loaded scenario and the given vehicle ID.
        This trajectory can then be plotted and used in the CLI.

        Args:
            arg (str): The vehicle ID of the trajectory to load.
        """

        if not self.scenario_loaded():
            return

        vehicle_id = arg.split()[0]

        self.loaded_trajectory = Trajectory(
            self.loaded_scenario, specific_id=vehicle_id
        )

        print(f"The trajectory for vehicle {vehicle_id} has successfully been loaded.")

    def do_print_loaded_trajectory_coordinates(self, arg: str):
        """Prints the splined coordinates of the loaded trajectory.

        Args:
            arg (str): No arguments required.
        """
        print(self.loaded_trajectory.splined_coordinates)

    def do_print_loaded_ego_coordinates(self, arg: str):
        """Prints the ego coordinates using the static method in the Trajectory class for coordinate parsing.

        Args:
            arg (str): No arguments required.
        """

        (
            rotated_coordinates,
            rotated_angle,
            original_starting_x,
            original_starting_y,
        ) = Trajectory.get_rotated_ego_coordinates_from_coordinates(
            self.loaded_trajectory.splined_coordinates
        )
        unrotated_coordinates = Trajectory.get_coordinates_from_rotated_ego_coordinates(
            rotated_coordinates, rotated_angle, original_starting_x, original_starting_y
        )

        print(unrotated_coordinates)

    def do_print_current_raw_scenario(self, arg: str):
        """Prints the current scenario that has been loaded in its decoded form.
        This function is for debugging purposes only.

        """
        print(self.loaded_scenario.data)

    def do_print_roadgraph(self, arg: str):
        """Prints the roadgraph of the loaded scenario."""
        pass
        # print(self.loaded_scenario.data["roadgraph"])

    def do_animate_scenario(self, arg: str):
        """Plots the scenario that has previously been
        loaded with the 'load_scenario' command.

        Args:
            arg (str): No arguments are required.
        """

        # Load config file
        with open("config.yml", "r") as file:
            config = yaml.safe_load(file)
            output_folder = config["output_folder"]

        parser = self.arg_parser()
        args = parser.parse_args(arg.split())
        if self.loaded_scenario is not None:
            anim = self.loaded_scenario.get_animation(args.ids)
            anim.save(
                f"/home/pmueller/llama_traffic/output/{self.loaded_scenario.name}.mp4",
                writer="ffmpeg",
                fps=10,
            )
            print(
                f"Successfully created animation in {output_folder}{self.loaded_scenario.name}.mp4!\n"
            )
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

    def do_print_bert_similarity_to_word(self, arg: str):
        """Returns the cosine similarity between the given text input and the
        different direction buckets.

        Args:
            arg (str): The text input for which to get the similarity.
        """

        # Check for empty arguments (no text input provided)
        if arg == "":
            print(
                (
                    "\nYou have provided no text input for which to get the similarity.\nPlease provide a text input!\n"
                )
            )
            return

        text_input = arg
        input_embedding = get_bert_embedding(text_input)
        bucket_embeddings = init_bucket_embeddings()
        # buckets = [
        #     "Left",
        #     "Right",
        #     "Stationary",
        #     "Straight",
        #     "Straight-Left",
        #     "Straight-Right",
        #     "Right-U-Turn",
        #     "Left-U-Turn",
        # ]

        # for bucket in buckets:
        #     bucket_embeddings[bucket] = get_llama_embedding(bucket)
        similarities = {}

        for bucket, embedding in bucket_embeddings.items():
            # Calculate cosine similarity between input_text and bucket
            similarity = np.dot(input_embedding, np.array(embedding))
            similarities[bucket] = similarity

        print("\n")
        print(*similarities.items(), sep="\n")
        print("\n")

    def do_get_trajectories_for_text_input(self, arg: str):
        """Returns a list of the scenarios that contain the given text input in their name.

        Args:
            arg (str): The text input for which to get the scenarios.
        """

        # Load config file
        with open("config.yml", "r") as file:
            config = yaml.safe_load(file)
            scenario_data_folder = config["scenario_data_folder"]
            output_folder = config["output_folder"]

        # Check for empty arguments (no text input provided)
        if arg == "":
            print(
                (
                    "\nYou have provided no text input for which to get the scenarios.\nPlease provide a text input!\n"
                )
            )
            return

        filtered_ids = get_trajectories_for_text_input(arg)

        # Format output and print the trajectoriy IDs

        print("\n")
        print(f"Number of filtered trajectories: {len(filtered_ids)}")
        print(*filtered_ids, sep="\n")
        print("\n")

        # for index, trajectory_id in enumerate(filtered_ids):
        #     scenario_index = trajectory_id.split("_")[0]
        #     vehicle_id = trajectory_id.split("_")[1]
        #     scenario_path = (
        #         scenario_data_folder + get_scenario_list()[int(scenario_index)]
        #     )
        #     print(f"Scenario path: {scenario_path}")
        #     trajectory_plot = self.loaded_scenario.visualize_trajectory(
        #         specific_id=vehicle_id
        #     )

        #     trajectory_plot.savefig(f"{output_folder}{scenario_index}_{vehicle_id}.png")
        #     # trajectory_plot.close()

        # # List of image paths
        # image_folder = "/home/pmueller/llama_traffic/output/"  # Update this with your image folder path
        # image_files = [
        #     os.path.join(image_folder, f)
        #     for f in os.listdir(image_folder)
        #     if f.endswith(".png")
        # ]

        # # Total number of images
        # total_images = len(image_files)

        # # Define the number of rows and columns you want in your grid
        # num_rows = 3  # Adjust as needed
        # num_cols = math.ceil(total_images / num_rows)

        # # Create a figure and a set of subplots
        # fig, axs = plt.subplots(num_rows, num_cols, figsize=(20, 10))

        # # Flatten the axis array for easy iteration
        # axs = axs.flatten()

        # # Loop through images and add them to the subplots
        # for idx, img_path in enumerate(image_files):
        #     img = mpimg.imread(img_path)
        #     axs[idx].imshow(img)
        #     axs[idx].axis("off")  # Hide axis

        # # Hide any unused subplots
        # for ax in axs[total_images:]:
        #     ax.axis("off")

        # plt.tight_layout()

        # # Save total plot
        # plt.savefig("/home/pmueller/llama_traffic/output/total.png")

    def do_plot_trajectory_by_id(self, arg):
        # Load config file
        with open("config.yml", "r") as file:
            config = yaml.safe_load(file)
            scenario_data_folder = config["scenario_data_folder"]
            output_folder = config["output_folder"]

        # Check for empty arguments (no ID provided)
        if arg == "":
            print(
                (
                    "\nYou have provided no ID for the vehicle "
                    "that you want to plot.\nPlease provide a path!\n"
                )
            )
            return

        plot_trajectory_by_id(arg)

    def do_list_scenarios(self, arg: str):
        """Lists all available scenarios in the training folder.
        See: config.yml under "scenario_data_folder".

        Args:
            arg (str): No arguments are required.
        """

        # Load config file
        with open("config.yml", "r") as file:
            config = yaml.safe_load(file)
            scenario_data_folder = config["scenario_data_folder"]

        scenarios = get_scenario_list()
        print("\n")
        counter = 0
        for file in scenarios:
            print(f"{counter}: {scenario_data_folder}{file}")
            counter += 1

        print("\n")

    def do_plot_trajectories_for_text_input(self, arg: str):
        # Check for empty arguments (no ID provided)
        if arg == "":
            print(
                (
                    "\nYou have provided no text input."
                    "\nPlease input a verbal description of the type of trajectory which you want to plot.\n"
                )
            )
            return

        filtered_ids = get_trajectories_for_text_input(arg)
        for id in filtered_ids:
            plot_trajectory_by_id(id)

    def do_animate_vehicle(self, arg: str):
        """Creates a mp4 animation of the trajectory of
        the given vehicle for the loaded scenario.
        Format should be: animate_vehicle <ID>
        Pleas make sure, that you have loaded a scenario before.

        Args:
            arg (str): the vehicle ID for which to plot the trajectory.
        """

        # Load config file
        with open("config.yml", "r") as file:
            config = yaml.safe_load(file)
            output_folder = config["output_folder"]

        # Checking if a scenario has been loaded already.
        if not self.scenario_loaded():
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
        images = self.loaded_scenario.visualize_all_agents_smooth(
            with_ids=False,
            specific_id=vehicle_id,
        )
        anim = self.loaded_scenario.create_animation(images[::1])
        timestamp = datetime.now()
        anim.save(f"{output_folder}{timestamp}.mp4", writer="ffmpeg", fps=10)
        print(
            ("Successfully created animation in " f"{output_folder}{timestamp}.mp4!\n")
        )

    def do_plot_trajectory(self, arg: str):
        """Saves a trajectory (represented as a line) for the given vehicle.
        Format should be: get_trajectory <ID>
        Pleas make sure, that you have loaded a scenario before.

        Args:
            arg (string): The vehicle ID for which to plot the trajectory.
        """

        # Load config file
        with open("config.yml", "r") as file:
            config = yaml.safe_load(file)
            output_folder = config["output_folder"]

        # Checking if a scenario has been loaded already.
        if not self.scenario_loaded():
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

        trajectory_plot = self.loaded_scenario.visualize_trajectory(
            specific_id=vehicle_id
        )
        trajectory_plot.savefig(f"{output_folder}{vehicle_id}.png")
        print(
            (
                "Successfully created trajectory plot in "
                f"{output_folder}{timestamp}.png"
            )
        )

    def do_plot_raw_coordinates_without_scenario(self, arg: str):
        """Saves a trajectory (represented as a line) for the given vehicle.
        Format should be: get_trajectory <ID>
        Pleas make sure, that you have loaded a scenario before.

        Args:
            arg (string): The vehicle ID for which to plot the trajectory.
        """

        # Load config file
        with open("config.yml", "r") as file:
            config = yaml.safe_load(file)
            output_folder = config["output_folder"]

        # Checking if a scenario has been loaded already.
        if not self.scenario_loaded():
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
        trajectory = Trajectory(self.loaded_scenario, vehicle_id)
        trajectory_plot = trajectory.visualize_raw_coordinates_without_scenario(
            trajectory.splined_coordinates
        )
        trajectory_plot.savefig(f"{output_folder}raw_trajectory_{vehicle_id}.png")

        print(
            ("Successfully created trajectory plot in "),
            f"{output_folder}raw_trajectory_{vehicle_id}.png",
        )

    def do_plot_ego_coordinates_for(self, arg: str):
        """Plots the ego coordinates of the vehicle for which the ID was given as an argument.

        Args:
            arg (str): The vehicle ID.
        """

        # Load config file
        with open("config.yml", "r") as file:
            config = yaml.safe_load(file)
            output_folder = config["output_folder"]

        # Checking if a scenario has been loaded already.
        if not self.scenario_loaded():
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
        trajectory = Trajectory(self.loaded_scenario, vehicle_id)

        trajectory_plot = trajectory.visualize_raw_coordinates_without_scenario(
            trajectory.ego_coordinates
        )
        trajectory_plot.savefig(f"{output_folder}raw_ego_trajectory_{vehicle_id}.png")

    def do_plot_rotated_coordinates_for(self, arg: str):
        """Plots the ego coordinates of the vehicle for which the ID was given as an argument.

        Args:
            arg (str): The vehicle ID.
        """

        # Load config file
        with open("config.yml", "r") as file:
            config = yaml.safe_load(file)
            output_folder = config["output_folder"]

        # Checking if a scenario has been loaded already.
        if not self.scenario_loaded():
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
        trajectory = Trajectory(self.loaded_scenario, vehicle_id)

        trajectory_plot = trajectory.visualize_raw_coordinates_without_scenario(
            trajectory.rotated_coordinates
        )
        trajectory_plot.savefig(
            f"{output_folder}raw_rotated_trajectory_{vehicle_id}.png"
        )

    def do_plot_architecture_example_picture(self, arg: str):
        """Plots a comparison of the coordinates before, during and after their transformation to splined, rotated, ego trajectories.
        It does so for the vehicle which's ID has been provided in the loaded scenario.

        Args:
            arg (str): The vehicle ID.
        """

        # Load config file
        with open("config.yml", "r") as file:
            config = yaml.safe_load(file)
            output_folder = config["output_folder"]

        # Checking if a scenario has been loaded already.
        if not self.scenario_loaded():
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
        trajectory = Trajectory(self.loaded_scenario, vehicle_id)

        # Set the style of the plot using Seaborn
        sns.set_theme(style="whitegrid")

        # Plot settings
        plot_settings = {"markersize": 6, "linewidth": 2, "marker": "o"}

        # Original Coordinates
        plt.plot(
            trajectory.coordinates["X"],
            trajectory.coordinates["Y"],
            "r-",  # blue line
            **plot_settings,
        )

        plt.title("Trajectory coordinates")

        # Adjust layout and save the figure
        plt.tight_layout()
        plt.savefig("architecure_example_picture.png")
        plt.close()

    def do_plot_compared_raw_coordinates(self, arg: str):
        """Plots a comparison of the coordinates before, during and after their transformation to splined, rotated, ego trajectories.
        It does so for the vehicle which's ID has been provided in the loaded scenario.

        Args:
            arg (str): The vehicle ID.
        """

        # Load config file
        with open("config.yml", "r") as file:
            config = yaml.safe_load(file)
            output_folder = config["output_folder"]

        # Checking if a scenario has been loaded already.
        if not self.scenario_loaded():
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
        trajectory = Trajectory(self.loaded_scenario, vehicle_id)

        # Set the style of the plot using Seaborn
        sns.set_theme(style="whitegrid")

        # Create figure for the 2x2 grid of subplots
        fig, axs = plt.subplots(2, 2, figsize=(12, 12))

        # Plot settings
        plot_settings = {"markersize": 6, "linewidth": 2, "marker": "o"}

        # Original Coordinates
        axs[0, 0].plot(
            trajectory.coordinates["X"],
            trajectory.coordinates["Y"],
            "b-",  # blue line
            **plot_settings,
        )
        axs[0, 0].set_title("Original Coordinates")
        axs[0, 0].set_xlabel("X Coordinate")
        axs[0, 0].set_ylabel("Y Coordinate")

        # Splined Coordinates
        axs[0, 1].plot(
            trajectory.splined_coordinates["X"],
            trajectory.splined_coordinates["Y"],
            "g-",  # green line
            **plot_settings,
        )
        axs[0, 1].set_title("Splined Coordinates")
        axs[0, 1].set_xlabel("X Coordinate")
        axs[0, 1].set_ylabel("Y Coordinate")

        # Ego Coordinates
        axs[1, 0].plot(
            trajectory.ego_coordinates["X"],
            trajectory.ego_coordinates["Y"],
            "r-",  # red line
            **plot_settings,
        )
        axs[1, 0].set_title("Ego Coordinates")
        axs[1, 0].set_xlabel("X Coordinate")
        axs[1, 0].set_ylabel("Y Coordinate")

        # Rotated Ego Coordinates
        axs[1, 1].plot(
            trajectory.rotated_coordinates["X"],
            trajectory.rotated_coordinates["Y"],
            "m-",  # magenta line
            **plot_settings,
        )
        axs[1, 1].set_title("Rotated Ego Coordinates")
        axs[1, 1].set_xlabel("X Coordinate")
        axs[1, 1].set_ylabel("Y Coordinate")

        # Adjust layout and save the figure
        plt.tight_layout()
        plt.savefig(f"{output_folder}compared_coordinates_for_{vehicle_id}.png")

    def do_plot_all_trajectories(self, arg: str):
        """Saves a trajectory (represented as a line) for all vehicles in the scenario.
        Format should be: plot_all_trajectories
        Please make sure, that you have loaded a scenario before.

        Args:
            arg (str): No arguments are required.
        """

        # Load config file
        with open("config.yml", "r") as file:
            config = yaml.safe_load(file)
            output_folder = config["output_folder"]

        # Checking if a scenario has been loaded already.
        if not self.scenario_loaded():
            return

        print("\nPlotting trajectories for all vehicles...")
        vehicle_ids = get_vehicles_for_scenario(self.loaded_scenario.data)

        for vehicle_id in vehicle_ids:
            print(f"Plotting trajectory for vehicle {vehicle_id}...")
            trajectory_plot = self.loaded_scenario.visualize_trajectory(
                specific_id=vehicle_id
            )
            trajectory_plot.savefig(f"{output_folder}{vehicle_id}.png")

        print(("Plotting complete.\n" f"You can find the plots in {output_folder}"))

    def do_save_coordinates_for_vehicle(self, arg: str):
        """Saves the coordinates of the given vehicle as a csv file.
        Format should be: get_coordinates <ID>
        Pleas make sure, that you have loaded a scenario before.

        Args:
            arg (str): The vehicle ID for which to get the coordinates.
        """

        # Load config file
        with open("config.yml", "r") as file:
            config = yaml.safe_load(file)
            output_folder = config["output_folder"]

        # Checking if a scenario has been loaded already.
        if not self.scenario_loaded():
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
        trajectory = Trajectory(self.loaded_scenario, vehicle_id)
        trajectory.coordinates.to_csv(
            f"{output_folder}coordinates_for_{vehicle_id}.scsv"
        )

        print(
            (
                f"\nThe coordinates of vehicle {vehicle_id} "
                f"have been saved to {output_folder}coordinates_for_{vehicle_id}.scsv\n"
            )
        )

    def do_print_viewport(self, arg: str):
        """Prints the viewport of the loaded scenario."""

        print(self.loaded_scenario.viewport)

    def do_save_normalized_coordinates(self, arg: str):
        """Saves the normalized coordinates of the given vehicle as a csv file.
        Format should be: get_normalized_coordinates <ID>
        Pleas make sure, that you have loaded a scenario before.

        Args:
            arg (str): The vehicle ID for which to get the normalized coordinates.
        """

        # Load config file
        with open("config.yml", "r") as file:
            config = yaml.safe_load(file)
            output_folder = config["output_folder"]

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
        trajectory = Trajectory(self.loaded_scenario, vehicle_id)
        trajectory.normalized_splined_coordinates.to_csv(
            f"{output_folder}/normalized_coordinates_for_{vehicle_id}.scsv"
        )

        print(
            (
                f"\nThe normalized coordinates of vehicle {vehicle_id} "
                f"have been saved to normalized_coordinates_for_{vehicle_id}.scsv\n"
            )
        )

    def do_print_direction(self, arg: str):
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
        trajectory = Trajectory(self.loaded_scenario, vehicle_id)

        print(f"\n{trajectory.direction}!\n")

    def do_print_displacement_for_vehicle(self, arg: str):
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
        trajectory = Trajectory(self.loaded_scenario, vehicle_id)

        print(
            f"\nThe relative displacement is: {round(trajectory.relative_displacement*100, 2)} %\n"
        )

    def do_filter_trajectories(self, arg: str):
        """Filters the trajectories in the loaded scenario in left, right and straight.

        Args:
            arg (str): Arguments for the command.

        Returns:
            str: The path to the folder containing
            one folder for each direction bucket (see get_direction).
        """

        # Checking if a scenario has been loaded already.
        if not self.scenario_loaded():
            return

        vehicle_ids = get_vehicles_for_scenario(self.loaded_scenario.data)

        for vehicle_id in vehicle_ids:
            trajectory = Trajectory(self.loaded_scenario, vehicle_id)

            # \nAngle: {delta_angle_sum}\nDirection: {direction}"

            trajectory_plot = self.loaded_scenario.visualize_trajectory(
                specific_id=vehicle_id
            )

            trajectory_plot.savefig(
                f"/home/pmueller/llama_traffic/{trajectory.direction}/{vehicle_id}.png"
            )

    def do_plot_spline(self, arg: str):
        """Plots the spline for the given vehicle ID.

        Args:
            arg (str): The vehicle ID for which to plot the spline.
        """

        # Load config file
        with open("config.yml", "r") as file:
            config = yaml.safe_load(file)
            output_folder = config["output_folder"]

        # Checking if a scenario has been loaded already.
        if not self.scenario_loaded():
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

        trajectory = Trajectory(self.loaded_scenario, vehicle_id)
        trajectory_plot = self.loaded_scenario.visualize_coordinates(
            trajectory.splined_coordinates
        )
        trajectory_plot.savefig(f"{output_folder}{vehicle_id}_spline.png")

    def do_vehicles_in_loaded_scenario(self, arg: str):
        """Prints the IDs of all vehicles in the loaded scenario.

        Args:
            arg (str): No arguments are required.
        """

        # Checking if a scenario has been loaded already.
        if not self.scenario_loaded():
            return

        filtered_ids = get_vehicles_for_scenario(self.loaded_scenario.data)

        print("\n")
        print(*filtered_ids, sep="\n")
        print("\n")

    def do_plot_all_maps(self, arg: str):
        # Load config file
        with open("config.yml", "r") as file:
            config = yaml.safe_load(file)
            output_folder = config["output_folder"]
            scenario_data_folder = config["scenario_data_folder"]

        scenarios = get_scenario_list()

        for scenario in scenarios:
            print(scenario)
            scenario_obj = Scenario(scenario_data_folder + scenario)
            image = scenario_obj.visualize_map()

            image.savefig(
                f"{output_folder}roadgraph_{get_scenario_index(scenario_obj.name)}.png"
            )

    def do_save_delta_angles_for_vehicle(self, arg: str):
        """Returns the delta angles of the trajectory of the given vehicle."""

        # Load config file
        with open("config.yml", "r") as file:
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
        trajectory = Trajectory(self.loaded_scenario, vehicle_id)
        coordinates = trajectory.splined_coordinates
        angles = trajectory.get_delta_angles(coordinates)

        output_df = pd.DataFrame(angles, columns=["Delta Angle"])
        output_df.to_csv(f"{output_folder}{vehicle_id}_delta_angles.csv")

        print(f"The total heading change is: {angles} degrees!")
        # print(sum(angles))

    def do_total_delta_angle_for_vehicle(self, arg: str):
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
        trajectory = Trajectory(self.loaded_scenario, vehicle_id)

        print(f"The total heading change is: {trajectory.sum_of_delta_angles} degrees!")

    def do_print_spline(self, arg: str):
        """Prints the spline for the given vehicle ID."""

        # Checking if a scenario has been loaded already.
        if not self.scenario_loaded():
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

        trajectory = Trajectory(self.loaded_scenario, vehicle_id)
        spline = trajectory.splined_coordinates

        # Store spline from dataframe
        spline.to_csv(f"/home/pmueller/llama_traffic/spline_{vehicle_id}.csv")

        print(spline)

    def do_clear_buckets(self, arg: str):
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

    def do_clear_output_folder(self, arg: str):
        """Clears the standard output folder.

        Args:
            arg (str): No arguments are required.
        """

        for file in os.listdir("/home/pmueller/llama_traffic/output"):
            os.remove(f"/home/pmueller/llama_traffic/output/{file}")

        print("\nSuccessfully cleared the output folder!\n")

    def do_create_labeled_trajectories_for_loaded_scenario(self, arg: str):
        """Returns a dictionary with the vehicle IDs of the loaded scenario as
        keys and the corresponding trajectories (as numpy arrays of X and Y coordinates) and labels as values.

        Args:
            arg (str): No arguments are required.
        """

        # Load config file
        with open("config.yml", "r") as file:
            config = yaml.safe_load(file)
            output_folder = config["output_folder"]

        # Checking if a scenario has been loaded already.
        if not self.scenario_loaded():
            return

        print("\nGetting the labeled trajectories...")
        labeled_trajectories = create_labeled_trajectories_for_scenario(
            self.loaded_scenario
        )

        # Save the labeled trajectories to a txt file in the output folder
        with open(
            f"{output_folder}{get_scenario_index(self.loaded_scenario.name)}_labeled_trajectories.json",
            "w",
        ) as file:
            json.dump(labeled_trajectories, file, indent=4)

        print("Successfully got the labeled trajectories!\n")

    def do_create_labeled_training_data(self, arg: str):
        """Returns a dictionary of training data.

        Args:
            arg (str): No arguments required.
        """

        get_labeled_trajectories_for_all_scenarios_json()

    def do_create_zipped_training_data(self, arg: str):
        """Returns a dictionary with the scenario IDs as keys and the corresponding
        labeled trajectories for each vehicle as values.
        'Labeled' in this context refers to the direction buckets that the trajectories
        are sorted into.

        Args:
            arg (str): No arguments are required.
        """

        print("\nGetting the labeled trajectories for all scenarios...")
        create_zipped_labeled_trajectories_for_all_scenarios_json()

        print("Successfully got the labeled trajectories for all scenarios!\n")

    def do_create_labeled_ego_trajectories(self, arg: str):
        """Creates a json file with the training data as labeled ego trajectories. This means,
        that the trajectories all start at (0, 0) and are rotated to point to the right side.

        Args:
            arg (str): No arguments required.
        """

        create_labeled_ego_trajectories()

    def do_embed_with_bert(self, arg: str):
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

    def do_print_llama_embedding(self, arg: str):
        """Returns the llama embedding for the given text input."""
        input_text = arg
        embedding = get_llama_embedding(input_text)

        print(embedding)

    def do_embed_cohere_similarities_with_word(self, arg: str):
        """Returns the similarities of the available buckets to the given text input.

        Args:
            arg (str): The text input for which to get the similarities.
        """
        bucket_similarities = get_cohere_encoding(arg)
        print(bucket_similarities)

    def do_test_cohere_embedding_similarities(self, arg):
        right_u_turn = [
            "Rightward complete reversal",
            "180-degree turn to the right",
            "Clockwise U-turn",
            "Right circular turnaround",
            "Right-hand loopback",
            "Right flip turn",
            "Full right pivot",
            "Right about-face",
            "Rightward return turn",
            "Rightward reversing curve",
        ]
        left_u_turn = [
            "Leftward complete reversal",
            "180-degree turn to the left",
            "Counterclockwise U-turn",
            "Left circular turnaround",
            "Left-hand loopback",
            "Left flip turn",
            "Full left pivot",
            "Left about-face",
            "Leftward return turn",
            "Leftward reversing curve",
        ]
        stationary = [
            "At a standstill",
            "Motionless",
            "Unmoving",
            "Static position",
            "Immobilized",
            "Not in motion",
            "Fixed in place",
            "Idle",
            "Inert",
            "Anchored",
        ]
        right = [
            "Rightward",
            "To the right",
            "Right-hand side",
            "Starboard",
            "Rightward direction",
            "Clockwise direction",
            "Right-leaning",
            "Rightward bound",
            "Bearing right",
            "Veering right",
        ]
        left = [
            "Leftward",
            "To the left",
            "Left-hand side",
            "Port",
            "Leftward direction",
            "Counterclockwise direction",
            "Left-leaning",
            "Leftward bound",
            "Bearing left",
            "Veering left",
        ]
        straight_right = [
            "Straight then right",
            "Forward followed by a right turn",
            "Proceed straight, then veer right",
            "Continue straight before turning right",
            "Advance straight, then bear right",
            "Go straight, then curve right",
            "Head straight, then pivot right",
            "Move straight, then angle right",
            "Straight-line, followed by a right deviation",
            "Directly ahead, then a rightward shift",
        ]
        straight_left = [
            "Straight then left",
            "Forward followed by a left turn",
            "Proceed straight, then veer left",
            "Continue straight before turning left",
            "Advance straight, then bear left",
            "Go straight, then curve left",
            "Head straight, then pivot left",
            "Move straight, then angle left",
            "Straight-line, followed by a left deviation",
            "Directly ahead, then a leftward shift",
        ]
        straight = [
            "Directly ahead",
            "Forward",
            "Straightforward",
            "In a straight line",
            "Linearly",
            "Unswervingly",
            "Onward",
            "Direct path",
            "True course",
            "Non-curving path",
        ]

        bucket_synonym_lists = [
            right_u_turn,
            left_u_turn,
            stationary,
            right,
            left,
            straight_right,
            straight_left,
            straight,
        ]

        buckets = [
            "Right-U-Turn",
            "Left-U-Turn",
            "Stationary",
            "Right",
            "Left",
            "Straight-Right",
            "Straight-Left",
            "Straight",
        ]

        output = {}

        for index, synonym_list in enumerate(bucket_synonym_lists):
            current_bucket = buckets[index]
            if current_bucket not in output:
                output[current_bucket] = {}
            for synonym in synonym_list:
                bucket_similarities = get_cohere_encoding(synonym)
                print(synonym)
                output[buckets[index]][synonym] = bucket_similarities
                print(bucket_similarities)
                print()
        with open("output/cohere_embedding_test.json", "w") as file:
            json.dump(output, file, indent=4)

    def do_test_uae_embedding_similarities(self, arg):
        right_u_turn = [
            "Rightward complete reversal",
            "180-degree turn to the right",
            "Clockwise U-turn",
            "Right circular turnaround",
            "Right-hand loopback",
            "Right flip turn",
            "Full right pivot",
            "Right about-face",
            "Rightward return turn",
            "Rightward reversing curve",
        ]
        left_u_turn = [
            "Leftward complete reversal",
            "180-degree turn to the left",
            "Counterclockwise U-turn",
            "Left circular turnaround",
            "Left-hand loopback",
            "Left flip turn",
            "Full left pivot",
            "Left about-face",
            "Leftward return turn",
            "Leftward reversing curve",
        ]
        stationary = [
            "At a standstill",
            "Motionless",
            "Unmoving",
            "Static position",
            "Immobilized",
            "Not in motion",
            "Fixed in place",
            "Idle",
            "Inert",
            "Anchored",
        ]
        right = [
            "Rightward",
            "To the right",
            "Right-hand side",
            "Starboard",
            "Rightward direction",
            "Clockwise direction",
            "Right-leaning",
            "Rightward bound",
            "Bearing right",
            "Veering right",
        ]
        left = [
            "Leftward",
            "To the left",
            "Left-hand side",
            "Port",
            "Leftward direction",
            "Counterclockwise direction",
            "Left-leaning",
            "Leftward bound",
            "Bearing left",
            "Veering left",
        ]
        straight_right = [
            "Straight then right",
            "Forward followed by a right turn",
            "Proceed straight, then veer right",
            "Continue straight before turning right",
            "Advance straight, then bear right",
            "Go straight, then curve right",
            "Head straight, then pivot right",
            "Move straight, then angle right",
            "Straight-line, followed by a right deviation",
            "Directly ahead, then a rightward shift",
        ]
        straight_left = [
            "Straight then left",
            "Forward followed by a left turn",
            "Proceed straight, then veer left",
            "Continue straight before turning left",
            "Advance straight, then bear left",
            "Go straight, then curve left",
            "Head straight, then pivot left",
            "Move straight, then angle left",
            "Straight-line, followed by a left deviation",
            "Directly ahead, then a leftward shift",
        ]
        straight = [
            "Directly ahead",
            "Forward",
            "Straightforward",
            "In a straight line",
            "Linearly",
            "Unswervingly",
            "Onward",
            "Direct path",
            "True course",
            "Non-curving path",
        ]

        bucket_synonym_lists = [
            right_u_turn,
            left_u_turn,
            stationary,
            right,
            left,
            straight_right,
            straight_left,
            straight,
        ]

        buckets = [
            "Right-U-Turn",
            "Left-U-Turn",
            "Stationary",
            "Right",
            "Left",
            "Straight-Right",
            "Straight-Left",
            "Straight",
        ]

        output = {}

        for index, synonym_list in enumerate(bucket_synonym_lists):
            current_bucket = buckets[index]
            if current_bucket not in output:
                output[current_bucket] = {}
            for synonym in synonym_list:
                bucket_similarities = get_uae_encoding(synonym)
                print(synonym)
                output[buckets[index]][synonym] = bucket_similarities
                print(bucket_similarities)
                print()
        with open("output/uae_embedding_test.json", "w") as file:
            json.dump(str(output), file, indent=4)

    # def do_test_voyage_embedding_similarities(self, arg):
    #     right_u_turn = [
    #         "Rightward complete reversal",
    #         "180-degree turn to the right",
    #         "Clockwise U-turn",
    #         "Right circular turnaround",
    #         "Right-hand loopback",
    #         "Right flip turn",
    #         "Full right pivot",
    #         "Right about-face",
    #         "Rightward return turn",
    #         "Rightward reversing curve",
    #     ]
    #     left_u_turn = [
    #         "Leftward complete reversal",
    #         "180-degree turn to the left",
    #         "Counterclockwise U-turn",
    #         "Left circular turnaround",
    #         "Left-hand loopback",
    #         "Left flip turn",
    #         "Full left pivot",
    #         "Left about-face",
    #         "Leftward return turn",
    #         "Leftward reversing curve",
    #     ]
    #     stationary = [
    #         "At a standstill",
    #         "Motionless",
    #         "Unmoving",
    #         "Static position",
    #         "Immobilized",
    #         "Not in motion",
    #         "Fixed in place",
    #         "Idle",
    #         "Inert",
    #         "Anchored",
    #     ]
    #     right = [
    #         "Rightward",
    #         "To the right",
    #         "Right-hand side",
    #         "Starboard",
    #         "Rightward direction",
    #         "Clockwise direction",
    #         "Right-leaning",
    #         "Rightward bound",
    #         "Bearing right",
    #         "Veering right",
    #     ]
    #     left = [
    #         "Leftward",
    #         "To the left",
    #         "Left-hand side",
    #         "Port",
    #         "Leftward direction",
    #         "Counterclockwise direction",
    #         "Left-leaning",
    #         "Leftward bound",
    #         "Bearing left",
    #         "Veering left",
    #     ]
    #     straight_right = [
    #         "Straight then right",
    #         "Forward followed by a right turn",
    #         "Proceed straight, then veer right",
    #         "Continue straight before turning right",
    #         "Advance straight, then bear right",
    #         "Go straight, then curve right",
    #         "Head straight, then pivot right",
    #         "Move straight, then angle right",
    #         "Straight-line, followed by a right deviation",
    #         "Directly ahead, then a rightward shift",
    #     ]
    #     straight_left = [
    #         "Straight then left",
    #         "Forward followed by a left turn",
    #         "Proceed straight, then veer left",
    #         "Continue straight before turning left",
    #         "Advance straight, then bear left",
    #         "Go straight, then curve left",
    #         "Head straight, then pivot left",
    #         "Move straight, then angle left",
    #         "Straight-line, followed by a left deviation",
    #         "Directly ahead, then a leftward shift",
    #     ]
    #     straight = [
    #         "Directly ahead",
    #         "Forward",
    #         "Straightforward",
    #         "In a straight line",
    #         "Linearly",
    #         "Unswervingly",
    #         "Onward",
    #         "Direct path",
    #         "True course",
    #         "Non-curving path",
    #     ]

    #     bucket_synonym_lists = [
    #         right_u_turn,
    #         left_u_turn,
    #         stationary,
    #         right,
    #         left,
    #         straight_right,
    #         straight_left,
    #         straight,
    #     ]

    #     buckets = [
    #         "Right-U-Turn",
    #         "Left-U-Turn",
    #         "Stationary",
    #         "Right",
    #         "Left",
    #         "Straight-Right",
    #         "Straight-Left",
    #         "Straight",
    #     ]

    #     output = {}

    #     for index, synonym_list in enumerate(bucket_synonym_lists):
    #         current_bucket = buckets[index]
    #         if current_bucket not in output:
    #             output[current_bucket] = {}
    #         for synonym in synonym_list:
    #             bucket_similarities = get_voyage_encoding(synonym)
    #             print(synonym)
    #             output[buckets[index]][synonym] = bucket_similarities
    #             print(bucket_similarities)
    #             print()
    #     with open("output/voyage_embedding_test.json", "w") as file:
    #         json.dump(str(output), file, indent=4)

    # def do_print_scenario_index(self, arg: str):
    #     """Returns the ID of the loaded scenario.

    #     Args:
    #         arg (str): No arguments are required.
    #     """

    #     print(
    #         f"\nThe ID of the loaded scenario is: {get_scenario_index(self.loaded_scenario.name)}\n"
    #     )

    def do_test_bert_encoding(self, arg):
        right_u_turn = [
            "Rightward complete reversal",
            "180-degree turn to the right",
            "Clockwise U-turn",
            "Right circular turnaround",
            "Right-hand loopback",
            "Right flip turn",
            "Full right pivot",
            "Right about-face",
            "Rightward return turn",
            "Rightward reversing curve",
        ]
        left_u_turn = [
            "Leftward complete reversal",
            "180-degree turn to the left",
            "Counterclockwise U-turn",
            "Left circular turnaround",
            "Left-hand loopback",
            "Left flip turn",
            "Full left pivot",
            "Left about-face",
            "Leftward return turn",
            "Leftward reversing curve",
        ]
        stationary = [
            "At a standstill",
            "Motionless",
            "Unmoving",
            "Static position",
            "Immobilized",
            "Not in motion",
            "Fixed in place",
            "Idle",
            "Inert",
            "Anchored",
        ]
        right = [
            "Rightward",
            "To the right",
            "Right-hand side",
            "Starboard",
            "Rightward direction",
            "Clockwise direction",
            "Right-leaning",
            "Rightward bound",
            "Bearing right",
            "Veering right",
        ]
        left = [
            "Leftward",
            "To the left",
            "Left-hand side",
            "Port",
            "Leftward direction",
            "Counterclockwise direction",
            "Left-leaning",
            "Leftward bound",
            "Bearing left",
            "Veering left",
        ]
        straight_right = [
            "Straight then right",
            "Forward followed by a right turn",
            "Proceed straight, then veer right",
            "Continue straight before turning right",
            "Advance straight, then bear right",
            "Go straight, then curve right",
            "Head straight, then pivot right",
            "Move straight, then angle right",
            "Straight-line, followed by a right deviation",
            "Directly ahead, then a rightward shift",
        ]
        straight_left = [
            "Straight then left",
            "Forward followed by a left turn",
            "Proceed straight, then veer left",
            "Continue straight before turning left",
            "Advance straight, then bear left",
            "Go straight, then curve left",
            "Head straight, then pivot left",
            "Move straight, then angle left",
            "Straight-line, followed by a left deviation",
            "Directly ahead, then a leftward shift",
        ]
        straight = [
            "Directly ahead",
            "Forward",
            "Straightforward",
            "In a straight line",
            "Linearly",
            "Unswervingly",
            "Onward",
            "Direct path",
            "True course",
            "Non-curving path",
        ]

        buckets = [
            "Right-U-Turn",
            "Left-U-Turn",
            "Stationary",
            "Right",
            "Left",
            "Straight-Right",
            "Straight-Left",
            "Straight",
        ]

        bucket_synonym_lists = [
            right_u_turn,
            left_u_turn,
            stationary,
            right,
            left,
            straight_right,
            straight_left,
            straight,
        ]

        output = {}

        for index, synonym_list in enumerate(bucket_synonym_lists):
            current_bucket = buckets[index]
            if current_bucket not in output:
                output[current_bucket] = {}
            for synonym in synonym_list:
                bucket_similarities = test_bert_encoding(synonym)
                # print(synonym)
                output[buckets[index]][synonym] = bucket_similarities
                # print(bucket_similarities)
                # print()

        print(output)
        with open("output/bert_embedding_test.json", "w") as file:
            json.dump(str(output), file, indent=4)

    def do_test_trajectory_bucketing(self, arg: str):
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
        with open("config.yml", "r") as file:
            config = yaml.safe_load(file)
            output_folder = config["output_folder"]

        for i in range(20):
            # Get random scenario
            random_scenario_index = random.randint(0, number_of_scenarios)
            random_scenario = Scenario(scenarios[random_scenario_index])

            # Get random vehicle ID
            vehicles_for_scenario = get_vehicles_for_scenario(random_scenario)
            number_of_vehicles_in_scenario = len(vehicles_for_scenario)
            random_vehicle_id = vehicles_for_scenario[
                random.randint(0, number_of_vehicles_in_scenario - 1)
            ]
            trajectory = Trajectory(random_scenario, random_vehicle_id)

            # Plot the vehicle trajectory
            visualized_trajectory = self.loaded_scenario.visualize_trajectory(
                random_scenario, specific_id=random_vehicle_id
            )

            # Safe trajectory with the defined name convention
            visualized_trajectory.savefig(
                f"{output_folder}{trajectory.direction}_{random_scenario_index}_{random_vehicle_id}.png"
            )

            # Safe trajectory coordinates to csv with same naming convention as trajectory visualization
            trajectory.splined_coordinates.to_csv(
                f"{output_folder}{trajectory.direction}_{random_scenario_index}_{random_vehicle_id}.scsv"
            )

        print("Successfully prepared the trajectory bucket data for training!\n")

    # def do_training_data_length(self, arg: str):
    #     """Returns the length of the labeled training data.

    #     Args:
    #         arg (str): No arguments required.
    #     """

    #     training_data = TrainingData(
    #         "/home/pmueller/llama_traffic/datasets/zipped_labeled_trajectories.json"
    #     )
    #     print(training_data.get_size())

    def do_test_transformer_training(self, arg: str):
        train_transformer()

    def do_infer_with_transformer(self, arg: str):
        model = keras.models.load_model("models/my_transformer_model")
        # Check for empty arguments (no bucket provided)
        if arg == "":
            print(
                (
                    "\nYou have provided no bucket for which to predict the embedding.\nPlease provide a bucket!\n"
                )
            )
            return

        # Checking if a scenario has been loaded already.
        if not self.scenario_loaded():
            return
        vehicle_id = arg.split()[0]

        trajectory = Trajectory(self.loaded_scenario, vehicle_id)
        print(model((trajectory.rotated_coordinates, trajectory.rotated_coordinates)))

    def do_test_positional_encoding(self, arg: str):
        # Check for empty arguments (no bucket provided)
        if arg == "":
            print(
                (
                    "\nYou have provided no bucket for which to predict the embedding.\nPlease provide a bucket!\n"
                )
            )
            return

        # Checking if a scenario has been loaded already.
        if not self.scenario_loaded():
            return

        vehicle_id = arg.split()[0]
        trajectory = Trajectory(self.loaded_scenario, vehicle_id)

        coordinates = np.array(trajectory.rotated_coordinates)

        encoding = positional_encoding(101, 2)

        positional_embedding = PositionalEmbedding()
        global_embedding = GlobalSelfAttention(num_heads=2, key_dim=101)
        feedforward = FeedForward(2, 64)
        encoder_layer = EncoderLayer(d_model=2, num_heads=3, dff=64)
        encoder = Encoder(num_layers=3, d_model=2, num_heads=8, dff=64)
        cross_attention = CrossAttention(num_heads=2, key_dim=512)
        causal_self_attention = CausalSelfAttention(num_heads=2, key_dim=2)
        decoder_layer = DecoderLayer(d_model=2, num_heads=8, dff=64)

        org_coordinates = coordinates
        coordinates = np.expand_dims(coordinates, axis=0).reshape(1, 101, 2)

        num_layers = 4
        d_model = 2
        dff = 512
        num_heads = 8
        dropout_rate = 0.1

        transformer = Transformer(
            num_layers=num_layers,
            d_model=d_model,
            num_heads=num_heads,
            dff=dff,
            dropout_rate=dropout_rate,
        )

        def masked_loss(label, pred):
            print(f"Shape label: {label.shape}")
            print(f"Shape prediction: {pred.shape}")
            mask = label != 0
            loss_object = tf.keras.losses.MeanSquaredError()
            loss = loss_object(label, pred)

            mask = tf.cast(mask, dtype=loss.dtype)
            loss *= mask

            loss = tf.reduce_sum(loss) / tf.reduce_sum(mask)
            return loss

        learning_rate = CustomSchedule(d_model)

        optimizer = tf.keras.optimizers.Adam(
            learning_rate, beta_1=0.9, beta_2=0.98, epsilon=1e-9
        )

        transformer.compile(loss=masked_loss, optimizer=optimizer)

        data = tf.data.Dataset.from_tensor_slices((coordinates, coordinates))

        def prepare_batch(coordinates):
            coordinates = coordinates.to_tensor()  # Convert to 0-padded dense Tensor

            return (coordinates, coordinates), coordinates

        def make_batches(ds):
            print(str(ds.batch(1)))
            return ds.batch(1)

        print(make_batches(data))

        train_batches = make_batches(data)

        wandb.init(config={"bs": 12})

        transformer.fit(
            x=(
                tf.convert_to_tensor(coordinates, dtype=tf.float64),
                tf.convert_to_tensor(coordinates, dtype=tf.float64),
            ),
            y=tf.convert_to_tensor(coordinates, dtype=tf.float64),
            epochs=1000,
            callbacks=[WandbMetricsLogger()],
        )
        wandb.finish()

        output = transformer((coordinates, coordinates))
        print(output.shape)
        print(masked_loss(coordinates, output))

        # print(coordinates + encoding)

    def do_train_transformer_encoder(self, arg: str):
        encoder = Encoder(num_layers=3, d_model=2, num_heads=8, dff=64)
        encoder.compile(loss="mean_squared_error", optimizer="adam")
        # Load labeled trajectory data
        with open("datasets/labeled_ego_trajectories.json", "r") as file:
            trajectories_data = json.load(file)
        encoder.fit()

    def do_classification(self, arg: str):
        """Trains a classification model and tests for its accuracy.

        Args:
            arg (str): No arguments required.
        """
        train_classifier()

    def do_infer_with_simple_neural_network(self, arg):
        """Infer with the neural network.

        Args:
            arg (str): Bucket for which to predict embedding.
        """

        # Load config file
        # with open("config.yml", "r") as file:
        #     config = yaml.safe_load(file)
        #     output_folder = config["output_folder"]

        # Check for empty arguments (no bucket provided)
        if arg == "":
            print(
                (
                    "\nYou have provided no bucket for which to predict the embedding.\nPlease provide a bucket!\n"
                )
            )
            return

        # Checking if a scenario has been loaded already.
        if not self.scenario_loaded():
            return

        vehicle_id = arg.split()[0]

        trajectory = Trajectory(scenario=self.loaded_scenario, specific_id=vehicle_id)

        prediction = infer_with_simple_neural_network(trajectory)

        print(prediction)

        prediction_plot = self.loaded_scenario.visualize_coordinates(
            prediction, title=f"Prediction {vehicle_id}"
        )
        prediction_plot.savefig(
            f"output/simple_prediction_{vehicle_id}_plot_with_map.png"
        )
        plt.close()

        fig, ax = plt.subplots(
            figsize=(10, 10)
        )  # Create a figure and a set of subplots
        prediction.to_csv(f"output/simple_nn_prediction_{vehicle_id}.csv")
        ax.plot(
            prediction["X"],
            prediction["Y"],
            "ro-",
            markersize=5,
            linewidth=2,
        )  # 'ro-' creates a red line with circle markers

        # Set aspect of the plot to be equal
        ax.set_aspect("equal")

        plt.savefig(f"output/simple_prediction_{vehicle_id}_plot_without_map.png")
        print(prediction)
        print(prediction.shape)

        # First 101 dimensions are the predicted x coordinates, the last 101 dimensions are the predicted y coordinatesk
        # x_coords = prediction[0][:101]
        # y_coords = prediction[0][101:]

        # Store the coordinates in coordinate dictionary
        # coordinates = {"X": x_coords, "Y": y_coords}

        # trajectory = Trajectory(None, None)
        # plot = trajectory.visualize_raw_coordinates_without_scenario(coordinates)
        # plot.savefig(f"{output_folder}predicted_trajectory.png")

    def do_infer_with_neural_network(self, arg):
        """Infer with the neural network.

        Args:
            arg (str): Bucket for which to predict embedding.
        """

        # Load config file
        # with open("config.yml", "r") as file:
        #     config = yaml.safe_load(file)
        #     output_folder = config["output_folder"]

        # Check for empty arguments (no bucket provided)
        if arg == "":
            print(
                (
                    "\nYou have provided no bucket for which to predict the embedding.\nPlease provide a bucket!\n"
                )
            )
            return

        # Split the argument once
        args = arg.split()
        bucket = args[0]
        starting_x = int(args[1])
        starting_y = int(args[2])

        starting_x_array = np.array([[starting_x]])  # Shape becomes (1, 1)
        starting_y_array = np.array([[starting_y]])  # Shape becomes (1, 1)

        # Get BERT embedding
        embedding = get_bert_embedding(bucket)

        # Convert starting_x and starting_y to arrays and concatenate
        # Assuming you want to append these as new elements along an existing axis (e.g., axis 0)
        embedding = np.concatenate(
            [embedding, starting_x_array, starting_y_array], axis=1
        )

        prediction = infer_with_neural_network(embedding)

        fig, ax = plt.subplots(
            figsize=(10, 10)
        )  # Create a figure and a set of subplots

        # Scale the normalized trajectory to fit the figure

        # Plot the trajectory
        x_coordinates = prediction[0][0::2]
        y_coordinates = prediction[0][1::2]
        ax.plot(
            x_coordinates,
            y_coordinates,
            "ro-",
            markersize=5,
            linewidth=2,
        )  # 'ro-' creates a red line with circle markers

        # Set aspect of the plot to be equal
        ax.set_aspect("equal")

        plt.savefig("test.png")
        print(prediction)
        print(prediction.shape)

        # First 101 dimensions are the predicted x coordinates, the last 101 dimensions are the predicted y coordinatesk
        # x_coords = prediction[0][:101]
        # y_coords = prediction[0][101:]

        # Store the coordinates in coordinate dictionary
        # coordinates = {"X": x_coords, "Y": y_coords}

        # trajectory = Trajectory(None, None)
        # plot = trajectory.visualize_raw_coordinates_without_scenario(coordinates)
        # plot.savefig(f"{output_folder}predicted_trajectory.png")

    def do_print_available_device_for_training(self, arg):
        """Prints the available device for training.

        Args:
            arg (str): No arguments required.
        """

        device = (
            "cuda"
            if torch.cuda.is_available()
            else "mps"
            if torch.backends.mps.is_available()
            else "cpu"
        )
        print(f"Using {device} device")

    def do_print_mean_squared_error(self, arg: str):
        """Prints the mean squared error between the simple model's predicted trajectory
        and the trajectory of the car whose ID has been given.

        Args:
            arg (str): The vehicle ID of the vehicle to be compared.
        """
        # Checking if a scenario has been loaded already.
        if not self.scenario_loaded():
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

        trajectory = Trajectory(self.loaded_scenario, vehicle_id)
        real_coordinates = trajectory.splined_coordinates[0:20]
        real_coordinates = np.array(
            list(zip(real_coordinates["X"], real_coordinates["Y"]))
        ).flatten()
        print(f"Real coordinates shape: {real_coordinates.shape}")
        print(real_coordinates)

        predicted_coordinates = infer_with_simple_neural_network(trajectory)
        predicted_coordinates = predicted_coordinates.reshape(
            -1,
        )
        print(f"Predicted coordinates shape: {predicted_coordinates.shape}")
        print(predicted_coordinates)
        mse_loss = MeanSquaredError()

        print(
            f"Mean Squared Error: {mse_loss(real_coordinates, predicted_coordinates).numpy()}"
        )

    def do_print_ego_coordinates(self, arg: str):
        """Prints the ego coordinates of the vehicle whose ID has been given in the loaded scenario.

        Args:
            arg (str): The vehicle ID of the vehicle for whicht to print the ego coordinates.
        """

        # Check for empty arguments (no bucket provided)
        if arg == "":
            print(
                (
                    "\nYou have provided no bucket for which to predict the embedding.\nPlease provide a bucket!\n"
                )
            )
            return

        # Checking if a scenario has been loaded already.
        if not self.scenario_loaded():
            return

        vehicle_id = arg.split()[0]

        trajectory = Trajectory(self.loaded_scenario, vehicle_id)
        print("Test")
        print(trajectory.get_ego_coordinates())

    def do_plot_ego_coordinates(self, arg: str):
        # Check for empty arguments (no bucket provided)
        if arg == "":
            print(
                (
                    "\nYou have provided no bucket for which to predict the embedding.\nPlease provide a bucket!\n"
                )
            )
            return

        # Checking if a scenario has been loaded already.
        if not self.scenario_loaded():
            return

        vehicle_id = arg.split()[0]

        trajectory = Trajectory(self.loaded_scenario, vehicle_id)

        ego_plot = visualize_raw_coordinates_without_scenario(
            trajectory.ego_coordinates["X"], trajectory.ego_coordinates["Y"]
        )
        ego_plot.savefig(f"output/ego_plot_{vehicle_id}.png")
        plt.close()

    # def do_init_dataset(self, arg: str):
    #     """Initialize the dataset.

    #     Args:
    #         arg (str): No arguments required.
    #     """

    #     print("\nInitializing the dataset...")
    #     data = TrainingData("datasets/zipped_labeled_trajectories.json")
    #     print(data)
    #     print(len(data))
    #     print(data[0][0].shape)
    #     print("Successfully initialized the dataset!\n")

    def do_train_neural_network(self, arg: str):
        """Train the neural network.

        Args:
            arg (str): No arguments required.
        """

        train_neural_network()

    def do_train_simple_neural_network(self, arg: str):
        """Train the neural network.

        Args:
            arg (str): No arguments required.
        """

        train_simple_neural_network()

    def do_train_lstm_neural_network(self, arg: str):
        """Train a LSTM neural network for trajectory generation.

        Args:
            args(str): No arguments required.
        """
        train_lstm_neural_network()

    def do_print_training_data_length(self, arg: str):
        """Prints the number of training trajectories to the console.

        Args:
            arg (str): No arguments required.
        """
        # Load labeled trajectory data
        with open("datasets/labeled_trajectories.json", "r") as file:
            trajectories_data = json.load(file)
        print(len(trajectories_data))

    def do_train_right_neural_network(self, arg: str):
        """Train the neural network for right trajectories.

        Args:
            arg (str): No arguments required.
        """

        train_right_neural_network()

    def do_train_stationary_neural_network(self, arg: str):
        """Trains a neural network to generate stationary trajectories.

        Args:
            arg (str): No arguments required.
        """
        train_stationary_neural_network()

    def do_train_rnn_neural_network(self, arg: str):
        """Trains a very simple RNN neural network.

        Args:
            arg (str): No arguments required.
        """
        train_rnn_neural_network()

    def do_init_bucket_embeddings(self, arg: str):
        """Initialize the bucket embeddings.

        Args:
            arg (str): No arguments required.
        """

        print("\nInitializing the bucket embeddings...")
        init_bucket_embeddings()
        print("Successfully initialized the bucket embeddings!\n")

    def do_print_bucket_embedding_for_bucket(self, arg: str):
        """Returns the bucket embedding for the given bucket.

        Args:
            arg (str): Bucket for which to predict embedding.
        """

        # Check for empty arguments (no bucket provided)
        if arg == "":
            print(
                (
                    "\nYou have provided no bucket for which to predict the embedding.\nPlease provide a bucket!\n"
                )
            )
            return

        bucket = arg.split()[0]
        bucket_embeddings = {}
        with open("bucket_embeddings.json", "r") as file:
            bucket_embeddings = json.load(file)

        print(bucket_embeddings[bucket])
        print(len(bucket_embeddings[bucket]))

    def do_print_training_dataset_stats(self, args: str):
        """Prints the number of training examples for each of the buckets

        Args:
            args (str): No arguments required.
        """

        buckets = {
            "Left": 0,
            "Right": 0,
            "Stationary": 0,
            "Straight": 0,
            "Straight-Left": 0,
            "Straight-Right": 0,
            "Right-U-Turn": 0,
            "Left-U-Turn": 0,
        }
        print("Before loading")
        with open("output/processed.json", "r") as file:
            text = file.read()

        print("After loading")

        for bucket in buckets:
            buckets[bucket] = text.count(bucket)

        # for value in trajectories_data.values():
        #     for bucket in buckets:
        #         if value["Direction"] == bucket:
        #             buckets[value[bucket]] += 1

        sns.set_theme(style="whitegrid")

        # Convert buckets into a DataFrame suitable for plotting
        buckets_df = pd.DataFrame(buckets, index=[0])

        buckets_df.plot.bar(
            edgecolor="white", linewidth=3, figsize=(6, 8)
        ).set_xticklabels("")

        # Adding plot details
        plt.title("Direction Counts in Trajectory Data")
        plt.xlabel("Directions")
        plt.ylabel("Counts")

        plt.savefig("output/training_data_stats_1.png")
        plt.close()

        print(buckets_df.head())

        buckets_df = buckets_df.loc[:, ["Right-U-Turn", "Left-U-Turn"]]
        sns.set_theme(style="whitegrid")

        # Convert buckets into a DataFrame suitable for plotting

        buckets_df.plot.bar(
            edgecolor="white",
            linewidth=3,
            figsize=(6, 8),
            color={"Right-U-Turn": "pink", "Left-U-Turn": "grey"},
        ).set_xticklabels("")

        # Adding plot details
        plt.title("Direction Counts in Trajectory Data")
        plt.xlabel("Directions")
        plt.ylabel("Counts")

        plt.savefig("output/training_data_stats_2.png")
        plt.close()

        print(buckets)

    def do_plot_bucketing_limits(self, arg: str):
        # Load config file
        with open("config.yml", "r") as file:
            config = yaml.safe_load(file)
            scenario_data_folder = config["scenario_data_folder"]
            example_scenario_path = config["example_scenario_path"]

        if not self.scenario_loaded():
            return

        center_y, center_x, width = self.loaded_scenario.get_viewport()

        # Stationary Plot Creation

        stationary_step_size = (0.03 * width) / 101

        stationary_x = [center_x for i in range(101)]
        stationary_y = [center_y + (stationary_step_size * i) for i in range(101)]

        stationary_coordinates = pd.DataFrame({"X": stationary_x, "Y": stationary_y})

        stationary_plot = self.loaded_scenario.visualize_coordinates(
            stationary_coordinates
        )
        stationary_plot.savefig("output/stationary_plot.png")
        plt.close()

        # Straight Plot

        y_straight_step_size = (0.10 * width) / 101
        x_straight_step_size = math.tan(math.radians(15)) * y_straight_step_size

        straight_x = [
            center_x + (x_straight_step_size * i) - center_x + 0.01 for i in range(101)
        ]

        test_straight_x = [center_x + (x_straight_step_size * i) for i in range(101)]

        straight_y = [
            center_y + (y_straight_step_size * i) - center_y for i in range(101)
        ]

        y1 = straight_y[0]
        y2 = straight_y[-1]

        x1 = straight_x[0]
        x2 = straight_x[-1]

        a, b = self.get_exponential_function(y1, x1, y2, x2)

        straight_x = [a * math.pow(b, y) + center_x for y in straight_y]

        straight_y = [center_y + (y_straight_step_size * i) for i in range(101)]

        straight_coordinates = pd.DataFrame({"X": straight_x, "Y": straight_y})

        straight_plot = self.loaded_scenario.visualize_coordinates(straight_coordinates)
        straight_plot.savefig("output/straight_plot.png")
        plt.close()

        # Straight Plot 2

        y_straight_step_size = (0.10 * width) / 101
        x_straight_step_size = math.tan(math.radians(0)) * y_straight_step_size

        straight_x = [
            center_x + (x_straight_step_size * i) - center_x + 0.01 for i in range(101)
        ]

        straight_y = [
            center_y + (y_straight_step_size * i) - center_y for i in range(101)
        ]

        y1 = straight_y[0]
        y2 = straight_y[-1]

        x1 = straight_x[0]
        x2 = straight_x[-1]

        a, b = self.get_exponential_function(y1, x1, y2, x2)

        straight_x = [a * math.pow(b, y) + center_x for y in straight_y]

        straight_y = [center_y + (y_straight_step_size * i) for i in range(101)]

        straight_coordinates = pd.DataFrame({"X": straight_x, "Y": straight_y})

        straight_plot = self.loaded_scenario.visualize_coordinates(straight_coordinates)
        straight_plot.savefig("output/straight_plot_2.png")
        plt.close()

        # Straight Plot 3

        y_straight_step_size = (0.10 * width) / 101
        x_straight_step_size = math.tan(math.radians(15)) * y_straight_step_size

        straight_x = [
            center_x + (x_straight_step_size * i) - center_x + 0.01 for i in range(101)
        ]

        straight_y = [
            center_y + (y_straight_step_size * i) - center_y for i in range(101)
        ]

        y1 = straight_y[0]
        y2 = straight_y[-1]

        x1 = straight_x[0]
        x2 = straight_x[-1]

        a, b = self.get_exponential_function(y1, x1, y2, x2)

        straight_x = [a * math.pow(b, y) + center_x for y in straight_y]
        strict_straight_x = [center_x for i in range(101)]

        diffs = []

        for index, x in enumerate(straight_x):
            diffs.append(straight_x[index] - strict_straight_x[index])

        straight_transformed_x = []
        for index, diff in enumerate(diffs):
            straight_transformed_x.append(strict_straight_x[index] - diffs[index])

        straight_y = [center_y + (y_straight_step_size * i) for i in range(101)]

        straight_coordinates = pd.DataFrame(
            {"X": straight_transformed_x, "Y": straight_y}
        )

        straight_plot = self.loaded_scenario.visualize_coordinates(straight_coordinates)
        straight_plot.savefig("output/straight_plot_3.png")
        plt.close()

        # Straight-Right Plot 1

        y_straight_step_size = (0.10 * width) / 101
        x_straight_step_size = math.tan(math.radians(40)) * y_straight_step_size

        straight_x = [
            center_x + (x_straight_step_size * i) - center_x + 0.01 for i in range(101)
        ]

        test_straight_x = [center_x + (x_straight_step_size * i) for i in range(101)]

        straight_y = [
            center_y + (y_straight_step_size * i) - center_y for i in range(101)
        ]

        y1 = straight_y[0]
        y2 = straight_y[-1]

        x1 = straight_x[0]
        x2 = straight_x[-1]

        a, b = self.get_exponential_function(y1, x1, y2, x2)

        straight_x = [a * math.pow(b, y) + center_x for y in straight_y]

        straight_y = [center_y + (y_straight_step_size * i) for i in range(101)]

        straight_coordinates = pd.DataFrame({"X": straight_x, "Y": straight_y})

        straight_plot = self.loaded_scenario.visualize_coordinates(straight_coordinates)
        straight_plot.savefig("output/straight_right_plot_1.png")
        plt.close()

        # Straight Left Plot

        y_straight_step_size = (0.10 * width) / 101
        x_straight_step_size = math.tan(math.radians(40)) * y_straight_step_size

        straight_x = [
            center_x + (x_straight_step_size * i) - center_x + 0.01 for i in range(101)
        ]

        straight_y = [
            center_y + (y_straight_step_size * i) - center_y for i in range(101)
        ]

        y1 = straight_y[0]
        y2 = straight_y[-1]

        x1 = straight_x[0]
        x2 = straight_x[-1]

        a, b = self.get_exponential_function(y1, x1, y2, x2)

        straight_x = [a * math.pow(b, y) + center_x for y in straight_y]
        strict_straight_x = [center_x for i in range(101)]

        diffs = []

        for index, x in enumerate(straight_x):
            diffs.append(straight_x[index] - strict_straight_x[index])

        straight_transformed_x = []
        for index, diff in enumerate(diffs):
            straight_transformed_x.append(strict_straight_x[index] - diffs[index])

        straight_y = [center_y + (y_straight_step_size * i) for i in range(101)]

        straight_coordinates = pd.DataFrame(
            {"X": straight_transformed_x, "Y": straight_y}
        )

        straight_plot = self.loaded_scenario.visualize_coordinates(straight_coordinates)
        straight_plot.savefig("output/straight_right_plot_2.png")
        plt.close()

        # Left Plot

        step_size = (0.15 * width) / 101
        starting_vector = np.array([0.001, step_size])
        angles = [i * 130 / 101 for i in range(101)]
        vectors = [self.rotate_vector(starting_vector, alpha) for alpha in angles]

        starting_point = np.array([center_x, center_y])
        coordinates = [starting_point, starting_point + starting_vector]
        for vector in vectors:
            coordinates.append(coordinates[-1] + vector)

        print(coordinates)

        x = [coordinate[0] for coordinate in coordinates]
        y = [coordinate[1] for coordinate in coordinates]

        coordinates = pd.DataFrame({"X": x, "Y": y})

        straight_plot = self.loaded_scenario.visualize_coordinates(coordinates)
        straight_plot.savefig("output/left_plot.png")
        plt.close()

        # Right Plot

        step_size = (0.15 * width) / 101
        starting_vector = np.array([0.001, step_size])
        angles = [i * -130 / 101 for i in range(101)]
        vectors = [self.rotate_vector(starting_vector, alpha) for alpha in angles]

        starting_point = np.array([center_x, center_y])
        coordinates = [starting_point, starting_point + starting_vector]
        for vector in vectors:
            coordinates.append(coordinates[-1] + vector)

        print(coordinates)

        x = [coordinate[0] for coordinate in coordinates]
        y = [coordinate[1] for coordinate in coordinates]

        coordinates = pd.DataFrame({"X": x, "Y": y})

        straight_plot = self.loaded_scenario.visualize_coordinates(coordinates)
        straight_plot.savefig("output/right_plot.png")
        plt.close()

        # # Right Plot

        # y_straight_step_size = (0.10 * width) / 101
        # x_straight_step_size = math.tan(math.radians(130)) * y_straight_step_size

        # straight_x = [
        #     center_x + (x_straight_step_size * i) - center_x + 0.01 for i in range(101)
        # ]

        # test_straight_x = [center_x + (x_straight_step_size * i) for i in range(101)]
        # print(test_straight_x)

        # straight_y = [
        #     center_y + (y_straight_step_size * i) - center_y for i in range(101)
        # ]

        # y1 = straight_y[0]
        # y2 = straight_y[-1]

        # x1 = straight_x[0]
        # x2 = straight_x[-1]

        # a, b = self.get_exponential_function(y1, x1, y2, x2)

        # straight_x = [a * math.pow(b, y) + center_x for y in straight_y]
        # print(straight_x)

        # straight_y = [center_y + (y_straight_step_size * i) for i in range(101)]

        # straight_coordinates = pd.DataFrame({"X": straight_x, "Y": straight_y})

        # straight_plot = self.loaded_scenario.visualize_coordinates(straight_coordinates)
        # straight_plot.savefig("output/right_plot.png")
        # plt.close()

    @staticmethod
    def rotate_vector(vector, alpha):
        # Convert alpha from degrees to radians
        alpha_rad = math.radians(alpha)

        # Rotation matrix
        rotation_matrix = np.array(
            [
                [math.cos(alpha_rad), -math.sin(alpha_rad)],
                [math.sin(alpha_rad), math.cos(alpha_rad)],
            ]
        )

        # Rotate the vector
        rotated_vector = np.dot(rotation_matrix, vector)
        return rotated_vector

    @staticmethod
    def get_90_degree_vector(alpha, vector):
        v1 = vector[0]
        v2 = vector[1]

        v2_star = math.sqrt(
            (
                math.pow(math.tan(math.radians(alpha)), 2)
                * (math.pow(v1, 2) + math.pow(v2, 2))
                * math.pow(v1, 2)
            )
            / ((math.pow(v1, 2) + math.pow(v2, 2)))
        )
        v1_star = -(v2_star * v2) / v1

        return np.array([v1_star, v2_star])

    @staticmethod
    def get_exponential_function(x1, y1, x2, y2):
        """Returns a and b for an exponential function of form y = a * b^2"""
        b = math.pow(y2 / y1, 1 / (x2 - x1))
        a = y1 / math.pow(b, x1)

        return a, b

    def do_plot_predicted_trajectory(self, arg: str):
        """Plot the predicted trajectory for the given bucket.

        Args:
            arg (str): No arguments required.
        """

        # Load config file
        with open("config.yml", "r") as file:
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
        # x_coordinates = prediction[:101]
        # y_coordinates = prediction[101:]
        # complete_coordinates = {"X": x_coordinates, "Y": y_coordinates}

        # trajectory = visualize_raw_coordinates_without_scenario(complete_coordinates)
        # trajectory.savefig(f"{output_folder}predicted_trajectory.png")

        print(
            ("Successfully created trajectory plot in "),
            f"{output_folder}predicted_trajectory.png",
        )

    def do_print_x_axis_angle(self, arg: str):
        # Checking if a scenario has been loaded already.
        if not self.scenario_loaded():
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

        trajectory = Trajectory(self.loaded_scenario, vehicle_id)
        angle = trajectory.x_axis_angle
        print(math.degrees(angle))

    def do_print_rotated_coordinates_for_vehicle(self, arg: str):
        # Checking if a scenario has been loaded already.
        if not self.scenario_loaded():
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
        trajectory = Trajectory(self.loaded_scenario, vehicle_id)

        trajectory_plot = trajectory.visualize_raw_coordinates_without_scenario(
            trajectory.rotated_coordinates
        )
        trajectory_plot.savefig(f"output/{vehicle_id}_rotated.png")

        print(trajectory.rotated_coordinates)

    def do_store_raw_scenario(self, arg: str):
        """Store the raw scenario in a JSON file.

        Args:
            arg (str): No arguments required.
        """

        # Checking if a scenario has been loaded already.
        if not self.scenario_loaded():
            return

        # Store the raw scenario in a JSON file
        with open(
            "/home/pmueller/llama_traffic/datasets/raw_scenario.json", "w"
        ) as file:
            file.write(str(self.loaded_scenario.data))

    def scenario_loaded(self):
        scenario_has_been_loaded = self.loaded_scenario != None

        if not scenario_has_been_loaded:
            print(
                (
                    "\nNo scenario has been initialized yet! \nPlease use 'load_scenario'"
                    " to load a scenario before calling the 'plot_scenario' command.\n"
                )
            )
        return scenario_has_been_loaded

    # Basic command to exit the shell
    def do_exit(self, arg: str):
        "Exit the shell: EXIT"
        print("\nExiting the shell...\n")
        return True

    do_EOF = do_exit


if __name__ == "__main__":
    SimpleShell().cmdloop()
