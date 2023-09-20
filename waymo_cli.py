import cmd
import argparse

import uuid
import time
from datetime import datetime

from matplotlib import cm
import matplotlib.animation as animation
import matplotlib.pyplot as plt

import numpy as np
from IPython.display import HTML
import itertools
import tensorflow as tf

from google.protobuf import text_format
from waymo_open_dataset.metrics.ops import py_metrics_ops
from waymo_open_dataset.metrics.python import config_util_py as config_util
from waymo_open_dataset.protos import motion_metrics_pb2

from waymo_visualize import visualize_all_agents_smooth, create_animation
from waymo_initialize import init_waymo

class SimpleShell(cmd.Cmd):
    prompt = '(waymo_cli) '
    waymo_dataset = {}


    def arg_parser(self):
        parser = argparse.ArgumentParser()
        parser.add_argument('-n', '--name', default='World')
        parser.add_argument('-p', '--path')
        parser.add_argument('--ids', action='store_true')
        return parser

    def do_greet(self, arg):
        'Greet someone. Flags: --name=NAME'
        try:
            parsed = self.arg_parser().parse_args(arg.split())
            print(f"Hello, {parsed.name}!")
        except SystemExit:
            pass  # argparse calls sys.exit(), catch the exception to prevent shell exit

    def do_load_scenario(self, arg):
        """Initialize the Waymo Open Motion Dataset Scenario for the given path.
        Flags: --path or -p=PATH_TO_SCENARIO"""

        filename = arg.split()[0]
        self.waymo_dataset = init_waymo(filename)
        print("Successfully initialized the given scenario!")

    def do_print_current_raw_scenario(self, arg):
        print(self.waymo_dataset)

    def do_plot_scenario(self, arg):

        parser = self.arg_parser()
        args = parser.parse_args(arg.split())
        if args.ids:
            print("Plotting scenario with agent ids...")
            images = visualize_all_agents_smooth(
                self.waymo_dataset, with_ids=True)
        else:
            print("Plotting scenario without agent ids...")
            images = visualize_all_agents_smooth(
                self.waymo_dataset, with_ids=False)

        if self.waymo_dataset == {}:
            print("No scenario has been initialized yet!")
        else:
            anim = create_animation(images[::5])
            timestamp = datetime.now()
            anim.save(f'/home/pmueller/llama_traffic/output/{timestamp}.mp4',
                      writer='ffmpeg', fps=1)
            print(f"Successfully created animation in /home/pmueller/llama_traffic/output/{timestamp}.mp4!")

    def do_plot_vehicle(self, arg):
        vehicle_id = arg.split()[0]
        print("Plotting vehicle for given ID...")
        images = visualize_all_agents_smooth(
                decoded_example=self.waymo_dataset, with_ids=False, specific_id=vehicle_id)
        anim = create_animation(images[::5])
        timestamp = datetime.now()
        anim.save(f'/home/pmueller/llama_traffic/output/{timestamp}.mp4',
                      writer='ffmpeg', fps=1)
        print(f"Successfully created animation in /home/pmueller/llama_traffic/output/{timestamp}.mp4!")
        

    # Basic command to exit the shell
    def do_exit(self, arg):
        'Exit the shell: EXIT'
        print("Exiting the shell...")
        return True

    do_EOF = do_exit

if __name__ == '__main__':
    SimpleShell().cmdloop()
