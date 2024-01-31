import pickle
import json

with open("datasets/trajectory_encoder_output.json") as file:
    data = file.read()
    print(data[:400])
