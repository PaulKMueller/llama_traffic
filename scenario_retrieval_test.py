import torch
import numpy as np
from tqdm import tqdm
from npz_utils import one_hot_encode_trajectory, decode_one_hot_vector

import json

scenario_synonyms = {
    "vehicle": [
        "Automobile",
        "Car",
        "Motor vehicle",
        "Conveyance",
        "Transport",
        "Machine",
        "Motorcar",
        "Auto",
        "Truck",
        "Van",
    ],
    "pedestrian": [
        "Walker",
        "Foot traveler",
        "Stroller",
        "Hiker",
        "Jogger",
        "Passerby",
        "Wayfarer",
        "Ramblers",
        "Perambulator",
        "Pedestrian traffic",
    ],
    "cyclist": [
        "Biker",
        "Bicycle rider",
        "Bike enthusiast",
        "Cyclist",
        "Mountain biker",
        "Road cyclist",
        "Bicyclist",
        "Cycle rider",
        "Velocipedist",
        "BMX rider",
    ],
    "freeway": [
        "Expressway",
        "Highway",
        "Motorway",
        "Interstate",
        "Turnpike",
        "Tollway",
        "Superhighway",
        "Thruway",
        "Autobahn",
        "Dual carriageway",
    ],
    "surface_street": [
        "Road",
        "City street",
        "Urban roadway",
        "Town road",
        "Local street",
        "Main street",
        "Secondary road",
        "Residential street",
        "Public road",
        "Thoroughfare",
    ],
    "bike_lane": [
        "Bicycle path",
        "Cycling lane",
        "Bike path",
        "Bicycle track",
        "Cycle path",
        "Bike trail",
        "Cycling track",
        "Bicycle lane",
        "Bike route",
        "Cycling route",
    ],
    "stop_sign": [
        "Stop signal",
        "Traffic stop sign",
        "Road stop indicator",
        "STOP board",
        "Halt sign",
        "Stop traffic sign",
        "Roadblock sign",
        "Intersection control sign",
        "Mandatory stop sign",
        "Octagonal traffic sign",
    ],
    "crosswalk": [
        "Pedestrian crossing",
        "Zebra crossing",
        "Walkway",
        "Crossing path",
        "Pedestrian path",
        "Cross path",
        "Footpath",
        "Pedestrian walkway",
        "Street crossing",
        "Pedestrian crossway",
    ],
    "driveway": [
        "Drive",
        "Private road",
        "Carriageway",
        "Access road",
        "Residential drive",
        "Entryway",
        "Service road",
        "Approach road",
        "Front drive",
        "Pathway",
    ],
    "parking_lot": [
        "Car park",
        "Parking area",
        "Parking space",
        "Vehicle parking",
        "Auto lot",
        "Parking garage",
        "Parking ground",
        "Motor park",
        "Parkade",
        "Parking deck",
    ],
    "intersection": [
        "Crossroads",
        "Junction",
        "Road junction",
        "Traffic intersection",
        "Crossway",
        "Four-way",
        "Roundabout",
        "T-intersection",
        "Road crossing",
        "Interchange",
    ],
    "turnaround": [
        "U-turn spot",
        "Turnabout area",
        "Revolving area",
        "Swing area",
        "Turning point",
        "Circular drive",
        "Loop area",
        "Turnback area",
        "Rotating space",
        "180-degree turn area",
    ],
}

feature_counts = {
    "vehicle": 468108,
    "pedestrian": 296067,
    "cyclist": 84441,
    "freeway": 42215,
    "surface_street": 466749,
    "bike_lane": 361176,
    "stop_sign": 161908,
    "crosswalk": 209908,
    "driveway": 179437,
    "parking_lot": 370959,
    "turnaround": 53560,
    "intersection": 445434,
}

with open("output/scenario_features.json") as scenario_features:
    feature_data = json.load(scenario_features)

scenario_features_real = np.load("output/scenario_features.npy")

with open("datasets/scenario_embedding_cache.json") as scenario_embedding_cache:
    embedding_cache = json.load(scenario_embedding_cache)

with open(
    "datasets/scenario_synonym_embedding_cache.json"
) as scenario_synonym_embedding_cache:
    synonym_embedding_cache = json.load(scenario_synonym_embedding_cache)

scenario_features_embeddings = np.load("output/scenario_features_embeddings.npy")

synonyms_keys = list(scenario_synonyms.keys())

for i in tqdm(range(len(synonyms_keys))):
    synonym_key = synonyms_keys[i]
    occurence = feature_counts[synonym_key]
    one_hot_synonym_key = one_hot_encode_trajectory(synonym_key)

    for synonym in scenario_synonyms[synonym_key]:
        correct = 0

        # Take synonym as user input
        embedding = synonym_embedding_cache[synonym]

        # Do the topk retrieval and sort based on this
        cos_sim = torch.nn.CosineSimilarity()
        similarities = cos_sim(
            torch.Tensor(scenario_features_embeddings), torch.Tensor(embedding)
        )
        torch.clamp(similarities, min=-1, max=1)
        values, indices = torch.topk(similarities, 468108)
        # print(indices)

        for j in range(occurence):
            real_scenario_features = scenario_features_real[indices[j]]
            correct += np.sum(real_scenario_features * one_hot_synonym_key) == 1

        print(f"Synonym: {synonym}, Correct: {correct}, Accuracy: {correct/occurence}")

    # print(scenario_synonyms[synonym_key])
    # print(one_hot_encode_trajectory(synonym_key))
