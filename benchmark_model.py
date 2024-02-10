import torch
from uae_explore import get_uae_encoding
import json
from ego_trajectory_encoder import EgoTrajectoryEncoder

from tqdm import tqdm

true = 0
false = 0

with open("datasets/encoder_preds_vehicle_b_mae.json") as predictions:
    pred_data = json.load(predictions)
with open("datasets/direction_labeled_npz_vehicle_b.json") as ground_truth:
    ground_truth_data = json.load(ground_truth)


ground_truth_data_keys = list(ground_truth_data.keys())
pred_data_keys = list(pred_data.keys())

for i in tqdm(range(len(ground_truth_data_keys))):
    pred_label = pred_data[pred_data_keys[i]]
    # print(pred_data_keys[i])
    # print(pred_label)
    ground_truth_label = ground_truth_data[ground_truth_data_keys[i]]["Direction"]
    # print(ground_truth_data_keys[i])
    # print(ground_truth_label)
    # print()

    if ground_truth_label == pred_label:
        true += 1
    else:
        false += 1

print(f"False: {false}")
print(f"True: {true}")


def test_trajectory_encoder_on_synonyms():
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
    model = EgoTrajectoryEncoder()
    model.load_state_dict(
        torch.load("/home/pmueller/llama_traffic/models/trajectory_encoder_wv_mae.pth")
    )
    model.eval()

    with open("datasets/trajectory"):
        pass

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
    with open("output/trajectory_encoder_benchmark.json", "w") as file:
        json.dump(str(output), file, indent=4)
