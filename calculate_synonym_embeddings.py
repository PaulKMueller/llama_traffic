from angle_emb import AnglE, Prompts

import json
from uae_explore import encode_with_uae

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


print("Test 1")
cache = {}
angle = AnglE.from_pretrained("WhereIsAI/UAE-Large-V1", pooling_strategy="cls").cuda()
print("Test 2")
angle.set_prompt(prompt=Prompts.C)
print("Test 3")
for synonyms in bucket_synonym_lists:
    for synonym in synonyms:
        print(synonym)

        input_text_embedding = angle.encode({"text": synonym}, to_numpy=True)
        cache[synonym] = input_text_embedding.tolist()[0]

with open("datasets/synonyms_uae_cache.json", "w") as cache_file:
    json.dump(cache, cache_file, indent=4)
