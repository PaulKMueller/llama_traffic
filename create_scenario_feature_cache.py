import json
from uae_explore import encode_with_uae
import numpy as np
from npz_utils import decode_one_hot_vector
from tqdm import tqdm
from npz_utils import SCENARIO_FEATURES

with open("output/scenario_features.json") as scenario_features_file:
    scenario_features = json.load(scenario_features_file)

keys = list(scenario_features.keys())

# # cache = {}

print(len(SCENARIO_FEATURES))


def generate_12bit_combinations():
    """
    Generate all possible 12-bit combinations.
    Each combination is represented as a list of bits.

    Returns:
        list of lists: A list containing all 12-bit combinations
    """
    # There are 2^12 possible combinations for 12 bits
    num_combinations = 2**12
    # Generate all combinations using a list comprehension
    # The format function is used to convert each number into a binary string with leading zeros
    # The inner list comprehension converts each binary string into a list of integers (bits)
    combinations = [
        [int(bit) for bit in format(i, "012b")] for i in range(num_combinations)
    ]
    return combinations


cache_keys = generate_12bit_combinations()
cache = {}
for i in tqdm(range(len(cache_keys))):
    text = decode_one_hot_vector(cache_keys[i])
    embedding = encode_with_uae(text)[-1]
    # print(embedding)
    # print(str(cache_keys[i]))
    # print(list(embedding))
    cache[str(cache_keys[i])] = list(embedding)

# # print(cache)
with open("datasets/scenario_embedding_cache.json", "w") as cache_file:
    cache_file.write(str(cache))
