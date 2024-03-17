import json
from uae_explore import encode_with_uae
import numpy as np
from npz_utils import decode_one_hot_vector
from tqdm import tqdm

# with open("output/scenario_features.json") as scenario_features_file:
#     scenario_features = json.load(scenario_features_file)

# keys = list(scenario_features.keys())

# # cache = {}


# def generate_12bit_combinations():
#     """
#     Generate all possible 12-bit combinations.
#     Each combination is represented as a list of bits.

#     Returns:
#         list of lists: A list containing all 12-bit combinations
#     """
#     # There are 2^12 possible combinations for 12 bits
#     num_combinations = 2**12
#     # Generate all combinations using a list comprehension
#     # The format function is used to convert each number into a binary string with leading zeros
#     # The inner list comprehension converts each binary string into a list of integers (bits)
#     combinations = [
#         [int(bit) for bit in format(i, "012b")] for i in range(num_combinations)
#     ]
#     return combinations


# cache_keys = generate_12bit_combinations()
# cache = {}
# for i in tqdm(range(len(cache_keys))):
#     text = decode_one_hot_vector(cache_keys[i])
#     embedding = encode_with_uae(text)[-1]
#     # print(embedding)
#     # print(str(cache_keys[i]))
#     # print(list(embedding))
#     cache[str(cache_keys[i])] = list(embedding)

# # # print(cache)
# with open("datasets/scenario_embedding_cache.json") as cache_file:
#     json.dump(cache, cache_file, indent=4)
#     # cache = json.load(cache_file)


# output = []

# for index in tqdm(range(len(scenario_features))):
#     key = keys[index]
#     value = scenario_features[key]

#     embedding = cache[str(value)]
#     # output_file.write(str(embedding) + "\n")
#     output.append(embedding)

# np.save("output/scenario_features_embeddings.npy", np.array(output))

# data = np.load("output/scenario_features_embeddings.npy")
# print(data.shape)


# with open("output/scenario_features.json") as features:
#     feature_data = json.load(features)
#     feature_data = np.array([np.array(value) for value in feature_data.values()])
#     print(feature_data.shape)
# np.save("output/scenario_features.npy", feature_data)


with open("output/scenario_features.json") as features:
    feature_data = json.load(features)

values = list(feature_data.values())

with open("datasets/scenario_embedding_cache.json") as embedding_cache:
    cache_data = json.load(embedding_cache)

output = []


for i in tqdm(range(len(values))):
    output.append(cache_data[str(values[i])])

output = np.array(output)

print(output.shape)

np.save("output/scenario_embeddings.npy", output)
