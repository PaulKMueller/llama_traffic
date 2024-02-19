from tqdm import tqdm
from uae_explore import encode_with_uae
import json
import torch

with open("datasets/encoder_output_vehicle_a_mse.json") as encoder_output_file:
    encoder_output_data = json.load(encoder_output_file)

user_input = str(input("What kind of trajectory do you want to find?"))

input_embedding = encode_with_uae(user_input)
cosine_sim = torch.nn.CosineSimilarity()
encoder_data_keys = list(encoder_output_data.keys())

max_sim = 0
best_fit = ""

for i in tqdm(range(len(encoder_data_keys))):
    key = encoder_data_keys[i]
    sim = cosine_sim(
        torch.Tensor(encoder_output_data[key]),
        torch.Tensor(input_embedding),
    )
    if sim > max_sim:
        max_sim = sim
        best_fit = encoder_data_keys[i]

print(best_fit)
print(max_sim)
