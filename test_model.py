from tqdm import tqdm
import numpy as np
import torch

from uae_explore import encode_with_uae
import json

from ego_trajectory_encoder import EgoTrajectoryEncoder
from npz_trajectory import NpzTrajectory
from trajectory_encoder_dataset import TrajectoryEncoderDataset
from torch.utils.data import DataLoader


def compute_similarities(
    model_path: str = "/home/pmueller/llama_traffic/models/trajectory_encoder.pth",
    trajectory_path="/storage_local/fzi_datasets_tmp/waymo_open_motion_dataset/unzipped/train-2e6/vehicle_d_13657_00002_4856147881.npz",
) -> dict:
    model = EgoTrajectoryEncoder()
    model.load_state_dict(
        torch.load("/home/pmueller/llama_traffic/models/trajectory_encoder.pth")
    )
    model.eval()

    # dataset = TrajectoryEncoderDataset()

    # data_loader = DataLoader(dataset, batch_size=1)

    npz_trajectory = NpzTrajectory(
        "/storage_local/fzi_datasets_tmp/waymo_open_motion_dataset/unzipped/train-2e6/vehicle_d_13657_00002_4856147881.npz"
    )

    coordinates = list(
        zip(npz_trajectory.coordinates["X"], npz_trajectory.coordinates["Y"])
    )

    coordinates = torch.Tensor(coordinates)

    # coordinates.to("cuda")

    # input = next(iter(data_loader)).to("cuda")

    print("Test")
    coordinates = coordinates.unsqueeze(0)
    with open("output/test.txt", "w") as file:
        model.eval()
        # model.to("cuda")
        with torch.no_grad():
            output = model(coordinates)
            torch.set_printoptions(profile="full")
            file.write(str(output))
            print(output.shape)

    with open("datasets/uae_buckets_cache.json") as cache:
        print("Test2")
        loaded_cache = json.load(cache)
        similarities = {}

        for bucket in loaded_cache.keys():
            similarities[bucket] = np.dot(np.array(loaded_cache[bucket]), output.T)

    print(similarities)
    return similarities


# sims = compute_similarities()
# print(max(sims, key=sims.get))


def benchmark_model_retrieval():
    preds = {}
    with open("datasets/trajectory_encoder_output_v2.json") as enc_output:
        enc_output_data = enc_output.read()
        with open("datasets/processed_vehicle_a.json") as processed:
            processed_data = processed.read()
            processed_data = json.loads(processed_data)
            enc_output_data = json.loads(enc_output_data)
            processed_keys = processed_data.keys()
            enc_output_keys = [key.split("/")[-1] for key in processed_data.keys()]
            for index in tqdm(range(len(processed_keys))):
                similarities = np.dot(
                    np.array(
                        processed_data[list(processed_keys)[index]],
                        np.array(enc_output_data[list(enc_output_keys)[index]]),
                    ),
                )
                max_sim = max(similarities, key=similarities.get)
                preds[enc_output_keys[index]] == max_sim

    with open("datasets/trajectory_encoder_preds.json") as output:
        json.dump(preds, output, indent=4)
        print("finished")


benchmark_model_retrieval()
