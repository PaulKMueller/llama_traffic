import numpy as np
import json
from npz_utils import list_vehicle_files_absolute
import torch
from uae_explore import encode_with_uae

from sklearn.metrics import roc_curve, roc_auc_score
import matplotlib.pyplot as plt


class TRAGRetriever:
    def __init__(self):

        self.direction_index_mapping = {
            "Left": 0,
            "Right": 1,
            "Stationary": 2,
            "Straight": 3,
            "Straight-Left": 4,
            "Straight-Right": 5,
            "Right-U-Turn": 6,
            "Left-U-Turn": 7,
        }

        self.index_direction_mapping = {
            0: "Left",
            1: "Right",
            2: "Stationary",
            3: "Straight",
            4: "Straight-Left",
            5: "Straight-Right",
            6: "Right-U-Turn",
            7: "Left-U-Turn",
        }

        print("Loading bucket embedding cache...")
        with open("datasets/uae_buckets_cache.json") as bucket_cache:
            self.bucket_embedding_cache = json.load(bucket_cache)
        print("Finished")
        print("Loading Synonym embedding cache...")
        with open("datasets/synonyms_uae_cache.json") as synonym_cache:
            self.synonym_embedding_cache = json.load(synonym_cache)
        print("Finished")

        print("Loading trajectory encoder output...")
        self.encoded_trajectories_indirect = np.load(
            "datasets/encoder_output_a_mse.npy"
        )
        print("Finished")

        with open("datasets/synonym_bucket_mapping.json") as synonym_bucket_mapping:
            self.synonym_bucket_mapping = json.load(synonym_bucket_mapping)

        print("Loading trajectory buckets...")
        self.trajectory_buckets = np.load("datasets/raw_direction_labels.npy")

        print("Finished")

        print("Loading direct encoded trajectory buckets...")
        self.encoded_trajectories_direct = np.zeros(
            (self.trajectory_buckets.size, 1024)
        )

        for i, index in enumerate(self.trajectory_buckets):
            self.encoded_trajectories_direct[i] = (
                self.get_bucket_encoding_for_direction_index(index)
            )
        print(self.encoded_trajectories_direct.shape)

        print("Finished")
        self.vehicle_list = list_vehicle_files_absolute()

    def retrieve_trajectory_direct(self, user_input: str, k: int = 1):
        embedded_user_input = torch.Tensor(encode_with_uae(user_input))
        cos_sim = torch.nn.CosineSimilarity()
        similarities = cos_sim(
            torch.Tensor(self.encoded_trajectories_direct), embedded_user_input
        )
        torch.clamp(similarities, min=-1, max=1)
        values, indices = torch.topk(similarities, 10)
        vehicles = [self.get_vehicle_for_index(index) for index in indices]
        values_list = values.tolist()
        indices_list = indices.tolist()

        # Print the values and indices
        # print("Indices of top 10 values:", indices_list)
        # print("Top 10 values:", values_list)
        # print("Vehicles:", vehicles)
        # print(similarities.shape)
        return values, indices, vehicles

    def retrieve_trajectory_indirect(self, user_input: str, k: int = 1):
        embedded_user_input = torch.Tensor(encode_with_uae(user_input))
        cos_sim = torch.nn.CosineSimilarity()
        similarities = cos_sim(
            torch.Tensor(self.encoded_trajectories_indirect), embedded_user_input
        )
        torch.clamp(similarities, min=-1, max=1)
        values, indices = torch.topk(similarities, k)
        vehicles = np.array([self.get_vehicle_for_index(index) for index in indices])
        return values, indices, vehicles

    def retrieve_scenario_direct(self, user_input: str, k: int = 1):
        pass

    def get_vehicle_for_index(self, index: int):
        return self.vehicle_list[index].split("/")[-1]

    def get_bucket_encoding_for_direction_index(self, index: int):
        return np.array(
            self.bucket_embedding_cache[self.index_direction_mapping[index]]
        )

    def benchmark_indirect_trajectory_retrieval(self):

        occurences = {
            "Left": 79230,
            "Right": 68302,
            "Stationary": 26222,
            "Straight": 221035,
            "Straight-Left": 35270,
            "Straight-Right": 34643,
            "Left-U-Turn": 2787,
            "Right-U-Turn": 619,
        }
        cos_sim = torch.nn.CosineSimilarity()
        for key in list(self.synonym_embedding_cache.keys()):
            correct_bucket = self.synonym_bucket_mapping[key]
            occurence = occurences[self.index_direction_mapping[correct_bucket]]

            embedded_user_input = torch.Tensor(self.synonym_embedding_cache[key])
            similarities = cos_sim(
                torch.Tensor(self.encoded_trajectories_indirect), embedded_user_input
            )
            torch.clamp(similarities, min=-1, max=1)
            values, indices = torch.topk(similarities, 468108)
            vehicles = np.array(
                [self.get_vehicle_for_index(index) for index in indices]
            )
            correct = 0
            for i in indices[:occurence]:
                if correct_bucket == self.trajectory_buckets[i]:
                    correct += 1
            print(key)
            print(correct / occurence)

    def collect_scores_and_labels(self, key):
        scores = []
        labels = []
        cos_sim = torch.nn.CosineSimilarity()

        occurences = {
            "Left": 79230,
            "Right": 68302,
            "Stationary": 26222,
            "Straight": 221035,
            "Straight-Left": 35270,
            "Straight-Right": 34643,
            "Left-U-Turn": 2787,
            "Right-U-Turn": 619,
        }

        correct_bucket = self.synonym_bucket_mapping[key]
        occurence = occurences[self.index_direction_mapping[correct_bucket]]

        embedded_user_input = torch.Tensor(self.synonym_embedding_cache[key])
        similarities = cos_sim(
            torch.Tensor(self.encoded_trajectories_indirect), embedded_user_input
        )
        similarities = torch.clamp(similarities, min=-1, max=1)
        _, indices = torch.topk(similarities, occurence)

        preds = np.ones(occurence)
        labels = np.array(self.trajectory_buckets[indices] == correct_bucket)
        print(key)
        print(f"Accuracy: {np.count_nonzero(labels)/occurence}")
        # print(labels)

        return len(labels), labels

    def plot_roc_and_calculate_auc(self):
        for key in list(self.synonym_embedding_cache.keys()):
            occurence, labels = self.collect_scores_and_labels(key)

            # Convert lists to numpy arrays for compatibility with scikit-learn functions
            # scores = np.array(scores)
            # print(scores[:10])
            # labels = np.array(labels)

            x, y = 0, 0
            points_x = [x]
            points_y = [y]

            # Iterate through the boolean array
            for value in labels:
                if value:
                    y += 1  # Move up for True
                else:
                    x += 1  # Move right for False
                points_x.append(x)
                points_y.append(y)

            points_x = np.array(points_x) / (labels.shape[0] - np.count_nonzero(labels))
            points_y = np.array(points_y) / np.count_nonzero(labels)

            auc = np.trapz(points_y, points_x)

            # Plot the ROC curve
            plt.figure()
            plt.plot(
                points_x,
                points_y,
                color="darkorange",
                lw=2,
                label="ROC curve (area = %0.2f)" % auc,
            )
            plt.plot([0, 1], [0, 1], color="navy", lw=2, linestyle="--")
            plt.xlim([0.0, 1.0])
            plt.ylim([0.0, 1.05])
            plt.xlabel("False Positive Rate")
            plt.ylabel("True Positive Rate")
            plt.title(f"Receiver Operating Characteristic ({key})")
            plt.legend(loc="lower right")
            plt.savefig(f"output/roc_{key}.png")


# retriever = TRAGRetriever()
# print(retriever.retrieve_trajectory_direct("Turn right")[1])
# print(retriever.retrieve_trajectory_indirect("Turn right")[1])
# retriever.benchmark_indirect_trajectory_retrieval()
# retriever.plot_roc_and_calculate_auc()

with open("datasets/results_indirect_retrieval.json") as results_indirect:
    results = json.load(results_indirect)
with open("datasets/synonym_bucket_mapping.json") as synonym_bucket_mapping:
    mapping = json.load(synonym_bucket_mapping)

direction_index_mapping = {
    "Left": 0,
    "Right": 1,
    "Stationary": 2,
    "Straight": 3,
    "Straight-Left": 4,
    "Straight-Right": 5,
    "Right-U-Turn": 6,
    "Left-U-Turn": 7,
}

accuracies = {}
output = {}
keys = list(direction_index_mapping.keys())
for key in keys:
    accuracies = []
    bucket_number = direction_index_mapping[key]
    sum = 0
    for entry in results.items():
        synonym, accuracy = entry
        if mapping[synonym] == bucket_number:
            accuracies.append(accuracy)
            output[key] = accuracies
            sum += accuracy
    mean = np.array(output[key]).mean()
    std_dev = np.array(output[key]).std()
    print(key)
    print(mean)
    print(std_dev)
