import json

true = 0
false = 0

with open("datasets/trajectory_encoder_preds_mse.json") as predictions:
    pred_data = json.load(predictions)
with open("datasets/processed_vehicle_a.json") as ground_truth:
    ground_truth_data = json.load(ground_truth)


ground_truth_data_keys = list(ground_truth_data.keys())
pred_data_keys = list(pred_data.keys())

for i in range(len(ground_truth_data_keys)):
    pred_label = pred_data[pred_data_keys[i]]
    print(pred_data_keys[i])
    print(pred_label)
    ground_truth_label = ground_truth_data[ground_truth_data_keys[i]]["Direction"]
    print(ground_truth_data_keys[i])
    print(ground_truth_label)
    print()

    if ground_truth_label == pred_label:
        true += 1
    else:
        false += 1

print(f"False: {false}")
print(f"True: {true}")
