import json
import pandas as pd

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

files = [
    "datasets/bert_embedding.json",
    "datasets/uae_embedding.json",
    "datasets/voyage_embedding.json",
    "datasets/cohere_embedding.json",
]

accuracies = {}

for file in files:
    with open(file) as data:
        content = json.load(data)

    false_categorizations = 0
    correct_categorizations = 0

    false = []
    correct = []

    for bucket in content.keys():
        # print(f"Bucket: {bucket}")
        # print(content[bucket].keys())
        for key in content[bucket].keys():
            similarities = content[bucket][key]
            max_bucket = max(zip(similarities.values(), similarities.keys()))[1]
            # print(similarities)
            # print(max_bucket)
            if max_bucket == bucket:
                correct_categorizations += 1
                correct.append(bucket)
            else:
                false_categorizations += 1
                false.append(bucket)

    print(correct_categorizations)
    d = {x: correct.count(x) for x in correct}
    print(d)
    # print(correct)
    print(false_categorizations)
    d = {x: false.count(x) for x in false}
    print(d)

    accuracy = correct_categorizations / (
        correct_categorizations + false_categorizations
    )
    accuracies[file.split("/")[1].split(".")[0]] = [accuracy]
    # print(false)


# Assuming 'accuracies' is your dictionary with data
output_data = pd.DataFrame.from_dict(accuracies)

# Convert accuracies to percentages
output_data *= 100

# Use a nicer style for the plot
plt.style.use("ggplot")

# Create the bar plot with added space between bars
plot = output_data.plot.bar(
    rot=0, width=0.8
)  # Adjust 'width' to control space between bars

# Set the y-axis maximum to 100
plot.set_ylim(0, 100)

# Remove the label for 0 on the x-axis
plot.xaxis.set_major_locator(ticker.MultipleLocator(1))

# Remove the label '0'
plot.set_xticklabels(["" if x == 0 else int(x) for x in plot.get_xticks()])

# Set the title with bold font and add space above the plot
plot.set_title("Embedding benchmark accuracies", fontweight="bold", pad=20)

# Enhance the legend with a bold title
legend = plot.legend(title="Embeddings", bbox_to_anchor=(1.05, 1), loc="upper left")
plt.setp(legend.get_title(), fontweight="bold")

# Add text labels on the bars with black font color and include a '%' sign
for bar in plot.patches:
    plot.annotate(
        f"{format(bar.get_height(), '.1f')}%",  # format as one decimal place and add '%'
        (bar.get_x() + bar.get_width() / 2, bar.get_height()),  # position
        ha="center",
        va="center",
        size=10,
        xytext=(0, 8),
        textcoords="offset points",
        color="black",
    )  # change color to black

# Adjust layout
plt.tight_layout()

# Save the figure
plot.figure.savefig("output/embedding_accuracies.png")
