import yaml
import cohere
import numpy as np

# Get API key from yaml
# Load config file


def get_cohere_encoding(input_text: str) -> dict:
    with open("config.yaml", "r") as file:
        config = yaml.safe_load(file)
        cohere_api_key = config["cohere_api_key"]

    co = cohere.Client(cohere_api_key)

    buckets = [
        "Left",
        "Right",
        "Stationary",
        "Straight",
        "Straight-Left",
        "Straight-Right",
        "Right-U-Turn",
        "Left-U-Turn",
    ]

    doc_emb = co.embed(
        buckets, input_type="classification", model="embed-english-v3.0"
    ).embeddings
    doc_emb = np.asarray(doc_emb)

    input_text_embedding = co.embed(
        [input_text], input_type="search_query", model="embed-english-v3.0"
    ).embeddings
    input_text_embedding = np.asarray(input_text_embedding)
    print(input_text_embedding.shape)

    # Compute the dot product between query embedding and document embedding
    scores = np.dot(input_text_embedding, doc_emb.T)[0]

    # Find the highest scores
    max_idx = np.argsort(-scores)

    # print(f"Query: {input_text}")
    # for idx in max_idx:
    #     print(f"Score: {scores[idx]:.2f}")
    #     print(buckets[idx])
    #     print("--------")

    # Create dictionary for return
    output_dict = {}
    for idx in max_idx:
        output_dict[buckets[idx]] = scores[idx]

    return output_dict
