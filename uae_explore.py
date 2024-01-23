from angle_emb import AnglE, Prompts
import numpy as np

# angle = AnglE.from_pretrained("WhereIsAI/UAE-Large-V1", pooling_strategy="cls").cuda()
# angle.set_prompt(prompt=Prompts.C)
# vec = angle.encode({"text": "hello world"}, to_numpy=True)
# print(vec)
# print(vec.shape)

# vecs = angle.encode([{"text": "hello world1"}, {"text": "hello world2"}], to_numpy=True)
# print(vecs)


def get_uae_encoding(input_text: str) -> dict:
    buckets_parsed = [
        {"text": "Left"},
        {"text": "Right"},
        {"text": "Stationary"},
        {"text": "Straight"},
        {"text": "Straight-Left"},
        {"text": "Straight-Right"},
        {"text": "Right-U-Turn"},
        {"text": "Left-U-Turn"},
    ]

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

    angle = AnglE.from_pretrained(
        "WhereIsAI/UAE-Large-V1", pooling_strategy="cls"
    ).cuda()
    angle.set_prompt(prompt=Prompts.C)

    bucket_embeddings = np.array([angle.encode(bucket) for bucket in buckets_parsed])

    input_text_embedding = angle.encode({"text": input_text}, to_numpy=True)

    # Compute the dot product between query embedding and document embedding
    # scores = np.dot(input_text_embedding, bucket_embeddings.T)[0]d

    dot = np.dot(input_text_embedding, bucket_embeddings.reshape(1024, -1))[0][0]
    norm = np.linalg.norm(input_text_embedding) * np.linalg.norm(bucket_embeddings[0])
    print(dot / norm)
    scores = []

    for bucket in bucket_embeddings:
        dot = np.dot(input_text_embedding, bucket.T)
        norm = np.linalg.norm(input_text_embedding) * np.linalg.norm(bucket)
        score = dot / norm
        scores.append(score[0][0])

    print(scores)

    # Find the highest scores
    max_idx = np.argsort(-np.array(scores))
    print(max_idx)

    # print(f"Query: {input_text}")
    # for idx in max_idx:
    #     print(f"Score: {scores[idx]:.2f}")
    #     print(buckets[idx])
    #     print("--------")

    # Create dictionary for return

    # print(max_idx)
    output_dict = {}
    for idx in max_idx:
        output_dict[buckets[idx]] = scores[idx]
    print(output_dict)
    return output_dict


def encode_with_uae(input_text: str) -> dict:
    angle = AnglE.from_pretrained(
        "WhereIsAI/UAE-Large-V1", pooling_strategy="cls"
    ).cuda()
    angle.set_prompt(prompt=Prompts.C)

    input_text_embedding = angle.encode({"text": input_text}, to_numpy=True)

    return input_text_embedding
