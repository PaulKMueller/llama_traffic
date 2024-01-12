import os
import voyageai

import numpy as np

vo = voyageai.Client()
# This will automatically use the environment variable VOYAGE_API_KEY.
# Alternatively, you can use vo = voyageai.Client(api_key="<your secret key>")

texts = [
    "The Mediterranean diet emphasizes fish, olive oil, ...",
    "Photosynthesis in plants converts light energy into ...",
    "20th-century innovations, from radios to smartphones ...",
    "Rivers provide water, irrigation, and habitat for ...",
    "Apple’s conference call to discuss fourth fiscal ...",
    "Shakespeare's works, like 'Hamlet' and ...",
]

# Embed the documents
result = vo.embed("Right", model="voyage-02")
print(len(result.embeddings[0]))


def get_voyage_encoding(input_text: str) -> dict:
    vo = voyageai.Client()

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

    bucket_embeddings = np.array(
        [vo.embed(bucket, model="voyage-02").embeddings[0] for bucket in buckets]
    )

    input_text_embedding = vo.embed(input_text, model="voyage-02").embeddings[0]

    # Compute the dot product between query embedding and document embedding
    # scores = np.dot(input_text_embedding, bucket_embeddings.T)[0]d

    dot = np.dot(input_text_embedding, bucket_embeddings[0])
    norm = np.linalg.norm(input_text_embedding) * np.linalg.norm(bucket_embeddings[0])
    print(dot / norm)
    scores = []

    for bucket in bucket_embeddings:
        dot = np.dot(input_text_embedding, bucket.T)
        norm = np.linalg.norm(input_text_embedding) * np.linalg.norm(bucket)
        score = dot / norm
        scores.append(score)

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
