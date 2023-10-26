import pandas as pd


def get_trajectory_embedding(coordinates: pd.DataFrame):
    """Returns a trajectory embedding.

    Args:
        coordinates (pd.DataFrame): The coordinates of the trajectory as a DataFrame
    """    

    # Combine X and Y coordinates into a single list by writing X1, Y2, X2, Y2, ...
    embedding = []
    for _, row in coordinates.iterrows():
        embedding.append(row["X"])
        embedding.append(row["Y"])

    print(len(embedding))

    return embedding