import numpy as np


def cosine_similarity(vec1: np.ndarray, vec2: np.ndarray) -> float:
    dot_product = np.dot(vec1, vec2)
    norm_product = np.linalg.norm(vec1) * np.linalg.norm(vec2)

    if norm_product == 0:
        return 0.0

    return float(dot_product / norm_product)
