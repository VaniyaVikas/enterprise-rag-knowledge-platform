import math


def cosine_similarity(
    vector_a: list[float],
    vector_b: list[float],
) -> float:
    if not isinstance(vector_a, list):
        raise TypeError("vector_a must be a list of numbers.")

    if not isinstance(vector_b, list):
        raise TypeError("vector_b must be a list of numbers.")

    if not vector_a or not vector_b:
        raise ValueError("vectors cannot be empty.")

    if len(vector_a) != len(vector_b):
        raise ValueError("vectors must have the same dimensions.")

    if not all(isinstance(value, (int, float)) for value in vector_a):
        raise TypeError("vector_a must contain only numbers.")

    if not all(isinstance(value, (int, float)) for value in vector_b):
        raise TypeError("vector_b must contain only numbers.")

    dot_product = sum(
        a * b
        for a, b in zip(vector_a, vector_b)
    )

    magnitude_a = math.sqrt(
        sum(a * a for a in vector_a)
    )

    magnitude_b = math.sqrt(
        sum(b * b for b in vector_b)
    )

    if magnitude_a == 0 or magnitude_b == 0:
        raise ValueError("zero vectors are not supported.")

    return dot_product / (magnitude_a * magnitude_b)