
import math

import pytest

from src.vector_similarity import cosine_similarity


def test_identical_vectors_have_similarity_one():
    vector = [1.0, 2.0, 3.0]

    assert cosine_similarity(vector, vector) == pytest.approx(1.0)


def test_orthogonal_vectors_have_similarity_zero():
    vector_a = [1.0, 0.0]
    vector_b = [0.0, 1.0]

    assert cosine_similarity(vector_a, vector_b) == pytest.approx(0.0)


def test_opposite_vectors_have_similarity_negative_one():
    vector_a = [1.0, 2.0]
    vector_b = [-1.0, -2.0]

    assert cosine_similarity(vector_a, vector_b) == pytest.approx(-1.0)


def test_similarity_is_symmetric():
    vector_a = [0.2, 0.5, 0.8]
    vector_b = [0.7, 0.1, 0.4]

    assert cosine_similarity(vector_a, vector_b) == pytest.approx(
        cosine_similarity(vector_b, vector_a)
    )


def test_similarity_is_unchanged_by_positive_scaling():
    vector_a = [1.0, 2.0, 3.0]
    vector_b = [2.0, 4.0, 6.0]

    assert cosine_similarity(vector_a, vector_b) == pytest.approx(1.0)


def test_empty_vectors_are_rejected():
    with pytest.raises(ValueError, match="vectors cannot be empty."):
        cosine_similarity([], [])


def test_different_dimensions_are_rejected():
    with pytest.raises(
        ValueError,
        match="vectors must have the same dimensions.",
    ):
        cosine_similarity([1.0, 2.0], [1.0])


@pytest.mark.parametrize(
    "vector_a, vector_b",
    [
        ("not a vector", [1.0]),
        ([1.0], "not a vector"),
    ],
)
def test_non_list_vectors_are_rejected(vector_a, vector_b):
    with pytest.raises(TypeError):
        cosine_similarity(vector_a, vector_b)


@pytest.mark.parametrize(
    "vector_a, vector_b",
    [
        ([1.0, "bad"], [1.0, 2.0]),
        ([1.0, 2.0], [1.0, None]),
    ],
)
def test_non_numeric_values_are_rejected(vector_a, vector_b):
    with pytest.raises(TypeError):
        cosine_similarity(vector_a, vector_b)


@pytest.mark.parametrize(
    "vector_a, vector_b",
    [
        ([0.0, 0.0], [1.0, 2.0]),
        ([1.0, 2.0], [0.0, 0.0]),
    ],
)
def test_zero_vectors_are_rejected(vector_a, vector_b):
    with pytest.raises(ValueError, match="zero vectors are not supported."):
        cosine_similarity(vector_a, vector_b)


def test_result_is_a_finite_float():
    result = cosine_similarity([1.0, 2.0], [3.0, 4.0])

    assert isinstance(result, float)
    assert math.isfinite(result)

