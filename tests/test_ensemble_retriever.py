import pytest

from src.ensemble_retriever import EnsembleRetriever


class FakeRetriever:
    def __init__(self, results):
        self.results = results
        self.calls = []

    def search(self, query, k=3):
        self.calls.append((query, k))
        return self.results[:k]


def make_result(index, document=None):
    return {
        "index": index,
        "document": document or f"Document {index}",
        "score": 1.0,
    }


def make_ensemble(results_by_name, weights=None, rrf_constant=60):
    retrievers = {
        name: FakeRetriever(results)
        for name, results in results_by_name.items()
    }

    ensemble = EnsembleRetriever(
        retrievers=retrievers,
        weights=weights,
        rrf_constant=rrf_constant,
    )

    return ensemble, retrievers


def test_constructor_accepts_valid_retrievers():
    ensemble, _ = make_ensemble(
        {
            "bm25": [make_result(1)],
            "vector": [make_result(2)],
        }
    )

    assert set(ensemble.retrievers) == {"bm25", "vector"}


def test_constructor_assigns_equal_weights_by_default():
    ensemble, _ = make_ensemble(
        {
            "bm25": [],
            "vector": [],
        }
    )

    assert ensemble.weights == {
        "bm25": 0.5,
        "vector": 0.5,
    }


def test_constructor_copies_retriever_dictionary():
    original = {
        "bm25": FakeRetriever([]),
    }

    ensemble = EnsembleRetriever(retrievers=original)

    original["vector"] = FakeRetriever([])

    assert set(ensemble.retrievers) == {"bm25"}


def test_constructor_rejects_non_dictionary_retrievers():
    with pytest.raises(
        TypeError,
        match="retrievers must be a dictionary",
    ):
        EnsembleRetriever(retrievers=[])


def test_constructor_rejects_empty_retrievers():
    with pytest.raises(
        ValueError,
        match="retrievers cannot be empty",
    ):
        EnsembleRetriever(retrievers={})


@pytest.mark.parametrize("name", ["", "  ", 123])
def test_constructor_rejects_invalid_retriever_names(name):
    with pytest.raises(
        ValueError,
        match="retriever names must be non-empty strings",
    ):
        EnsembleRetriever(
            retrievers={
                name: FakeRetriever([]),
            }
        )


@pytest.mark.parametrize(
    "retriever",
    [None, "invalid", 123],
)
def test_constructor_rejects_invalid_retriever_objects(retriever):
    with pytest.raises(
        TypeError,
        match="each retriever must be callable",
    ):
        EnsembleRetriever(
            retrievers={
                "invalid": retriever,
            }
        )


@pytest.mark.parametrize(
    "constant",
    [True, False, 1.5, "60"],
)
def test_constructor_rejects_non_integer_rrf_constant(constant):
    with pytest.raises(
        TypeError,
        match="rrf_constant must be an integer",
    ):
        EnsembleRetriever(
            retrievers={
                "bm25": FakeRetriever([]),
            },
            rrf_constant=constant,
        )


@pytest.mark.parametrize("constant", [0, -1])
def test_constructor_rejects_non_positive_rrf_constant(constant):
    with pytest.raises(
        ValueError,
        match="rrf_constant must be greater than zero",
    ):
        EnsembleRetriever(
            retrievers={
                "bm25": FakeRetriever([]),
            },
            rrf_constant=constant,
        )


def test_constructor_accepts_custom_weights():
    ensemble, _ = make_ensemble(
        {
            "bm25": [],
            "vector": [],
        },
        weights={
            "bm25": 0.3,
            "vector": 0.7,
        },
    )

    assert ensemble.weights == {
        "bm25": 0.3,
        "vector": 0.7,
    }


def test_constructor_rejects_non_dictionary_weights():
    with pytest.raises(
        TypeError,
        match="weights must be a dictionary",
    ):
        EnsembleRetriever(
            retrievers={
                "bm25": FakeRetriever([]),
            },
            weights=[1.0],
        )


def test_constructor_rejects_mismatched_weight_names():
    with pytest.raises(
        ValueError,
        match="exactly the configured retriever names",
    ):
        EnsembleRetriever(
            retrievers={
                "bm25": FakeRetriever([]),
            },
            weights={
                "vector": 1.0,
            },
        )


@pytest.mark.parametrize(
    "weight",
    [True, False, "0.5", None],
)
def test_constructor_rejects_non_numeric_weights(weight):
    with pytest.raises(
        TypeError,
        match="each weight must be a number",
    ):
        EnsembleRetriever(
            retrievers={
                "bm25": FakeRetriever([]),
            },
            weights={
                "bm25": weight,
            },
        )


@pytest.mark.parametrize("weight", [-1, -0.1])
def test_constructor_rejects_negative_weights(weight):
    with pytest.raises(
        ValueError,
        match="weights cannot be negative",
    ):
        EnsembleRetriever(
            retrievers={
                "bm25": FakeRetriever([]),
            },
            weights={
                "bm25": weight,
            },
        )


def test_constructor_rejects_all_zero_weights():
    with pytest.raises(
        ValueError,
        match="at least one weight",
    ):
        EnsembleRetriever(
            retrievers={
                "bm25": FakeRetriever([]),
                "vector": FakeRetriever([]),
            },
            weights={
                "bm25": 0,
                "vector": 0,
            },
        )


@pytest.mark.parametrize(
    "query",
    [None, 123, []],
)
def test_search_rejects_non_string_query(query):
    ensemble, _ = make_ensemble(
        {
            "bm25": [],
        }
    )

    with pytest.raises(
        TypeError,
        match="query must be a string",
    ):
        ensemble.search(query)


@pytest.mark.parametrize(
    "query",
    ["", "  ", "\n"],
)
def test_search_rejects_empty_query(query):
    ensemble, _ = make_ensemble(
        {
            "bm25": [],
        }
    )

    with pytest.raises(
        ValueError,
        match="query cannot be empty",
    ):
        ensemble.search(query)


@pytest.mark.parametrize(
    "k",
    [True, False, 1.5, "3"],
)
def test_search_rejects_non_integer_k(k):
    ensemble, _ = make_ensemble(
        {
            "bm25": [],
        }
    )

    with pytest.raises(
        TypeError,
        match="k must be an integer",
    ):
        ensemble.search("query", k=k)


@pytest.mark.parametrize("k", [0, -1])
def test_search_rejects_non_positive_k(k):
    ensemble, _ = make_ensemble(
        {
            "bm25": [],
        }
    )

    with pytest.raises(
        ValueError,
        match="k must be greater than zero",
    ):
        ensemble.search("query", k=k)


def test_search_combines_scores_using_weighted_rrf():
    ensemble, _ = make_ensemble(
        {
            "bm25": [
                make_result(1),
                make_result(2),
            ],
            "vector": [
                make_result(2),
                make_result(3),
            ],
        },
        weights={
            "bm25": 0.3,
            "vector": 0.7,
        },
        rrf_constant=1,
    )

    results = ensemble.search("query", k=3)

    assert [item["index"] for item in results] == [2, 3, 1]

    # Document 2: BM25 rank 2 and vector rank 1.
    assert results[0]["ensemble_score"] == pytest.approx(
        0.3 / 3 + 0.7 / 2
    )

    # Document 3: vector rank 2 only.
    assert results[1]["ensemble_score"] == pytest.approx(
        0.7 / 3
    )

    # Document 1: BM25 rank 1 only.
    assert results[2]["ensemble_score"] == pytest.approx(
        0.3 / 2
    )


def test_search_tracks_matched_retrievers():
    ensemble, _ = make_ensemble(
        {
            "bm25": [make_result(1)],
            "vector": [make_result(1)],
        }
    )

    results = ensemble.search("query")

    assert results[0]["matched_retrievers"] == [
        "bm25",
        "vector",
    ]


def test_search_passes_query_and_k_to_each_retriever():
    ensemble, retrievers = make_ensemble(
        {
            "bm25": [make_result(1)],
            "vector": [],
        }
    )

    ensemble.search("my query", k=2)

    assert retrievers["bm25"].calls == [
        ("my query", 2),
    ]

    assert retrievers["vector"].calls == [
        ("my query", 2),
    ]


def test_search_limits_results_to_k():
    ensemble, _ = make_ensemble(
        {
            "bm25": [
                make_result(1),
                make_result(2),
                make_result(3),
            ],
        }
    )

    assert len(ensemble.search("query", k=2)) == 2


def test_search_rejects_non_list_retriever_results():
    ensemble = EnsembleRetriever(
        retrievers={
            "function": lambda query: "not a list",
        }
    )

    with pytest.raises(
        TypeError,
        match="must return a list",
    ):
        ensemble.search("query")


def test_search_rejects_non_dictionary_results():
    ensemble, _ = make_ensemble(
        {
            "bm25": ["not a dictionary"],
        }
    )

    with pytest.raises(
        TypeError,
        match="each retrieval result must be a dictionary",
    ):
        ensemble.search("query")


def test_search_rejects_results_without_index():
    ensemble, _ = make_ensemble(
        {
            "bm25": [
                {
                    "document": "missing index",
                }
            ],
        }
    )

    with pytest.raises(
        ValueError,
        match="must contain an 'index' field",
    ):
        ensemble.search("query")


def test_search_rejects_unhashable_index():
    ensemble, _ = make_ensemble(
        {
            "bm25": [
                {
                    "index": [],
                    "document": "bad",
                }
            ],
        }
    )

    with pytest.raises(
        TypeError,
        match="index must be hashable",
    ):
        ensemble.search("query")


def test_search_supports_callable_retriever():
    calls = []

    def retriever(query):
        calls.append(query)
        return [make_result(7)]

    ensemble = EnsembleRetriever(
        retrievers={
            "callable": retriever,
        }
    )

    results = ensemble.search("query")

    assert results[0]["index"] == 7
    assert calls == ["query"]


def test_zero_weight_retriever_does_not_affect_score():
    ensemble, _ = make_ensemble(
        {
            "bm25": [make_result(1)],
            "vector": [make_result(2)],
        },
        weights={
            "bm25": 1.0,
            "vector": 0.0,
        },
    )

    results = ensemble.search("query", k=2)

    assert results[0]["index"] == 1
    assert results[1]["ensemble_score"] == 0.0


def test_search_ignores_duplicate_index_within_one_retriever():
    ensemble, _ = make_ensemble(
        {
            "bm25": [
                make_result(1),
                make_result(1, "duplicate"),
            ],
        }
    )

    results = ensemble.search("query")

    assert len(results) == 1
    assert results[0]["ensemble_score"] == pytest.approx(
        1 / 61
    )