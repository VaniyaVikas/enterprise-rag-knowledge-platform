
import pytest

from src.query_expansion import QueryExpander


def test_expand_preserves_original_query():
    expander = QueryExpander()

    results = expander.expand("What is RAG?")

    assert results[0] == "What is RAG?"


def test_expand_replaces_rag_with_full_form():
    expander = QueryExpander()

    results = expander.expand("What is RAG?")

    assert "What is retrieval augmented generation?" in results


def test_expand_is_case_insensitive():
    expander = QueryExpander()

    results = expander.expand("Explain LLM")

    assert "Explain large language model" in results


def test_expand_supports_multi_word_terms():
    expander = QueryExpander()

    results = expander.expand("Compare vector database tools")

    assert "Compare vector store tools" in results


def test_expand_supports_custom_synonyms():
    expander = QueryExpander(
        synonym_map={"python": ["programming language"]},
    )

    results = expander.expand("Learn Python")

    assert "Learn programming language" in results


def test_expand_does_not_replace_partial_words():
    expander = QueryExpander(
        synonym_map={"rag": ["retrieval augmented generation"]},
    )

    results = expander.expand("frag system")

    assert results == ["frag system"]


def test_expand_does_not_duplicate_queries():
    expander = QueryExpander(
        synonym_map={"ai": ["AI", "artificial intelligence"]},
    )

    results = expander.expand("AI tools")

    assert len(results) == len(
        {item.casefold() for item in results}
    )


def test_expand_respects_max_expansions():
    expander = QueryExpander()

    results = expander.expand(
        "cheap LLM",
        max_expansions=2,
    )

    assert len(results) <= 2
    assert results[0] == "cheap LLM"


def test_expand_rejects_non_string_query():
    expander = QueryExpander()

    with pytest.raises(TypeError, match="query must be a string"):
        expander.expand(123)


def test_expand_rejects_empty_query():
    expander = QueryExpander()

    with pytest.raises(ValueError, match="query cannot be empty"):
        expander.expand("   ")


@pytest.mark.parametrize("value", [0, -1])
def test_expand_rejects_non_positive_max_expansions(value):
    expander = QueryExpander()

    with pytest.raises(
        ValueError,
        match="max_expansions must be greater than zero",
    ):
        expander.expand("RAG", max_expansions=value)


@pytest.mark.parametrize("value", [True, 1.5, "3"])
def test_expand_rejects_non_integer_max_expansions(value):
    expander = QueryExpander()

    with pytest.raises(
        TypeError,
        match="max_expansions must be an integer",
    ):
        expander.expand("RAG", max_expansions=value)


def test_constructor_rejects_non_dictionary():
    with pytest.raises(
        TypeError,
        match="synonym_map must be a dictionary",
    ):
        QueryExpander(synonym_map=["rag"])


def test_constructor_rejects_non_string_term():
    with pytest.raises(
        TypeError,
        match="Each synonym key must be a string",
    ):
        QueryExpander(synonym_map={123: ["example"]})


def test_constructor_rejects_empty_term():
    with pytest.raises(
        ValueError,
        match="Synonym keys cannot be empty",
    ):
        QueryExpander(synonym_map={"  ": ["example"]})


def test_constructor_rejects_non_list_synonyms():
    with pytest.raises(
        TypeError,
        match="Synonyms for 'rag' must be a list",
    ):
        QueryExpander(synonym_map={"rag": "retrieval"})


def test_constructor_rejects_non_string_synonym():
    with pytest.raises(
        TypeError,
        match="Each synonym must be a string",
    ):
        QueryExpander(synonym_map={"rag": [123]})


def test_constructor_rejects_empty_synonym():
    with pytest.raises(
        ValueError,
        match="Synonyms cannot be empty",
    ):
        QueryExpander(synonym_map={"rag": ["  "]})
