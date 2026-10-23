
import pytest

from src.prompt_templates import (
    PromptTemplate,
    RAG_ANSWER_TEMPLATE,
    build_rag_prompt,
)


def test_prompt_template_formats_variables():
    template = PromptTemplate("Question: {query}\nContext: {context}")

    result = template.format(query="What is RAG?", context="RAG uses retrieval.")

    assert result == (
        "Question: What is RAG?\nContext: RAG uses retrieval."
    )


def test_prompt_template_rejects_non_string_template():
    with pytest.raises(TypeError, match="template must be a string"):
        PromptTemplate(123)


@pytest.mark.parametrize("template", ["", "   ", "\n"])
def test_prompt_template_rejects_empty_template(template):
    with pytest.raises(ValueError, match="template cannot be empty"):
        PromptTemplate(template)


def test_prompt_template_rejects_non_string_variable():
    template = PromptTemplate("Question: {query}")

    with pytest.raises(TypeError, match="must be a string"):
        template.format(query=123)


def test_prompt_template_rejects_missing_variable():
    template = PromptTemplate("Question: {query}")

    with pytest.raises(ValueError, match="missing template variable"):
        template.format()


def test_prompt_template_supports_multiple_variables():
    template = PromptTemplate("{first} and {second}")

    assert template.format(first="RAG", second="LLM") == "RAG and LLM"


def test_rag_answer_template_contains_grounding_instructions():
    prompt = RAG_ANSWER_TEMPLATE.format(
        query="What is RAG?",
        context="[1] RAG combines retrieval with generation.",
    )

    assert "using only the provided context" in prompt
    assert "Do not invent facts" in prompt
    assert "What is RAG?" in prompt
    assert "RAG combines retrieval with generation." in prompt


def test_build_rag_prompt_formats_query_and_context():
    prompt = build_rag_prompt(
        "  What is RAG?  ",
        "  RAG combines retrieval and generation.  ",
    )

    assert "What is RAG?" in prompt
    assert "RAG combines retrieval and generation." in prompt
    assert prompt.startswith("You are a helpful question-answering assistant.")


@pytest.mark.parametrize("query", [None, 123, [], {}])
def test_build_rag_prompt_rejects_non_string_query(query):
    with pytest.raises(TypeError, match="query must be a string"):
        build_rag_prompt(query, "Some context")


@pytest.mark.parametrize("query", ["", "   ", "\n"])
def test_build_rag_prompt_rejects_empty_query(query):
    with pytest.raises(ValueError, match="query cannot be empty"):
        build_rag_prompt(query, "Some context")


@pytest.mark.parametrize("context", [None, 123, [], {}])
def test_build_rag_prompt_rejects_non_string_context(context):
    with pytest.raises(TypeError, match="context must be a string"):
        build_rag_prompt("Some question", context)


@pytest.mark.parametrize("context", ["", "   ", "\n"])
def test_build_rag_prompt_rejects_empty_context(context):
    with pytest.raises(ValueError, match="context cannot be empty"):
        build_rag_prompt("Some question", context)
