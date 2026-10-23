"""Reusable prompt templates for grounded RAG answer generation."""


class PromptTemplate:
    """Format a template using validated string variables."""

    def __init__(self, template: str) -> None:
        if not isinstance(template, str):
            raise TypeError("template must be a string.")

        if not template.strip():
            raise ValueError("template cannot be empty.")

        self.template = template

    def format(self, **kwargs: str) -> str:
        """Format the template after validating its variables."""
        for name, value in kwargs.items():
            if not isinstance(name, str):
                raise TypeError("template variable names must be strings.")

            if not isinstance(value, str):
                raise TypeError(
                    f"template variable '{name}' must be a string."
                )

        try:
            return self.template.format(**kwargs)
        except KeyError as exc:
            raise ValueError(
                f"missing template variable: {exc.args[0]}"
            ) from exc


RAG_ANSWER_TEMPLATE = PromptTemplate(
    template=(
        "You are a helpful question-answering assistant.\n\n"
        "Answer the user's question using only the provided context.\n"
        "Do not invent facts or use unsupported assumptions.\n"
        "If the context does not contain enough information, "
        "clearly say that you cannot answer from the provided context.\n"
        "When possible, identify the numbered context sources "
        "that support your answer.\n\n"
        "Context:\n"
        "{context}\n\n"
        "Question:\n"
        "{query}\n\n"
        "Answer:"
    )
)


def build_rag_prompt(query: str, context: str) -> str:
    """Build a grounded RAG prompt from a query and retrieved context."""
    if not isinstance(query, str):
        raise TypeError("query must be a string.")

    if not query.strip():
        raise ValueError("query cannot be empty.")

    if not isinstance(context, str):
        raise TypeError("context must be a string.")

    if not context.strip():
        raise ValueError("context cannot be empty.")

    return RAG_ANSWER_TEMPLATE.format(
        query=query.strip(),
        context=context.strip(),
    )