from src.process_document import process_document


def test_process_document_returns_chunks():
    chunks = process_document(
        "data/raw/introduction.txt",
        chunk_size=100,
        overlap=20,
    )

    assert len(chunks) > 0
    assert all(isinstance(chunk, str) for chunk in chunks)


def test_process_document_preserves_document_text():
    chunks = process_document(
        "data/raw/introduction.txt",
        chunk_size=100,
        overlap=20,
    )

    combined_text = "".join(
        chunk if index == 0 else chunk[20:]
        for index, chunk in enumerate(chunks)
    )

    original_text = (
        "Enterprise RAG Knowledge Platform\n\n"
        "RAG stands for Retrieval-Augmented Generation.\n"
        "RAG systems retrieve relevant information from documents\n"
        "and use that information to generate grounded answers.\n\n"
        "LangChain helps developers build applications powered by language models.\n"
        "Document loading is the first step in a typical RAG pipeline.\n"
    )

    assert combined_text == original_text
