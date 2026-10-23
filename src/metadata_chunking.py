
from src.advanced_chunking import split_text_recursively
from src.document_chunk import DocumentChunk


def split_text_with_metadata(
    text: str,
    source: str,
    chunk_size: int = 500,
    chunk_overlap: int = 50,
    separators: list[str] | None = None,
) -> list[DocumentChunk]:
    if not isinstance(source, str):
        raise TypeError("source must be a string.")

    if not source.strip():
        raise ValueError("source cannot be empty.")

    text_chunks = split_text_recursively(
        text,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=separators,
    )

    document_chunks = [
        DocumentChunk(
            text=chunk,
            source=source,
            chunk_id=index,
        )
        for index, chunk in enumerate(text_chunks)
    ]

    return document_chunks
