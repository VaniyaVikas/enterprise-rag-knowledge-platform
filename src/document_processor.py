
from src.document_loader import load_text_file
from src.chunking import split_text
from src.document_chunk import DocumentChunk
from src.config_loader import load_chunking_config


def process_document_with_metadata(
    file_path: str,
    chunk_size: int = 100,
    overlap: int = 20,
) -> list[DocumentChunk]:
    text = load_text_file(file_path)

    text_chunks = split_text(
        text,
        chunk_size=chunk_size,
        overlap=overlap,
    )

    document_chunks = [
        DocumentChunk(
            text=chunk,
            source=file_path,
            chunk_id=index,
        )
        for index, chunk in enumerate(text_chunks)
    ]

    return document_chunks


def process_document_from_config(
    file_path: str,
    config_path: str = "configs/chunking.yaml",
) -> list[DocumentChunk]:
    config = load_chunking_config(config_path)

    return process_document_with_metadata(
        file_path,
        chunk_size=config["chunk_size"],
        overlap=config["overlap"],
    )


if __name__ == "__main__":
    document_path = "data/raw/introduction.txt"

    chunks = process_document_from_config(document_path)

    print(f"Total chunks: {len(chunks)}")

    for chunk in chunks:
        print(f"\n--- Chunk {chunk.chunk_id} ---")
        print(f"Source: {chunk.source}")
        print(f"Metadata: {chunk.metadata}")
        print(f"Text: {chunk.text}")
