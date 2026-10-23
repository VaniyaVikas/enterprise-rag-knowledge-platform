from src.document_loader import load_text_file
from src.chunking import split_text


def process_document(file_path: str, chunk_size: int = 100, overlap: int = 20) -> list[str]:
    text = load_text_file(file_path)

    chunks = split_text(
        text,
        chunk_size=chunk_size,
        overlap=overlap,
    )

    return chunks


if __name__ == "__main__":
    document_path = "data/raw/introduction.txt"

    chunks = process_document(
        document_path,
        chunk_size=100,
        overlap=20,
    )

    print(f"Total chunks: {len(chunks)}")

    for index, chunk in enumerate(chunks, start=1):
        print(f"\n--- Chunk {index} ---")
        print(chunk)
