from pathlib import Path


def load_text_file(file_path: str) -> str:
    path = Path(file_path)

    if path.suffix.lower() != ".txt":
        raise ValueError("Only .txt files are supported currently.")

    if not path.exists():
        raise FileNotFoundError(f"Document not found: {path}")

    return path.read_text(encoding="utf-8-sig")


if __name__ == "__main__":
    document_path = "data/raw/introduction.txt"
    content = load_text_file(document_path)

    print("Document loaded successfully!")
    print("-" * 40)
    print(content)
