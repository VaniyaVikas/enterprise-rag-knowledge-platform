
from pathlib import Path

from docx import Document


def load_docx_file(file_path: str) -> str:
    path = Path(file_path)

    if path.suffix.lower() != ".docx":
        raise ValueError("Only .docx files are supported.")

    if not path.exists():
        raise FileNotFoundError(f"Document not found: {path}")

    document = Document(path)

    paragraphs = [
        paragraph.text
        for paragraph in document.paragraphs
        if paragraph.text.strip()
    ]

    return "\n".join(paragraphs)


if __name__ == "__main__":
    document_path = "data/raw/sample.docx"

    content = load_docx_file(document_path)

    print("DOCX loaded successfully!")
    print("-" * 40)
    print(content)
