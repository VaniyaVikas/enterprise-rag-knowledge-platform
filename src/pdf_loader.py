from pathlib import Path

from pypdf import PdfReader


def load_pdf_file(file_path: str) -> str:
    path = Path(file_path)

    if path.suffix.lower() != ".pdf":
        raise ValueError("Only .pdf files are supported.")

    if not path.exists():
        raise FileNotFoundError(f"Document not found: {path}")

    reader = PdfReader(path)

    pages = []

    for page in reader.pages:
        text = page.extract_text()

        if text:
            pages.append(text)

    return "\n".join(pages)


if __name__ == "__main__":
    document_path = "data/raw/sample.pdf"

    content = load_pdf_file(document_path)

    print("PDF loaded successfully!")
    print("-" * 40)
    print(content)