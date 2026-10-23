
from pathlib import Path

from bs4 import BeautifulSoup


def load_html_file(file_path: str) -> str:
    path = Path(file_path)

    if path.suffix.lower() not in {".html", ".htm"}:
        raise ValueError("Only .html and .htm files are supported.")

    if not path.exists():
        raise FileNotFoundError(f"Document not found: {path}")

    html_content = path.read_text(encoding="utf-8-sig")

    soup = BeautifulSoup(html_content, "html.parser")

    for element in soup(["script", "style", "noscript"]):
        element.decompose()

    text = soup.get_text(separator="\n", strip=True)

    return text


if __name__ == "__main__":
    document_path = "data/raw/sample.html"

    content = load_html_file(document_path)

    print("HTML loaded successfully!")
    print("-" * 40)
    print(content)
