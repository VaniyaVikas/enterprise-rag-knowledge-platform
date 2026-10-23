
import pytest
from docx import Document

from src.docx_loader import load_docx_file


def create_test_docx(file_path, paragraphs):
    document = Document()

    for text in paragraphs:
        document.add_paragraph(text)

    document.save(file_path)


def test_load_docx_file_extracts_paragraph_text(tmp_path):
    docx_file = tmp_path / "document.docx"

    create_test_docx(
        docx_file,
        [
            "Enterprise RAG Knowledge Platform",
            "DOCX text extraction test",
        ],
    )

    content = load_docx_file(str(docx_file))

    assert "Enterprise RAG Knowledge Platform" in content
    assert "DOCX text extraction test" in content


def test_load_docx_file_joins_paragraphs_with_newlines(tmp_path):
    docx_file = tmp_path / "document.docx"

    create_test_docx(
        docx_file,
        ["First paragraph", "Second paragraph"],
    )

    content = load_docx_file(str(docx_file))

    assert content == "First paragraph\nSecond paragraph"


def test_load_docx_file_skips_empty_paragraphs(tmp_path):
    docx_file = tmp_path / "document.docx"

    create_test_docx(
        docx_file,
        ["First paragraph", "", "   ", "Second paragraph"],
    )

    content = load_docx_file(str(docx_file))

    assert content == "First paragraph\nSecond paragraph"


def test_load_docx_file_missing_file_raises_error(tmp_path):
    missing_file = tmp_path / "missing.docx"

    with pytest.raises(FileNotFoundError):
        load_docx_file(str(missing_file))


def test_load_docx_file_rejects_non_docx_file(tmp_path):
    text_file = tmp_path / "document.txt"
    text_file.write_text("This is a text file.", encoding="utf-8")

    with pytest.raises(ValueError, match="Only .docx files are supported."):
        load_docx_file(str(text_file))
