
import pytest

from src.html_loader import load_html_file


def test_load_html_file_extracts_text(tmp_path):
    html_file = tmp_path / "document.html"

    html_file.write_text(
        "<html><body>"
        "<h1>Enterprise RAG</h1>"
        "<p>HTML text extraction test.</p>"
        "</body></html>",
        encoding="utf-8",
    )

    content = load_html_file(str(html_file))

    assert "Enterprise RAG" in content
    assert "HTML text extraction test." in content


def test_load_html_file_removes_script_and_style(tmp_path):
    html_file = tmp_path / "document.html"

    html_file.write_text(
        "<html><head>"
        "<style>body { color: red; }</style>"
        "</head><body>"
        "<p>Keep this text.</p>"
        "<script>alert('remove this');</script>"
        "<noscript>Remove this too.</noscript>"
        "</body></html>",
        encoding="utf-8",
    )

    content = load_html_file(str(html_file))

    assert "Keep this text." in content
    assert "color: red" not in content
    assert "alert(" not in content
    assert "Remove this too." not in content


def test_load_html_file_supports_htm_extension(tmp_path):
    html_file = tmp_path / "document.htm"

    html_file.write_text(
        "<html><body><p>HTM document</p></body></html>",
        encoding="utf-8",
    )

    content = load_html_file(str(html_file))

    assert "HTM document" in content


def test_load_html_file_missing_file_raises_error(tmp_path):
    missing_file = tmp_path / "missing.html"

    with pytest.raises(FileNotFoundError):
        load_html_file(str(missing_file))


def test_load_html_file_rejects_unsupported_extension(tmp_path):
    text_file = tmp_path / "document.txt"
    text_file.write_text("Plain text", encoding="utf-8")

    with pytest.raises(
        ValueError,
        match="Only .html and .htm files are supported.",
    ):
        load_html_file(str(text_file))


def test_load_html_file_handles_empty_html(tmp_path):
    html_file = tmp_path / "empty.html"
    html_file.write_text("", encoding="utf-8")

    content = load_html_file(str(html_file))

    assert content == ""
