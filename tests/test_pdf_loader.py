import pytest

from src.pdf_loader import load_pdf_file


def create_test_pdf(file_path):
    pdf_content = b"""%PDF-1.4
1 0 obj
<< /Type /Catalog /Pages 2 0 R >>
endobj
2 0 obj
<< /Type /Pages /Kids [3 0 R] /Count 1 >>
endobj
3 0 obj
<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792]
/Resources << /Font << /F1 4 0 R >> >>
/Contents 5 0 R >>
endobj
4 0 obj
<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>
endobj
5 0 obj
<< /Length 62 >>
stream
BT
/F1 18 Tf
72 720 Td
(Enterprise RAG PDF Test) Tj
ET
endstream
endobj
xref
0 6
0000000000 65535 f 
0000000009 00000 n 
0000000058 00000 n 
0000000115 00000 n 
0000000274 00000 n 
0000000344 00000 n 
trailer
<< /Size 6 /Root 1 0 R >>
startxref
455
%%EOF
"""

    file_path.write_bytes(pdf_content)


def test_load_pdf_file_missing_file_raises_error(tmp_path):
    missing_file = tmp_path / "missing.pdf"

    with pytest.raises(FileNotFoundError):
        load_pdf_file(str(missing_file))


def test_load_pdf_file_rejects_non_pdf_file(tmp_path):
    text_file = tmp_path / "document.txt"
    text_file.write_text(
        "This is not a PDF.",
        encoding="utf-8",
    )

    with pytest.raises(ValueError):
        load_pdf_file(str(text_file))


def test_load_pdf_file_returns_string_for_valid_pdf(tmp_path):
    pdf_file = tmp_path / "document.pdf"

    create_test_pdf(pdf_file)

    content = load_pdf_file(str(pdf_file))

    assert isinstance(content, str)


def test_load_pdf_file_extracts_pdf_text(tmp_path):
    pdf_file = tmp_path / "text.pdf"

    create_test_pdf(pdf_file)

    content = load_pdf_file(str(pdf_file))

    assert "Enterprise RAG PDF Test" in content


def test_load_pdf_file_handles_blank_pdf_page(tmp_path):
    pdf_file = tmp_path / "blank.pdf"

    create_test_pdf(pdf_file)

    content = load_pdf_file(str(pdf_file))

    assert isinstance(content, str)