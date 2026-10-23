
import pytest
import requests

from unittest.mock import Mock, patch

from src.web_loader import load_web_page


def test_load_web_page_extracts_text_and_removes_unwanted_content():
    html = """
    <html>
        <head>
            <style>body { color: red; }</style>
        </head>
        <body>
            <h1>Enterprise RAG</h1>
            <p>Web extraction test.</p>
            <script>alert('remove this');</script>
            <noscript>Remove this too.</noscript>
        </body>
    </html>
    """

    response = Mock()
    response.text = html

    with patch("src.web_loader.requests.get", return_value=response):
        content = load_web_page("https://example.com")

    assert "Enterprise RAG" in content
    assert "Web extraction test." in content
    assert "color: red" not in content
    assert "alert(" not in content
    assert "Remove this too." not in content


def test_load_web_page_raises_http_error():
    response = Mock()
    response.raise_for_status.side_effect = requests.HTTPError("404 Not Found")

    with patch("src.web_loader.requests.get", return_value=response):
        with pytest.raises(requests.HTTPError, match="404 Not Found"):
            load_web_page("https://example.com")


def test_load_web_page_rejects_invalid_scheme():
    with pytest.raises(ValueError, match="URL must use HTTP or HTTPS."):
        load_web_page("ftp://example.com")


def test_load_web_page_rejects_missing_hostname():
    with pytest.raises(ValueError, match="URL must include a valid hostname."):
        load_web_page("https://")


@pytest.mark.parametrize("timeout", [0, -1])
def test_load_web_page_rejects_non_positive_timeout(timeout):
    with pytest.raises(ValueError, match="timeout must be greater than zero."):
        load_web_page("https://example.com", timeout=timeout)


def test_load_web_page_passes_timeout_to_request():
    response = Mock()
    response.text = "<html><body><p>Test page</p></body></html>"

    with patch("src.web_loader.requests.get", return_value=response) as mock_get:
        load_web_page("https://example.com", timeout=5)

    mock_get.assert_called_once_with(
        "https://example.com",
        timeout=5,
        headers={"User-Agent": "EnterpriseRAGKnowledgePlatform/1.0"},
    )


def test_load_web_page_propagates_request_timeout_error():
    with patch(
        "src.web_loader.requests.get",
        side_effect=requests.Timeout("Request timed out"),
    ):
        with pytest.raises(requests.Timeout, match="Request timed out"):
            load_web_page("https://example.com")

