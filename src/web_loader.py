
from urllib.parse import urlparse

import requests
from bs4 import BeautifulSoup


def load_web_page(
    url: str,
    timeout: int = 10,
) -> str:
    parsed_url = urlparse(url)

    if parsed_url.scheme not in {"http", "https"}:
        raise ValueError("URL must use HTTP or HTTPS.")

    if not parsed_url.netloc:
        raise ValueError("URL must include a valid hostname.")

    if timeout <= 0:
        raise ValueError("timeout must be greater than zero.")

    response = requests.get(
        url,
        timeout=timeout,
        headers={"User-Agent": "EnterpriseRAGKnowledgePlatform/1.0"},
    )

    response.raise_for_status()

    soup = BeautifulSoup(response.text, "html.parser")

    for element in soup(["script", "style", "noscript"]):
        element.decompose()

    return soup.get_text(separator="\n", strip=True)


if __name__ == "__main__":
    page_url = "https://example.com"

    content = load_web_page(page_url)

    print("Web page loaded successfully!")
    print("-" * 40)
    print(content)
