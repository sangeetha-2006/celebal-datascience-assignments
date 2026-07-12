"""
Step 1a: Web Scraping
Fetches raw HTML from a list of URLs and extracts visible text + basic metadata.
Saves one JSON file per page into data/raw/.

Usage:
    python -m app.ingestion.scraper --urls urls.txt
    (urls.txt = one URL per line)
"""
import argparse
import hashlib
import json
import time
from pathlib import Path
from urllib.parse import urlparse

import requests
from bs4 import BeautifulSoup

RAW_DIR = Path("data/raw")
RAW_DIR.mkdir(parents=True, exist_ok=True)

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (compatible; MemChatBot/1.0; "
        "+https://example.com/bot-info)"
    )
}


def url_to_filename(url: str) -> str:
    """Deterministic filename per URL so re-scraping overwrites cleanly."""
    h = hashlib.sha256(url.encode()).hexdigest()[:16]
    domain = urlparse(url).netloc.replace(".", "_")
    return f"{domain}_{h}.json"


def fetch_page(url: str, timeout: int = 15) -> dict | None:
    try:
        resp = requests.get(url, headers=HEADERS, timeout=timeout)
        resp.raise_for_status()
    except requests.RequestException as e:
        print(f"[scraper] FAILED {url}: {e}")
        return None

    soup = BeautifulSoup(resp.text, "lxml")

    # Strip non-content tags before extracting text
    for tag in soup(["script", "style", "noscript", "nav", "footer", "header", "form"]):
        tag.decompose()

    title = soup.title.string.strip() if soup.title and soup.title.string else ""
    text = " ".join(soup.get_text(separator=" ").split())

    return {
        "url": url,
        "title": title,
        "text": text,
        "fetched_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "char_count": len(text),
    }


def scrape_urls(urls: list[str], delay: float = 1.0) -> list[Path]:
    """Scrape each URL, save to data/raw/, return list of saved file paths."""
    saved = []
    for url in urls:
        url = url.strip()
        if not url:
            continue
        page = fetch_page(url)
        if page is None:
            continue
        out_path = RAW_DIR / url_to_filename(url)
        out_path.write_text(json.dumps(page, ensure_ascii=False, indent=2), encoding="utf-8")
        saved.append(out_path)
        print(f"[scraper] saved {url} -> {out_path} ({page['char_count']} chars)")
        time.sleep(delay)  # be polite to servers
    return saved


def main():
    parser = argparse.ArgumentParser(description="Scrape a list of URLs into data/raw/")
    parser.add_argument("--urls", type=str, required=True, help="Path to a text file, one URL per line")
    parser.add_argument("--delay", type=float, default=1.0, help="Delay between requests (seconds)")
    args = parser.parse_args()

    urls = Path(args.urls).read_text(encoding="utf-8").splitlines()
    scrape_urls(urls, delay=args.delay)


if __name__ == "__main__":
    main()
