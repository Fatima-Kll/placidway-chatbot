
import json
import time
from pathlib import Path
from urllib.parse import urlparse
from urllib.robotparser import RobotFileParser

import requests
from bs4 import BeautifulSoup

USER_AGENT = (
    "PlacidWayAssessmentBot/1.0 "
    "(educational assessment; contact: kelfatima22@gmail.com)"
)

URLS = [
    "https://www.placidway.com/profile/5/Alternative-Cancer-Treatment-by-ITC-Immunity-Therapy-Center",
    "https://www.placidway.com/package/7903/Alternative-Adenocarcinoma-Cancer-Treatment-Package-in-Tijuana-Mexico-by-ITC",
    "https://www.placidway.com/package/7902/Alternative-Adrenal-Cancer-Treatment-Package-in-Tijuana-Mexico-by-ITC",
    "https://www.placidway.com/package/7901/Alternative-Anal-Cancer-Treatment-Package-in-Tijuana-Mexico-by-ITC",
    "https://www.placidway.com/profile-videos/5/Alternative-Cancer-Treatment-by-ITC-Immunity-Therapy-Center",
    "https://www.placidway.com/price-list/5/Alternative-Cancer-Treatment-by-ITC-Immunity-Therapy-Center",
    "https://www.placidway.com/search-medical-pricings/Alternative-Medicine/All/1",
]

OUTPUT_DIR = Path("data")
OUTPUT_FILE = OUTPUT_DIR / "pages.json"

session = requests.Session()
session.headers.update({"User-Agent": USER_AGENT})


def check_robots(url):
    """Check whether our user agent may fetch this URL."""
    parsed = urlparse(url)
    robots_url = f"{parsed.scheme}://{parsed.netloc}/robots.txt"

    response = session.get(robots_url, timeout=20)
    response.raise_for_status()

    parser = RobotFileParser()
    parser.set_url(robots_url)
    parser.parse(response.text.splitlines())

    return parser.can_fetch(USER_AGENT, url)


def extract_main_text(soup):
    """Extract likely main content without following other links."""
    for tag in soup(["script", "style", "noscript", "nav", "footer"]):
        tag.decompose()

    # Prefer common main-content containers.
    candidates = [
        soup.find("main"),
        soup.find("article"),
        soup.find(id="content"),
        soup.find(class_="content"),
    ]

    for candidate in candidates:
        if candidate:
            text = candidate.get_text("\n", strip=True)
            if len(text) > 200:
                return text

    # Fallback: keep body text if no suitable container was found.
    body = soup.body or soup
    return body.get_text("\n", strip=True)


def ingest_page(url):
    print(f"\nChecking: {url} , flush=True")

    if not check_robots(url):
        print("Blocked by robots.txt; skipping.")
        return None

    response = session.get(url, timeout=30)
    response.raise_for_status()

    soup = BeautifulSoup(response.text, "html.parser")

    title = (
        soup.title.get_text(" ", strip=True)
        if soup.title
        else "Untitled page"
    )

    text = extract_main_text(soup)

    if not text.strip():
        print("No usable text extracted.")
        return None

    page = {
        "url": url,
        "title": title,
        "text": text,
    }

    print(f"Title: {title}")
    print(f"Extracted characters: {len(text)}")

    return page


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    pages = []

    for index, url in enumerate(URLS):
        # Keep page requests approximately one second apart.
        if index > 0:
            time.sleep(1)

        try:
            page = ingest_page(url)
            if page:
                pages.append(page)

        except requests.RequestException as exc:
            print(f"Request failed for {url}: {exc}")
        except Exception as exc:
            print(f"Unexpected error for {url}: {exc}")

    OUTPUT_FILE.write_text(
        json.dumps(pages, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print(f"\nSaved {len(pages)} pages to {OUTPUT_FILE}")


if __name__ == "__main__":
    main()