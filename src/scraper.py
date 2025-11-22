import time
from pathlib import Path

import pandas as pd
import requests
from bs4 import BeautifulSoup
from tqdm import tqdm

# Input: our cleaned catalog from make_catalog.py
CATALOG_PATH = Path("data/shl_catalog_clean.csv")

# Output: extra metadata scraped from SHL website
OUTPUT_PATH = Path("data/shl_web_catalog.csv")

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                  "AppleWebKit/537.36 (KHTML, like Gecko) "
                  "Chrome/120.0 Safari/537.36"
}


def fetch_product_page(url: str) -> str:
    """Download HTML of a single SHL product page."""
    try:
        resp = requests.get(url, headers=HEADERS, timeout=15)
        resp.raise_for_status()
        return resp.text
    except Exception as e:
        print(f"[WARN] Failed to fetch {url}: {e}")
        return ""


def parse_product(html: str, url: str) -> dict:
    """Parse title + description text from a product page."""
    if not html:
        return {
            "product_url": url,
            "product_title": "",
            "meta_description": "",
            "page_text": "",
        }

    soup = BeautifulSoup(html, "html.parser")

    # Product name – usually in the main <h1>
    h1 = soup.find("h1")
    title = h1.get_text(strip=True) if h1 else ""

    # Meta description (short marketing summary)
    meta_desc_tag = soup.find("meta", attrs={"name": "description"})
    meta_desc = ""
    if meta_desc_tag and meta_desc_tag.get("content"):
        meta_desc = meta_desc_tag["content"].strip()

    # Grab first few paragraphs as longer description
    paragraphs = [p.get_text(" ", strip=True) for p in soup.find_all("p")]
    page_text = " ".join(paragraphs[:8])  # limit length a bit
    if len(page_text) > 2000:
        page_text = page_text[:2000]

    return {
        "product_url": url,
        "product_title": title,
        "meta_description": meta_desc,
        "page_text": page_text,
    }


def build_web_catalog():
    print(f"Loading catalog from {CATALOG_PATH} ...")
    df = pd.read_csv(CATALOG_PATH)

    urls = sorted(df["Assessment_url"].dropna().unique())
    rows = []

    print(f"Scraping {len(urls)} SHL product pages...")
    for url in tqdm(urls):
        html = fetch_product_page(url)
        info = parse_product(html, url)
        rows.append(info)

        # polite delay so we don’t hammer SHL’s servers
        time.sleep(1.5)

    out_df = pd.DataFrame(rows)
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(OUTPUT_PATH, index=False)
    print(f"\nSaved scraped web catalog to: {OUTPUT_PATH}")


if __name__ == "__main__":
    build_web_catalog()
