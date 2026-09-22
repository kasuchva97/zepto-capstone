"""Scrapes book listings from books.toscrape.com (a public scraping-practice site).

Design: pick a handful of named categories (rather than the "All products"
catalogue) so each row is naturally tagged with a real category name, and walk
each category's pagination until exhausted.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, asdict

import requests
from bs4 import BeautifulSoup

BASE_URL = "http://books.toscrape.com/"
HEADERS = {"User-Agent": "zepto-capstone-data-pipeline/1.0 (educational scraping exercise)"}
REQUEST_DELAY_SECONDS = 0.2

# Categories chosen to comfortably clear the >= 60 row requirement across
# >= 3 categories (Mystery=32, Historical Fiction=26, Classics=19 -> 77 rows).
DEFAULT_CATEGORIES = ["Mystery", "Historical Fiction", "Classics"]


@dataclass
class RawBook:
    title: str
    price: str
    star_rating: str
    availability: str
    category: str


def _get_soup(url: str) -> BeautifulSoup:
    resp = requests.get(url, headers=HEADERS, timeout=15)
    resp.raise_for_status()
    resp.encoding = "utf-8"  # site serves UTF-8 (£ etc.) without a charset header, so requests would otherwise mis-guess latin-1
    return BeautifulSoup(resp.text, "html.parser")


def get_category_urls() -> dict[str, str]:
    """Returns {category_name: absolute_url} for every category on the site."""
    soup = _get_soup(BASE_URL + "index.html")
    links = soup.select("div.side_categories ul li ul li a")
    return {a.get_text(strip=True): BASE_URL + a["href"] for a in links}


def scrape_category(category_name: str, start_url: str) -> list[RawBook]:
    """Walks a category's pagination and returns every book listed in it."""
    books: list[RawBook] = []
    url = start_url
    while url:
        soup = _get_soup(url)
        for art in soup.select("article.product_pod"):
            title = art.h3.a["title"]
            price = art.select_one("p.price_color").get_text(strip=True)
            rating = art.select_one("p.star-rating")["class"][1]  # e.g. "star-rating Three" -> "Three"
            availability = art.select_one("p.instock.availability").get_text(strip=True)
            books.append(RawBook(title, price, rating, availability, category_name))

        next_link = soup.select_one("li.next a")
        if next_link:
            url = url.rsplit("/", 1)[0] + "/" + next_link["href"]
        else:
            url = None
        time.sleep(REQUEST_DELAY_SECONDS)
    return books


def scrape_categories(category_names: list[str] = DEFAULT_CATEGORIES) -> list[dict]:
    all_urls = get_category_urls()
    missing = [c for c in category_names if c not in all_urls]
    if missing:
        raise ValueError(f"Categories not found on site: {missing}. Available: {sorted(all_urls)}")

    rows: list[dict] = []
    for name in category_names:
        rows.extend(asdict(b) for b in scrape_category(name, all_urls[name]))
    return rows


if __name__ == "__main__":
    import json

    data = scrape_categories()
    print(f"Scraped {len(data)} books across {len(DEFAULT_CATEGORIES)} categories.")
    print(json.dumps(data[:3], indent=2))
