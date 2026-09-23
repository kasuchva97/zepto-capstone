# Module 1 — Data Pipeline

Scrapes book listings from [books.toscrape.com](http://books.toscrape.com/) (a public
scraping-practice site — no login, no API key, no paid tier), cleans and types the
fields, converts price to INR using a fixed project-defined rate, loads everything
into a normalized SQLite schema, and demonstrates SQL + pandas querying against it.

## Setup

```bash
cd data_pipeline
py -3.11 -m venv .venv        # Windows; `python3.11 -m venv .venv` on macOS/Linux
.venv\Scripts\activate        # Windows; `source .venv/bin/activate` on macOS/Linux
pip install -r requirements.txt
```

**Requires Python 3.11** (what this was built and tested against). Newer Pythons
(3.13+) may lack prebuilt wheels for some dependencies here and can hang trying to
build them from source — check `python --version` first, and if it isn't 3.11.x,
install Python 3.11 from [python.org](https://www.python.org/downloads/) alongside
your existing version rather than swapping `py -3.11` for plain `python` above.

## Run end to end

```bash
python pipeline.py
```

This scrapes fresh data from the live site (needs internet), cleans it, (re)builds
`data/zepto_books.db` from scratch, runs all required SQL queries, verifies the
`pd.read_sql` / `pd.merge` equivalence, and writes `data/query_output.md` with every
query's SQL and printed output.

The repository already includes the regenerated outputs under `data/` (`books_raw.csv`,
`books_clean.csv`, `zepto_books.db`, `query_output.md`) from the last run, so grading
doesn't strictly require re-running the scrape — but `pipeline.py` is the exact,
from-scratch regeneration script if you want to reproduce it.

## Run tests

```bash
python -m pytest -q
```

29 tests covering price/rating/availability parsing, the FX conversion, the
missing-value fallback rules, the SQLite schema's PK/FK and CHECK constraints, and
that every required SQL query runs and that `pd.read_sql` matches `pd.merge` exactly
on the JOIN query.

## Design decisions

**Category selection.** Rather than paginating the generic "All products" catalogue,
the scraper walks three named categories — Mystery (32 books), Historical Fiction
(26), Classics (19) — for 77 books total, comfortably over the 60-row minimum and
giving each row a real category label to normalize against, rather than picking that
count from pagination cutoffs. Categories/URLs are discovered dynamically from the
site's sidebar, not hardcoded paths, so the scraper adapts if the site's category
listing changes. See `src/scrape.py:DEFAULT_CATEGORIES`.

**Encoding fix.** `books.toscrape.com` serves UTF-8 (`£`) without declaring a charset
in its `Content-Type` header, so `requests` falls back to guessing Latin-1 and mangles
the pound sign into `Â£`. Fixed by forcing `resp.encoding = "utf-8"` in `src/scrape.py`
before parsing — otherwise the price-cleaning regex would have silently produced wrong
numbers instead of failing loudly.

**Row-level cleaning failure policy** (see `src/clean.py` docstring for the full
rationale):
- `title` / `category` missing → **row dropped**. They're the row's identity; nothing
  sensible to impute.
- `price_gbp` unparseable → **median-imputed** (numeric field, per the assignment's
  stated fallback rule).
- `rating` unparseable → **median-imputed**, rounded back to the nearest int 1–5
  (treated as an ordinal-numeric field, so the same median rule applies).
- `in_stock` unparseable → **row dropped**, not imputed. It's boolean, not numeric, so
  median imputation doesn't apply, and guessing a stock status would fabricate data
  rather than approximate it.

In the actual scrape, all 77 rows parsed cleanly (0 drops, 0 imputations) — the
fallback paths are exercised and verified in `tests/test_clean.py`, not in the live run.

**Currency conversion.** `price_inr = price_gbp * 105.50`, the project's fixed baseline
rate, exactly as specified — a project-defined constant, not a live or dated market
rate, so no external lookup or date reference is needed. See `src/clean.py:FX_GBP_TO_INR`.
The optional live-FX-with-fallback stretch goal was not implemented; `price_inr` is
fully correct from the required fixed-rate baseline alone.

**Schema.** Two tables, `categories(category_id PK, category_name UNIQUE)` and
`books(book_id PK, title, price_gbp, price_inr, rating, in_stock, category_id FK)`,
with `CHECK` constraints on `rating` (1–5) and `in_stock` (0/1) for defense in depth
beyond what the assignment strictly requires. See `src/db.py:SCHEMA_SQL`.

**Queries** (`src/queries.py:QUERIES`, output in `data/query_output.md`):
| Query | Clauses demonstrated |
|---|---|
| `affordable_in_stock_books` | `SELECT` / `WHERE` |
| `top_10_most_expensive_books` | `ORDER BY` / `LIMIT` |
| `distinct_categories` | `DISTINCT` |
| `high_rated_books` | `IN` |
| `midrange_priced_books` | `BETWEEN` |
| `top_5_rated_books_per_category` | `JOIN` (+ window function `ROW_NUMBER`, `ORDER BY`) — the actual top 5 highest-rated books within each category, not just all books sorted by category |

**pandas equivalence.** `pipeline.py` reads `books` and `categories` back with
`pd.read_sql`, and separately reproduces the JOIN query's result with `pd.merge`
directly on the in-memory DataFrames (`src/queries.py:merge_equivalent_top_rated_per_category`).
The two are asserted equal (`DataFrame.equals`) in the pipeline run and in
`tests/test_db_and_queries.py::test_read_sql_and_merge_equivalent_for_join_query`.

## Files

```
data_pipeline/
├── pipeline.py              # end-to-end orchestrator (scrape -> clean -> load -> query)
├── src/
│   ├── scrape.py             # requests + BeautifulSoup scraper
│   ├── clean.py               # field parsing/typing + missing-value handling
│   ├── db.py                  # SQLite schema + loader
│   └── queries.py             # the 6 required SQL queries + pandas-merge equivalence
├── tests/                    # pytest suite (29 tests)
├── data/                      # generated: raw/clean CSVs, zepto_books.db, query_output.md
└── requirements.txt
```
