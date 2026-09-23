# Zepto Data & AI Platform — Capstone Project

One connected platform, three modules, built and graded together:

| Module | Path | What it does | Marks |
|---|---|---|---|
| Data Pipeline | [`/data_pipeline`](data_pipeline/README.md) | Scrapes books.toscrape.com, cleans it, converts currency, loads it into a normalized SQLite schema, queries it with SQL and pandas | 25 |
| Analytics | [`/analytics`](analytics/README.md) | Profiles, cleans and visualizes the Titanic dataset, then builds and rigorously evaluates a full classification + regression modeling pipeline | 50 |
| Support Assistant | [`/support_assistant`](support_assistant/README.md) | A LangGraph-orchestrated RAG service (ChromaDB + sentence-transformers) answering Zepto policy questions, wrapped in FastAPI, fully testable offline via a deterministic mock LLM mode | 25 |

> This README is filled in progressively as each module is completed. See `PLANNING.md`
> for the full internal task checklist (not a deliverable — just working notes).

## Status

- [x] `/data_pipeline` — complete
- [x] `/analytics` — complete
- [ ] `/support_assistant` — not started

## Setup

Each module ships its **own `requirements.txt`**, not one consolidated file. Reason:
the three modules need very different, occasionally conflicting dependency stacks
(plain `requests`/`bs4`/`pandas` vs. `scikit-learn`/`imbalanced-learn`/`seaborn` vs.
`chromadb`/`sentence-transformers`/`torch`/`langgraph`/`fastapi`), so isolating them
per module avoids version conflicts and keeps each module independently runnable.

All modules were built and tested against **Python 3.11** (newer Pythons, e.g. 3.13+,
don't yet have stable wheels for some of the heavier ML/embedding libraries on Windows).
Create a separate virtual environment per module:

```bash
cd data_pipeline
python -m venv .venv
.venv\Scripts\activate   # Windows; use `source .venv/bin/activate` on macOS/Linux
pip install -r requirements.txt
```

Repeat the same pattern inside `analytics/` and `support_assistant/`.

## How to run each module

**Data Pipeline:**
```bash
cd data_pipeline && python -m venv .venv && .venv\Scripts\activate
pip install -r requirements.txt
python pipeline.py        # scrape -> clean -> load SQLite -> run SQL queries
python -m pytest -q       # 28 tests
```
See [`data_pipeline/README.md`](data_pipeline/README.md) for full details.

**Analytics:**
```bash
cd analytics && python -m venv .venv && .venv\Scripts\activate
pip install -r requirements.txt
jupyter nbconvert --to notebook --execute --inplace 01_eda.ipynb       # Part A: profile, clean, EDA story; writes titanic.csv
jupyter nbconvert --to notebook --execute --inplace 02_modeling.ipynb  # Part B: classification + regression pipeline
python -m pytest -q       # 43 tests
```
Both notebooks are already committed pre-executed with full output, so re-running
isn't required for grading. See [`analytics/README.md`](analytics/README.md) for
full details, all written interpretations, and the model comparison table.

**Support Assistant:** to be filled in once that module lands.

## Design decisions

Summarized per module below once each module is complete; full detail lives in each
module's own README.

### Data Pipeline
Scrapes 3 named categories (Mystery/Historical Fiction/Classics, 77 books total,
comfortably over the 60-row minimum) rather than paginating the generic catalogue, so
every row carries a real category label. Fixed a UTF-8 encoding bug where the site's
undeclared charset made `requests` mangle `£` into `Â£`. Unparseable numeric fields
(price, rating) are median-imputed; unparseable identity fields (title, category) or
the boolean `in_stock` field are dropped, since a boolean can't be meaningfully
median-imputed. `price_inr` uses the fixed 1 GBP = 105.50 INR baseline exactly as
specified. Full rationale in [`data_pipeline/README.md`](data_pipeline/README.md).

### Analytics
One dataset loaded once (`sns.load_dataset` in `01_eda.ipynb`, cached to
`titanic.csv`), cleaned with a measured-percentage threshold rule (drop `deck` at
77% missing, impute `age` at 20% missing by pclass+sex median, drop 2 rows for
`embarked` at 0.2% missing), then a full classification pipeline (Logistic
Regression / Decision Tree / Random Forest, all inside `ColumnTransformer`+`Pipeline`
so preprocessing is structurally fit-on-train-only) plus a `fare` regression
side-task. Logistic Regression won on test F1 (0.756) even after `GridSearchCV`-tuning
the Random Forest, and is the recommended deploy candidate. Full metrics, all written
interpretations, and the model comparison table are in
[`analytics/README.md`](analytics/README.md).

### Support Assistant
_TBD_

## Git workflow

This repository's history includes a feature branch created, committed to at least
twice, and merged back into `main` (see `git log --graph --all`), as required by the
project's git-workflow rubric item.
