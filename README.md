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
- [x] `/support_assistant` — complete

## Setup

Each module ships its **own `requirements.txt`**, not one consolidated file. Reason:
the three modules need very different, occasionally conflicting dependency stacks
(plain `requests`/`bs4`/`pandas` vs. `scikit-learn`/`imbalanced-learn`/`seaborn` vs.
`chromadb`/`sentence-transformers`/`torch`/`langgraph`/`fastapi`), so isolating them
per module avoids version conflicts and keeps each module independently runnable.

**Requires Python 3.11.** All three modules were built and tested against it.
Newer Pythons (3.13, 3.14) do **not** yet have prebuilt wheels for some of the
heavier dependencies here (pandas, scikit-learn, chromadb, sentence-transformers,
torch) — `pip install` on those versions can hang for a very long time trying to
compile them from source instead of failing fast, which is worse than a quick error.
Check `python --version` / `python3 --version` first; if it isn't 3.11.x, install
Python 3.11 from [python.org](https://www.python.org/downloads/) (safe to install
alongside whatever version you already have — it won't replace it) and use the
version-specific commands below instead of the plain `python` command.

Create a separate virtual environment per module:

```bash
cd data_pipeline
py -3.11 -m venv .venv          # Windows; use `python3.11 -m venv .venv` on macOS/Linux
.venv\Scripts\activate           # Windows; use `source .venv/bin/activate` on macOS/Linux
pip install -r requirements.txt
```
(If `python --version` already reports 3.11.x, plain `python -m venv .venv` works too.)

Repeat the same pattern inside `analytics/` and `support_assistant/`.

## How to run each module

**Data Pipeline:**
```bash
cd data_pipeline && py -3.11 -m venv .venv && .venv\Scripts\activate
pip install -r requirements.txt
python pipeline.py        # scrape -> clean -> load SQLite -> run SQL queries
python -m pytest -q       # 29 tests
```
See [`data_pipeline/README.md`](data_pipeline/README.md) for full details.

**Analytics:**
```bash
cd analytics && py -3.11 -m venv .venv && .venv\Scripts\activate
pip install -r requirements.txt
jupyter nbconvert --to notebook --execute --inplace 01_eda.ipynb       # Part A: profile, clean, EDA story; writes titanic.csv
jupyter nbconvert --to notebook --execute --inplace 02_modeling.ipynb  # Part B: classification + regression pipeline
python -m pytest -q       # 43 tests
```
Both notebooks are already committed pre-executed with full output, so re-running
isn't required for grading. See [`analytics/README.md`](analytics/README.md) for
full details, all written interpretations, and the model comparison table.

**Support Assistant:**
```bash
cd support_assistant && py -3.11 -m venv .venv && .venv\Scripts\activate
pip install -r requirements.txt
uvicorn main:app --host 127.0.0.1 --port 8000   # then open http://127.0.0.1:8000/
python -m pytest -q       # 39 tests
```
`MOCK_LLM` is left at its default (deterministic, offline, no API key) for grading.
See [`support_assistant/README.md`](support_assistant/README.md) for the architecture
description, example `/ask` calls, and Docker instructions.

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
8 policy documents embedded locally (`sentence-transformers` `all-MiniLM-L6-v2`, no
API key) into ChromaDB, orchestrated by a 3-node LangGraph `StateGraph`
(`classify_intent` → conditional edge → `retrieve_and_answer` / `direct_answer`).
Retrieval always runs for real; only each node's final answer-generation step is
gated behind `MOCK_LLM` (default: deterministic canned logic, no LLM call at all —
the graded baseline; `MOCK_LLM=0` is an optional, ungraded Groq extension, not
exercised against a real account in this submission). Wrapped in FastAPI (`POST
/ask`, Pydantic-validated schema) with a custom light-theme demo page. **Note:**
Docker isn't installed in the environment this was built in, so the `Dockerfile` was
written and reviewed carefully but not build-tested end-to-end — see
[`support_assistant/README.md`](support_assistant/README.md) for details and to
verify it yourself.

## Git workflow

This repository's history includes a feature branch created, committed to at least
twice, and merged back into `main` (see `git log --graph --all`), as required by the
project's git-workflow rubric item.
