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

- [ ] `/data_pipeline` — in progress
- [ ] `/analytics` — not started
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

Filled in as each module lands — see that module's own README for exact commands
in the meantime (each module README is self-contained).

## Design decisions

Summarized per module below once each module is complete; full detail lives in each
module's own README.

### Data Pipeline
_TBD_

### Analytics
_TBD_

### Support Assistant
_TBD_

## Git workflow

This repository's history includes a feature branch created, committed to at least
twice, and merged back into `main` (see `git log --graph --all`), as required by the
project's git-workflow rubric item.
