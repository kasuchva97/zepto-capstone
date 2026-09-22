# Capstone Planning & Task Checklist (internal working notes)

This file tracks every requirement from the assignment brief so nothing gets skipped.
It is a working document — check items off as they're implemented and verified.
Not a deliverable in itself; the actual write-ups live in the module/root READMEs.

Environment note: Python 3.14 is the machine default but is too new for reliable
wheels of pandas/scikit-learn/chromadb/torch on Windows. Each module gets its own
venv built with **Python 3.11** (`C:\Users\sree\AppData\Local\Programs\Python\Python311\python.exe`,
invoked as `py -3.11`). This is why requirements.txt is per-module, not consolidated —
documented in the root README.

UI/UX note (per user instruction): wherever a UI naturally belongs (the support
assistant's demo frontend), it must be a deliberately designed light-theme page —
not a generic default-Bootstrap/unstyled AI-generated look. Analytics/data-pipeline
outputs stay as notebooks/scripts/plots per spec (no UI required there).

Git workflow (graded once, across whole repo): create a feature branch, commit to
it >= 2 times, merge back to main. Doing this for Module 1's implementation is
sufficient to satisfy the rubric, but natural feature branches per module are fine too.

---

## Module 1 — /data_pipeline (25 marks)

- [ ] Scrape books.toscrape.com: >=60 books across >=3 categories (requests + BeautifulSoup)
  - [ ] capture title, price (GBP), star_rating (text), availability (text), category
- [ ] Clean fields:
  - [ ] price -> price_gbp (float, strip currency symbol)
  - [ ] star_rating text (One..Five) -> rating (int 1-5)
  - [ ] availability text -> in_stock (bool)
  - [ ] handle any unparseable rows: median-impute numeric or drop + justify in README
- [ ] price_inr = price_gbp * 105.50 (fixed baseline rate, stated exactly in README)
  - [ ] (optional, skip) live FX lookup with status-code handling + fallback
- [ ] SQLite schema: >=2 tables with PK/FK (categories, books)
- [ ] Insert cleaned data via sqlite3 or to_sql
- [ ] >=5 SQL queries covering: SELECT/WHERE, ORDER BY, LIMIT, DISTINCT, IN/BETWEEN, >=1 JOIN
  - [ ] save query strings + printed output
- [ ] pd.read_sql for >=2 queries; pd.merge reproduces the JOIN query from in-memory DataFrames; show equivalence
- [ ] module README: install/run steps, cleaning/parsing decisions, exact FX rate
- [ ] repo includes sqlite db file OR exact regeneration script
- [ ] tests for: price parsing, rating mapping, in_stock parsing, FX conversion, schema/FK integrity, query correctness
- [ ] feature branch created, >=2 commits, merged to main (whole-repo git rubric)

## Module 2 — /analytics (50 marks)

Part A — profiling/cleaning/EDA
- [ ] load via sns.load_dataset('titanic'); df.info/describe/shape; % missing per column
- [ ] save titanic.csv immediately after load (offline fallback); load exactly once, never reloaded for modeling
- [ ] missing-value handling: <5% drop rows, 5-30% impute, very-high% -> drop col or "missing" category (state % per column + justification)
- [ ] univariate: histogram + boxplot for age & fare; IQR outlier counts for both; mean/median/mode for fare + skew direction written
- [ ] bivariate: survival rate by sex, by pclass, by sex+pclass (boolean masking); correlation matrix on exactly
      [survived, pclass, age, sibsp, parch, fare] (exclude adult_male, alone); heatmap; top-2 |corr| pairs interpreted
- [ ] multivariate data story: >=4 distinct charts, each with 2-4 sentence interpretation
- [ ] z-score standardize age & fare (EDA sanity check only, not fed into modeling pipeline); before/after mean~0/std~1 shown

Part B — modeling
- [ ] stratified train/test split on survived; justify via class balance
- [ ] preprocessing fit on train only (ColumnTransformer + Pipeline): impute, encode sex/embarked, scale numerics
- [ ] train LogisticRegression, DecisionTree, RandomForest on same split
  - [ ] plot_tree for decision tree w/ feature & class names
- [ ] evaluate all 3: confusion matrix, accuracy, precision, recall, F1, ROC/AUC; comparison table
- [ ] imbalance comparison: baseline vs class_weight='balanced' vs SMOTE (train-fold only); compare P/R/F1; written conclusion
- [ ] GridSearchCV over RF n_estimators/max_depth/max_features; RandomForestClassifier(oob_score=True); report best params + OOB score
- [ ] regression side-task: predict fare from other features (multivariate linear regression); MAE, RMSE, R2, Adjusted R2; residual plot; heteroscedasticity conclusion
- [ ] final comparison table: classifier metrics block + regression metrics block (separate groups) + 3-5 sentence recommendation
- [ ] joblib.dump full fitted Pipeline (preprocessing+estimator) as one object; reload + predict-on-raw-input demo
- [ ] tests for: missing-value thresholds applied correctly, no test-set leakage (preprocessing fit only on train), pipeline round-trips via joblib, metric functions correct
- [ ] two notebooks (01_eda.ipynb, 02_modeling.ipynb) sharing committed titanic.csv, or equivalent clearly-ordered structure

## Module 3 — /support_assistant (25 marks)

- [ ] 8 corpus docs copied verbatim into docs/doc_01.txt..doc_08.txt
- [ ] chunk + embed with all-MiniLM-L6-v2 (sentence-transformers) -> store in ChromaDB collection
- [ ] structured prompt template: role/context/task/format/length + negative constraint + few-shot example (actual text, used by optional real-LLM path)
- [ ] LangGraph StateGraph, TypedDict state, >=3 nodes:
  - [ ] classify_intent: keyword heuristic (delivery/return/refund/membership/tracking/cancel/gift card/support hours) when MOCK_LLM unset/1
  - [ ] retrieve_and_answer: real embedding+Chroma top-3 retrieval always; mock answer = "Based on the retrieved context: {top_chunk_snippet}"
  - [ ] direct_answer: mock = fixed canned string
  - [ ] conditional edge routes policy_question -> retrieve_and_answer, general_question -> direct_answer
- [ ] Pydantic response schema: answer (str), sources (list[str]), confidence (float 0-1); deterministic in mock mode
  - [ ] (optional) retry-on-validation-failure logic present for real-LLM path
- [ ] FastAPI POST /ask; Pydantic request {"query": str}; >=2 example calls recorded (one retrieval, one not), MOCK_LLM default
- [ ] Dockerfile: builds and runs locally, serves /ask (uvicorn main:app --host 0.0.0.0 --port 7860)
- [ ] README: RAG architecture description (ingestion->embedding->retrieval->generation, which file/node does what) + MOCK_LLM branch explanation
- [ ] tests for: classify_intent heuristic both branches, retrieval returns correct-doc chunks, mock response schema validity, /ask endpoint via TestClient
- [ ] light-theme custom demo frontend for /ask (extra, not required by spec, per user's UI request) — simple static page, not generic

## Root repo

- [ ] root README.md: setup (per-module requirements.txt, why), how to run each module end to end, design-decision summary per module
- [ ] .gitignore (venvs, __pycache__, .env, chroma persisted dir if large, etc.)
- [ ] verify `git log --graph --all` shows branch + >=2 commits + merge
