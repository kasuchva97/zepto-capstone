# Module 2 — Analytics

One cohesive pipeline over the Titanic dataset: load once, profile it, clean it with
an explicit threshold rule, tell a visual "data story" about survival, then build and
rigorously evaluate a full classification + regression modeling pipeline on the same
cleaned data.

## Structure

```
analytics/
├── 01_eda.ipynb / .py          # Part A: load, profile, clean, EDA, data story
├── 02_modeling.ipynb / .py     # Part B: classification + regression pipeline
├── titanic.csv                  # committed offline fallback (raw, written by 01_eda)
├── src/                          # shared, tested logic used by both notebooks
├── tests/                        # pytest suite (43 tests)
├── charts/                       # saved PNGs (supporting artifacts, not a substitute for the written interpretations below)
└── models/best_titanic_classifier.joblib   # full fitted pipeline (preprocessing + estimator)
```

The `.py` files are [jupytext](https://jupytext.readthedocs.io/) percent-format
sources for the two notebooks (easier to review as a diff); the `.ipynb` files are
the actual executed deliverables with baked-in outputs. Regenerate either with:
```bash
jupytext --to notebook 01_eda.py && jupyter nbconvert --to notebook --execute --inplace 01_eda.ipynb
jupytext --to notebook 02_modeling.py && jupyter nbconvert --to notebook --execute --inplace 02_modeling.ipynb
```

## Setup

```bash
cd analytics
py -3.11 -m venv .venv        # Windows; `python3.11 -m venv .venv` on macOS/Linux
.venv\Scripts\activate        # Windows; `source .venv/bin/activate` on macOS/Linux
pip install -r requirements.txt
```

**Requires Python 3.11** (what this was built and tested against). Newer Pythons
(3.13+) may lack prebuilt wheels for some dependencies here (pandas, scikit-learn,
etc.) and can hang trying to build them from source — check `python --version` first,
and if it isn't 3.11.x, install Python 3.11 from
[python.org](https://www.python.org/downloads/) alongside your existing version
rather than swapping `py -3.11` for plain `python` above.

## Run end to end

```bash
jupyter nbconvert --to notebook --execute --inplace 01_eda.ipynb
jupyter nbconvert --to notebook --execute --inplace 02_modeling.ipynb
```
`01_eda.ipynb` must run first — it writes `titanic.csv`, which `02_modeling.ipynb`
reads. Both notebooks are already included pre-executed with full output, so
re-running them isn't necessary just to read the results, but they're fully
reproducible from scratch (only
`01_eda.ipynb`'s first cell needs internet, for the one `sns.load_dataset` call).

## Run tests

```bash
python -m pytest -q
```
43 tests across cleaning, EDA statistics, preprocessing/leakage checks, all three
classifiers, the imbalance comparison, GridSearchCV tuning, the regression side-task,
joblib persistence, and integration tests against the actual committed `titanic.csv`
and saved model artifact (not just synthetic fixtures).

## Design decisions & written interpretations

### Part A — profiling, cleaning, EDA

**Missing values** (891 raw rows): `deck` 77.22%, `age` 19.87%, `embarked` 0.22% (2
rows), `embark_town` 0.22% (same 2 rows). Threshold rule applied:
- `deck` (77%, "very high") → **dropped entirely**. Imputing over three-quarters of a
  column would mostly fabricate data, and it's largely a cabin-class proxy that
  `pclass`/`fare` already capture.
- `age` (19.87%, in the 5–30% bracket) → **imputed**, using the median age within
  each `(pclass, sex)` group rather than one global median, since age differs
  noticeably across class and sex in this data.
- `embarked` / `embark_town` (0.22%, < 5%) → **rows dropped** (2 rows). `embark_town`
  is kept as a column (it's just `embarked`'s long-form name) for readable chart
  labels.
- `alive` (0% missing, but pure target leakage — a string restatement of `survived`)
  → **dropped**, separately from the threshold rule, since it has no missing values
  but no analytical value either.

See `src/cleaning.py` for the implementation and full rationale.

**Univariate (`age`, `fare`).** IQR outliers: **32** for `age`, **114** for `fare`.
`fare`: mean 32.10, median 14.45, mode 8.05 — **mean > median > mode**, the signature
of a **right-skewed** distribution (a long tail of expensive 1st-class fares pulls the
mean well above the typical passenger's fare).

**Bivariate.** Survival rate by sex: female 74.0%, male 18.9%. By class: 1st 62.6%,
2nd 47.3%, 3rd 24.2%. By sex+class: highest is 1st-class women (96.7%), lowest is
3rd-class men (13.5%) — the two effects compound rather than substitute for each
other. Correlation matrix (`survived, pclass, age, sibsp, parch, fare`; `adult_male`/
`alone` excluded as derived flags): the two strongest absolute correlations are
**`pclass` ↔ `fare`** (−0.548) and **`sibsp` ↔ `parch`** (+0.415) — `survived`'s own
strongest ties (`pclass` −0.336, `fare` +0.255) are real but not quite the top two
once those two purely-explanatory relationships are included in the ranking.

**Multivariate data story** (4 charts, each with its own written interpretation in
the notebook): class+sex compound (chart 1), survivors skew younger within each class
consistent with a "children/women first" boarding norm (chart 2), higher fares
associate with survival across the whole age range (chart 3), and small families
(2–4 people) survived better than either solo travelers or large families (chart 4).

**Z-score sanity check.** `age`/`fare` standardized to mean ≈ 0, std ≈ 1 (verified
numerically and visually) — exploratory only, not fed into the modeling pipeline.

### Part B — modeling

**Stratified split.** `survived` is imbalanced (~62/38). A stratified 80/20 split
keeps that ratio in both folds (train: 61.7/38.3, test: 61.8/38.2), so metrics —
especially recall/F1 on the minority "survived" class — reflect real performance
rather than an unlucky random split on a dataset of only ~890 rows.

**Preprocessing.** A single `ColumnTransformer` (median-impute + `StandardScaler` for
`pclass, age, sibsp, parch, fare`; most-frequent-impute + `OneHotEncoder` for `sex,
embarked`) wrapped in each model's `Pipeline`, so `.fit(X_train)` fits it once on
the training fold and every later `.predict(X_test)` runs it transform-only — no code
path fits or refits on the test set or the full pre-split data (verified in
`tests/test_preprocessing_and_modeling.py`).

**Feature set.** `pclass, age, sibsp, parch, fare, sex, embarked`. `class`, `who`,
`adult_male`, `alone`, `embark_town` are excluded from modeling — each duplicates
signal already in a kept column (`class`~`pclass`, `who`/`adult_male`~`age`+`sex`,
`alone`~`sibsp`+`parch`, `embark_town`~`embarked`).

**Classifier comparison** (test set, 178 rows):

| Model | Accuracy | Precision | Recall | F1 | ROC-AUC |
|---|---|---|---|---|---|
| Logistic Regression | 0.826 | 0.814 | 0.706 | **0.756** | 0.860 |
| Decision Tree (depth 5) | 0.775 | 0.733 | 0.647 | 0.688 | 0.834 |
| Random Forest (200 trees) | 0.792 | 0.746 | 0.691 | 0.718 | 0.836 |
| Random Forest (tuned, GridSearchCV) | 0.809 | 0.766 | 0.721 | 0.742 | 0.836 |

**Imbalance comparison** (Logistic Regression; class balance ~62% not-survived / 38%
survived): baseline F1 0.756, `class_weight='balanced'` F1 0.739, SMOTE (train-fold
only) F1 0.735. **Conclusion:** the imbalance here is mild, not severe, so the
baseline (no handling) actually keeps the best F1 in this case — both rebalancing
strategies trade meaningful precision for a small recall gain, which isn't a good
trade when the classes aren't very imbalanced to begin with. `class_weight='balanced'`
and SMOTE would likely earn their keep on a more skewed dataset; here they mostly add
noise.

**Tuning.** `GridSearchCV` over `n_estimators∈{100,200,300}`, `max_depth∈{3,5,10,None}`,
`max_features∈{sqrt,log2}` (5-fold CV, scoring F1). Best params: `n_estimators=300,
max_depth=None, max_features='sqrt'`; best CV F1 0.747; **OOB score of the refit best
estimator: 0.816**. The tuned Random Forest (F1 0.742) still doesn't beat untuned
Logistic Regression (F1 0.756) on the held-out test set — a sign this dataset's
signal, with only 7 features, is close to fully captured by a simple linear decision
boundary, and the ensemble's extra complexity isn't buying much here.

**Regression side-task** (`fare` from `pclass, age, sibsp, parch, survived, sex,
embarked`): MAE 21.12, RMSE 41.68, R² 0.349, Adjusted R² 0.310. **Residual plot shows
heteroscedasticity**: the residual spread visibly widens (a funnel/cone shape) as
predicted fare increases — tight near zero for cheap tickets, fanning out much wider
for expensive ones — so the model's error variance isn't constant across the range of
predictions.

**Final recommendation.** Deploy **Logistic Regression**: it has the best F1 (0.756)
and accuracy (0.826) of the four classifier variants tested, is the simplest and most
interpretable, and the more complex Random Forest — even after `GridSearchCV` tuning
— doesn't beat it on this feature set. The regression side-task's R² (0.349) is far
below any classifier's accuracy, but the two aren't comparable: predicting a
continuous, heavy-tailed `fare` from a handful of passenger attributes is a
fundamentally harder, noisier problem than binary survival classification, which is
why classification and regression metrics are reported as two separate blocks above
rather than one merged scale.

**Persistence.** The best-by-F1 pipeline (Logistic Regression, full fitted
`ColumnTransformer` + estimator as one object) is saved via
`joblib.dump(...,"models/best_titanic_classifier.joblib")`, reloaded, and demonstrated
predicting on raw new passenger rows — including ones with missing `age`/`embarked` —
with no manual preprocessing step, confirming the saved artifact really is the
complete pipeline and not just the bare estimator.
