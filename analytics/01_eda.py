# %% [markdown]
# # Module 2, Part A — Titanic: Profiling, Cleaning & the Data Story
#
# This notebook loads the Titanic dataset **exactly once** (via
# `sns.load_dataset`, network/cache), profiles it, cleans it using an
# explicit missing-value threshold rule, and builds a visual "data story"
# about who was more likely to survive and why. `02_modeling.ipynb`
# continues from the `titanic.csv` this notebook writes — it never calls
# `sns.load_dataset` again.

# %%
import sys

sys.path.insert(0, "..") if ".." not in sys.path else None
sys.path.insert(0, ".")

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from src.cleaning import clean_titanic, missing_value_report
from src.data import load_titanic_and_cache
from src.eda_stats import (
    central_tendency_summary,
    correlation_matrix,
    iqr_outlier_count,
    survival_rate_by,
    top_correlated_pairs,
    zscore_standardize,
)

sns.set_theme(style="whitegrid")
CHARTS_DIR = "charts"
import os

os.makedirs(CHARTS_DIR, exist_ok=True)

# %% [markdown]
# ## 1. Load once, profile, and save the offline CSV fallback
#
# The very next line is the **only** `sns.load_dataset('titanic')` call in
# the whole `/analytics` module. It immediately writes `titanic.csv` so this
# still works even if `sns.load_dataset` can't reach the network on a later
# run (`pd.read_csv("titanic.csv")` reproduces the same raw frame).

# %%
raw_df = load_titanic_and_cache()
print("Shape:", raw_df.shape)
raw_df.info()

# %%
raw_df.describe(include="all").T

# %% [markdown]
# ### Missing values per column

# %%
missing_pct = missing_value_report(raw_df)
missing_pct.to_frame("pct_missing")

# %% [markdown]
# **Observed missing-value percentages** (891 rows):
# - `deck`: ~77% missing
# - `age`: ~20% missing
# - `embarked`: ~0.22% missing (2 rows)
# - `embark_town`: ~0.22% missing (same 2 rows)
#
# All other columns are complete.

# %% [markdown]
# ## 2. Missing-value handling (threshold rule)
#
# | Column | % missing | Bracket | Decision |
# |---|---|---|---|
# | `deck` | ~77% | very high (imputation unreliable) | **Drop the column.** Imputing over three-quarters of a column would mostly fabricate data, and `deck` is largely a proxy for `pclass`/`fare` (cabin class) that those columns already capture. |
# | `age` | ~20% | 5%–30% | **Impute** — median age within each `(pclass, sex)` group (closer than a single global median, since age differs noticeably across class and sex here). |
# | `embarked` | ~0.22% | < 5% | **Drop the 2 rows.** Too small a fraction to justify imputing, and dropping loses negligible data. |
# | `embark_town` | ~0.22% | < 5% | Same 2 rows dropped (it's the long-form name of `embarked`); the column itself is **kept** for readable chart labels later. |
#
# `alive` is also dropped here — not because of missing values (it has
# none), but because it's a string restatement of the `survived` target
# itself (`"yes"`/`"no"` for the same value), i.e. direct label leakage with
# no analytical value beyond what `survived` already provides.
#
# The full rule (and the modeling-only feature subset used in Part B) is
# implemented once in `src/cleaning.py` / `src/preprocessing.py` and reused
# by both notebooks — not re-derived by hand here.

# %%
df = clean_titanic(raw_df)
print("Shape after cleaning:", df.shape)
print("Any missing values left?", df.isna().sum().sum())
df.head()

# %% [markdown]
# ## 3. Univariate analysis: `age` and `fare`

# %%
fig, axes = plt.subplots(2, 2, figsize=(12, 8))
sns.histplot(df["age"], bins=30, kde=True, ax=axes[0, 0]).set_title("Age — histogram")
sns.boxplot(x=df["age"], ax=axes[0, 1]).set_title("Age — box plot")
sns.histplot(df["fare"], bins=30, kde=True, ax=axes[1, 0]).set_title("Fare — histogram")
sns.boxplot(x=df["fare"], ax=axes[1, 1]).set_title("Fare — box plot")
plt.tight_layout()
plt.savefig(f"{CHARTS_DIR}/01_univariate_age_fare.png", dpi=120)
plt.show()

# %%
age_outliers = iqr_outlier_count(df["age"])
fare_outliers = iqr_outlier_count(df["fare"])
print(f"IQR outliers in age:  {age_outliers}")
print(f"IQR outliers in fare: {fare_outliers}")

# %%
fare_stats = central_tendency_summary(df["fare"])
fare_stats

# %% [markdown]
# **Interpretation.** `age` has a small number of IQR outliers (mostly older
# passengers in their 60s–80s) and is roughly symmetric/mildly right-skewed
# around a mean in the high 20s. `fare` has a much larger count of IQR
# outliers, all on the high side (a handful of first-class fares paid well
# over £200 against a typical fare under £15). Its mean is well above its
# median, which is in turn above its mode (most common fare bucket is near
# the cheapest fares) — **mean > median > mode**, the textbook signature of a
# **right-skewed** distribution: a long tail of expensive tickets pulls the
# mean up while most passengers paid comparatively little.

# %% [markdown]
# ## 4. Bivariate analysis: survival rate by sex, pclass, and both

# %%
by_sex = survival_rate_by(df, "sex")
by_pclass = survival_rate_by(df, "pclass")
by_sex_pclass = survival_rate_by(df, "sex", "pclass")
print("Survival rate by sex:\n", by_sex, "\n")
print("Survival rate by pclass:\n", by_pclass, "\n")
print("Survival rate by sex & pclass:\n", by_sex_pclass)

# %%
corr = correlation_matrix(df)
fig, ax = plt.subplots(figsize=(6, 5))
sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", center=0, ax=ax)
ax.set_title("Correlation matrix (survived, pclass, age, sibsp, parch, fare)")
plt.tight_layout()
plt.savefig(f"{CHARTS_DIR}/02_correlation_heatmap.png", dpi=120)
plt.show()

# %%
top_pairs = top_correlated_pairs(corr, n=2)
for a, b, val in top_pairs:
    print(f"{a} <-> {b}: {val:.3f}")

# %% [markdown]
# **Interpretation.** Women survived at a far higher rate than men in every
# class (74.0% vs. 18.9% overall), and 1st class passengers survived at a
# far higher rate than 3rd class within both sexes — the `sex`+`pclass`
# breakdown shows these effects stack: 1st-class women survived at 96.7%,
# while 3rd-class men survived at just 13.5%. In the correlation matrix, the
# two strongest absolute correlations turn out to be **`pclass` ↔ `fare`**
# (-0.548 — a numerically higher class label, i.e. a cheaper/lower cabin
# class, goes with a lower fare, as expected) and **`sibsp` ↔ `parch`**
# (+0.415 — passengers traveling with more siblings/spouses also tend to be
# traveling with more parents/children, i.e. they're on board as part of a
# family group rather than these being independent variables). `survived`
# itself correlates most with `pclass` (-0.336) and `fare` (+0.255) — real,
# but neither is in the top two overall once `pclass`↔`fare` and
# `sibsp`↔`parch` (both purely-explanatory, non-target relationships) are
# included in the ranking.

# %% [markdown]
# ## 5. Multivariate data story: who survived, and why?
#
# Four charts building one coherent argument: sex and class together were
# the dominant survival factors, age mattered at the margins (children first),
# and family size had a "sweet spot" — travelling entirely alone or in a
# very large group both hurt your odds relative to a small family.

# %%
fig, ax = plt.subplots(figsize=(7, 5))
sns.barplot(data=df, x="pclass", y="survived", hue="sex", ax=ax, errorbar=None)
ax.set_title("Survival rate by passenger class and sex")
ax.set_ylabel("Survival rate")
plt.tight_layout()
plt.savefig(f"{CHARTS_DIR}/03_survival_by_class_sex.png", dpi=120)
plt.show()

# %% [markdown]
# **Chart 1 — Survival rate by class and sex.** Class and sex compound
# rather than substitute for each other: 1st-class women survived at close
# to 95%+, while 3rd-class men survived at well under 20%. Being a woman
# helped in every class, and being in 1st class helped within both sexes —
# together they explain most of the spread in outcomes across the ship.

# %%
fig, ax = plt.subplots(figsize=(7, 5))
sns.boxplot(data=df, x="pclass", y="age", hue="survived", ax=ax)
ax.set_title("Age distribution by class, split by survival")
plt.tight_layout()
plt.savefig(f"{CHARTS_DIR}/04_age_by_class_survival.png", dpi=120)
plt.show()

# %% [markdown]
# **Chart 2 — Age by class, split by survival.** Within every class,
# survivors skew slightly younger than non-survivors, most visibly in 1st
# and 2nd class — consistent with a "children and some women first"
# boarding norm for lifeboats. The effect is smaller in 3rd class, where
# passengers had less direct access to lifeboats regardless of age.

# %%
fig, ax = plt.subplots(figsize=(7, 5))
sns.scatterplot(data=df, x="age", y="fare", hue="survived", style="pclass", alpha=0.7, ax=ax)
ax.set_title("Fare vs. age, colored by survival, styled by class")
plt.tight_layout()
plt.savefig(f"{CHARTS_DIR}/05_fare_vs_age_survival.png", dpi=120)
plt.show()

# %% [markdown]
# **Chart 3 — Fare vs. age, colored by survival.** Survivors (orange) cluster
# more densely at higher fares than non-survivors (blue), and this pattern
# holds across the age range rather than being driven by a handful of young
# passengers. It visually reinforces that fare — a proxy for wealth and
# cabin class/location — mattered more to survival than age did on its own.

# %%
family_size = df["sibsp"] + df["parch"] + 1
family_bucket = pd.cut(
    family_size, bins=[0, 1, 2, 4, 20], labels=["Alone (1)", "Small (2)", "Medium (3-4)", "Large (5+)"]
)
family_df = df.assign(family_bucket=family_bucket)
fig, ax = plt.subplots(figsize=(7, 5))
sns.barplot(data=family_df, x="family_bucket", y="survived", ax=ax, errorbar=None, order=[
    "Alone (1)", "Small (2)", "Medium (3-4)", "Large (5+)"
])
ax.set_title("Survival rate by family size on board")
ax.set_ylabel("Survival rate")
plt.tight_layout()
plt.savefig(f"{CHARTS_DIR}/06_survival_by_family_size.png", dpi=120)
plt.show()

# %% [markdown]
# **Chart 4 — Survival rate by family size.** Traveling completely alone was
# the worst outcome, small families (2–4 people, i.e. couples or a parent
# with 1–3 children) did noticeably better, and very large families (5+) did
# worst of all — likely a mix of being harder to keep together while
# boarding lifeboats and overlapping heavily with large, poorer 3rd-class
# families. The "sweet spot" is a small family, not solo travel or a large
# group.

# %% [markdown]
# ## 6. EDA sanity check: z-score standardization of `age` and `fare`
#
# This is an exploratory check only — it does **not** feed into the
# modeling pipeline in `02_modeling.ipynb`, which fits its own
# `StandardScaler` on the training split only.

# %%
before = df[["age", "fare"]].agg(["mean", "std"])
standardized = zscore_standardize(df, ["age", "fare"])
after = standardized[["age", "fare"]].agg(["mean", "std"])
print("Before standardization:\n", before)
print("\nAfter standardization:\n", after)

# %%
fig, axes = plt.subplots(1, 2, figsize=(11, 4))
sns.histplot(standardized["age"], kde=True, ax=axes[0]).set_title("Standardized age (z-score)")
sns.histplot(standardized["fare"], kde=True, ax=axes[1]).set_title("Standardized fare (z-score)")
plt.tight_layout()
plt.savefig(f"{CHARTS_DIR}/07_zscore_check.png", dpi=120)
plt.show()

# %% [markdown]
# Both standardized columns now have mean ≈ 0 and standard deviation ≈ 1,
# confirming the z-score transform behaved as expected. Shape is unchanged
# (standardization doesn't fix skew) — `fare`'s standardized histogram is
# still visibly right-skewed, just rescaled.

# %% [markdown]
# ## Summary
#
# `age` and `fare` were cleaned/imputed above; `deck` and `alive` were
# dropped; everything else was kept. This cleaned frame (`df`) is what the
# rest of this notebook's charts use. `titanic.csv`, written in step 1, is
# the **raw** (pre-cleaning) frame — `02_modeling.ipynb` reads it and
# re-applies the exact same `clean_titanic` function from `src/cleaning.py`,
# so cleaning logic lives in one place, not duplicated by hand across
# notebooks, while the raw dataset is still only ever loaded from the
# network/cache once.
