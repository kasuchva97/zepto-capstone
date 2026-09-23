"""Missing-value profiling and threshold-based cleaning for the Titanic dataset.

Threshold rule applied per column (per assignment spec):
  < 5% missing   -> drop the affected rows
  5% - 30%       -> impute
  very high (>50%, unreliable to impute) -> drop the column, or encode
                    "missing" as its own category -- decided and justified
                    per column below.

Measured on the raw seaborn load (891 rows):
  deck          77.22% missing -> DROP COLUMN. Imputing ~77% of a column
                from the remaining ~23% would be mostly fabricated data, and
                deck is already a rough proxy for pclass/fare (cabin class),
                so little unique signal is lost.
  age           19.87% missing -> IMPUTE (falls in the 5-30% bracket). Uses
                median age within each (pclass, sex) group rather than a
                single global median, since age distributions differ
                noticeably across class and sex on this dataset -- a
                closer, still-simple imputation.
  embarked       0.22% missing (2 rows) -> DROP ROWS (< 5%).
  embark_town    0.22% missing (same 2 rows) -> DROP ROWS (< 5%). Kept as a
                column (it's just the long-form name of `embarked`) since
                it's still useful for readable chart labels in the EDA story.

This module only drops `alive`, a string restatement of the `survived`
target ("yes"/"no" for the same 0/1 value) -- direct label leakage with no
analytical value beyond what `survived` already gives, so it's removed here
rather than kept around as a temptation to accidentally use as a feature.
`class`, `who`, `embark_town`, `adult_male` and `alone` are all kept in the
cleaned EDA DataFrame (useful for the multivariate data story's charts);
the *modeling* feature subset is a separate, later decision made in
`src/preprocessing.py` (Task 8), not here.
"""
from __future__ import annotations

import pandas as pd

DROP_COLUMN_THRESHOLD = 0.30  # above this, imputation is treated as unreliable
IMPUTE_LOWER_THRESHOLD = 0.05

TARGET_LEAKAGE_COLUMNS = ["alive"]


def missing_value_report(df: pd.DataFrame) -> pd.Series:
    """Returns % missing per column, for columns with any missing values, descending."""
    pct = df.isna().mean() * 100
    return pct[pct > 0].sort_values(ascending=False)


def clean_titanic(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    report = missing_value_report(df)

    for col, pct in report.items():
        frac = pct / 100
        if frac >= DROP_COLUMN_THRESHOLD:
            df = df.drop(columns=[col])
        elif frac < IMPUTE_LOWER_THRESHOLD:
            df = df[df[col].notna()]
        else:
            if col == "age":
                df["age"] = df.groupby(["pclass", "sex"])["age"].transform(lambda s: s.fillna(s.median()))
                df["age"] = df["age"].fillna(df["age"].median())  # safety net for any empty group
            else:
                # generic 5-30% fallback: median for numeric, mode for categorical
                if pd.api.types.is_numeric_dtype(df[col]):
                    df[col] = df[col].fillna(df[col].median())
                else:
                    df[col] = df[col].fillna(df[col].mode().iloc[0])

    existing_leakage = [c for c in TARGET_LEAKAGE_COLUMNS if c in df.columns]
    df = df.drop(columns=existing_leakage)

    return df.reset_index(drop=True)
