"""Train-only-fit preprocessing for the classification pipeline.

Modeling feature set (documented decision): pclass, sex, age, sibsp,
parch, fare, embarked. `class`, `who`, `adult_male`, `alone`, `embark_town`
are dropped here -- all derived from / redundant with the columns already
kept (class~pclass, who/adult_male~age+sex, alone~sibsp+parch,
embark_town~embarked) -- to avoid feeding the same signal into a model
twice under different names.
"""
from __future__ import annotations

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

TARGET_COLUMN = "survived"
NUMERIC_FEATURES = ["pclass", "age", "sibsp", "parch", "fare"]
CATEGORICAL_FEATURES = ["sex", "embarked"]
MODEL_FEATURE_COLUMNS = NUMERIC_FEATURES + CATEGORICAL_FEATURES


def select_model_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Raw (uncleaned is fine) Titanic frame -> just the modeling feature columns + target."""
    cols = [c for c in MODEL_FEATURE_COLUMNS + [TARGET_COLUMN] if c in df.columns]
    return df[cols].copy()


def build_preprocessor() -> ColumnTransformer:
    """A ColumnTransformer that imputes + scales numerics and imputes + one-hot
    encodes categoricals. Fit only on the training split; called in
    transform-only mode on the test split (enforced structurally by being
    the first step of a scikit-learn Pipeline, never invoked directly)."""
    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(handle_unknown="ignore")),
        ]
    )
    return ColumnTransformer(
        transformers=[
            ("num", numeric_pipeline, NUMERIC_FEATURES),
            ("cat", categorical_pipeline, CATEGORICAL_FEATURES),
        ]
    )


def get_feature_names(preprocessor: ColumnTransformer) -> list[str]:
    """Human-readable feature names after the ColumnTransformer's fit (numeric
    names as-is, one-hot categorical names as `col_value`)."""
    return list(preprocessor.get_feature_names_out())
