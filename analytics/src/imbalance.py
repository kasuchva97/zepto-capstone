"""Three-way class-imbalance comparison: baseline, class_weight='balanced',
and SMOTE (oversampling the training fold only, via an imblearn Pipeline so
resampling is structurally excluded from the test-time transform path).

Uses Logistic Regression as the single model for this comparison, as the
assignment allows ("any one of the three models is enough for this sub-task").
"""
from __future__ import annotations

import pandas as pd
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, precision_score, recall_score
from sklearn.pipeline import Pipeline

from src.preprocessing import build_preprocessor

RANDOM_STATE = 42


def class_balance(y: pd.Series) -> pd.Series:
    return y.value_counts(normalize=True).rename("proportion")


def compare_imbalance_strategies(X_train, y_train, X_test, y_test) -> pd.DataFrame:
    variants = {
        "baseline (no handling)": Pipeline(
            steps=[
                ("preprocessor", build_preprocessor()),
                ("classifier", LogisticRegression(max_iter=1000, random_state=RANDOM_STATE)),
            ]
        ),
        "class_weight='balanced'": Pipeline(
            steps=[
                ("preprocessor", build_preprocessor()),
                (
                    "classifier",
                    LogisticRegression(max_iter=1000, random_state=RANDOM_STATE, class_weight="balanced"),
                ),
            ]
        ),
        "SMOTE (train fold only)": ImbPipeline(
            steps=[
                ("preprocessor", build_preprocessor()),
                ("smote", SMOTE(random_state=RANDOM_STATE)),
                ("classifier", LogisticRegression(max_iter=1000, random_state=RANDOM_STATE)),
            ]
        ),
    }

    rows = []
    for name, pipe in variants.items():
        pipe.fit(X_train, y_train)
        y_pred = pipe.predict(X_test)
        rows.append(
            {
                "strategy": name,
                "precision": precision_score(y_test, y_pred, zero_division=0),
                "recall": recall_score(y_test, y_pred),
                "f1": f1_score(y_test, y_pred),
            }
        )
    return pd.DataFrame(rows).set_index("strategy")
