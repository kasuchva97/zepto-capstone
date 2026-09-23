"""GridSearchCV tuning of the Random Forest's n_estimators/max_depth/max_features,
with an OOB score reported for the best refit estimator.

oob_score_ is only populated when oob_score=True was passed at construction
time, so the estimator inside the grid is built that way (Task 12 note).
"""
from __future__ import annotations

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline

from src.preprocessing import build_preprocessor

RANDOM_STATE = 42

PARAM_GRID = {
    "classifier__n_estimators": [100, 200, 300],
    "classifier__max_depth": [3, 5, 10, None],
    "classifier__max_features": ["sqrt", "log2"],
}


def tune_random_forest(X_train: pd.DataFrame, y_train: pd.Series, cv: int = 5) -> GridSearchCV:
    pipe = Pipeline(
        steps=[
            ("preprocessor", build_preprocessor()),
            (
                "classifier",
                RandomForestClassifier(oob_score=True, bootstrap=True, random_state=RANDOM_STATE),
            ),
        ]
    )
    search = GridSearchCV(pipe, PARAM_GRID, cv=cv, scoring="f1", n_jobs=-1)
    search.fit(X_train, y_train)
    return search


def oob_score_of_best_estimator(search: GridSearchCV) -> float:
    return search.best_estimator_.named_steps["classifier"].oob_score_
