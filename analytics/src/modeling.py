"""Builds/trains/evaluates the three required classifiers on top of the
shared preprocessing ColumnTransformer, all as full scikit-learn Pipelines
so preprocessing is structurally fit-on-train / transform-on-test only."""
from __future__ import annotations

import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

from src.preprocessing import build_preprocessor

RANDOM_STATE = 42


def build_classifier_pipelines() -> dict[str, Pipeline]:
    return {
        "Logistic Regression": Pipeline(
            steps=[
                ("preprocessor", build_preprocessor()),
                ("classifier", LogisticRegression(max_iter=1000, random_state=RANDOM_STATE)),
            ]
        ),
        "Decision Tree": Pipeline(
            steps=[
                ("preprocessor", build_preprocessor()),
                ("classifier", DecisionTreeClassifier(max_depth=5, random_state=RANDOM_STATE)),
            ]
        ),
        "Random Forest": Pipeline(
            steps=[
                ("preprocessor", build_preprocessor()),
                ("classifier", RandomForestClassifier(n_estimators=200, random_state=RANDOM_STATE)),
            ]
        ),
    }


def train_all(pipelines: dict[str, Pipeline], X_train: pd.DataFrame, y_train: pd.Series) -> dict[str, Pipeline]:
    for pipe in pipelines.values():
        pipe.fit(X_train, y_train)
    return pipelines


def evaluate_classifier(pipe: Pipeline, X_test: pd.DataFrame, y_test: pd.Series) -> dict:
    y_pred = pipe.predict(X_test)
    y_proba = pipe.predict_proba(X_test)[:, 1]
    fpr, tpr, _ = roc_curve(y_test, y_proba)
    return {
        "confusion_matrix": confusion_matrix(y_test, y_pred),
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred, zero_division=0),
        "recall": recall_score(y_test, y_pred),
        "f1": f1_score(y_test, y_pred),
        "roc_auc": roc_auc_score(y_test, y_proba),
        "roc_curve": (fpr, tpr),
    }


def evaluate_all(pipelines: dict[str, Pipeline], X_test: pd.DataFrame, y_test: pd.Series) -> pd.DataFrame:
    rows = []
    for name, pipe in pipelines.items():
        m = evaluate_classifier(pipe, X_test, y_test)
        rows.append(
            {
                "model": name,
                "accuracy": m["accuracy"],
                "precision": m["precision"],
                "recall": m["recall"],
                "f1": m["f1"],
                "roc_auc": m["roc_auc"],
            }
        )
    return pd.DataFrame(rows).set_index("model")
