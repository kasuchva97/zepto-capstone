"""Regression side-task: predict `fare` from the other available features
with a multivariate linear regression, on the same cleaned dataset."""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

TARGET_COLUMN = "fare"
NUMERIC_FEATURES = ["pclass", "age", "sibsp", "parch", "survived"]
CATEGORICAL_FEATURES = ["sex", "embarked"]
REGRESSION_FEATURE_COLUMNS = NUMERIC_FEATURES + CATEGORICAL_FEATURES


def select_regression_columns(df: pd.DataFrame) -> pd.DataFrame:
    cols = [c for c in REGRESSION_FEATURE_COLUMNS + [TARGET_COLUMN] if c in df.columns]
    return df[cols].copy()


def build_regression_preprocessor() -> ColumnTransformer:
    numeric_pipeline = Pipeline(
        steps=[("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler())]
    )
    categorical_pipeline = Pipeline(
        steps=[("imputer", SimpleImputer(strategy="most_frequent")), ("encoder", OneHotEncoder(handle_unknown="ignore"))]
    )
    return ColumnTransformer(
        transformers=[
            ("num", numeric_pipeline, NUMERIC_FEATURES),
            ("cat", categorical_pipeline, CATEGORICAL_FEATURES),
        ]
    )


def build_regression_pipeline() -> Pipeline:
    return Pipeline(
        steps=[("preprocessor", build_regression_preprocessor()), ("regressor", LinearRegression())]
    )


def adjusted_r2(r2: float, n_samples: int, n_predictors: int) -> float:
    """n_predictors = number of columns in the actual (post-encoding) design
    matrix -- the number of parameters the model estimates -- not the raw
    pre-encoding feature count."""
    denom = n_samples - n_predictors - 1
    if denom <= 0:
        return float("nan")
    return 1 - (1 - r2) * (n_samples - 1) / denom


def evaluate_regression(pipe: Pipeline, X_test: pd.DataFrame, y_test: pd.Series) -> dict:
    y_pred = pipe.predict(X_test)
    residuals = y_test.values - y_pred
    r2 = r2_score(y_test, y_pred)
    n_predictors = pipe.named_steps["preprocessor"].transform(X_test).shape[1]
    return {
        "mae": mean_absolute_error(y_test, y_pred),
        "rmse": float(np.sqrt(mean_squared_error(y_test, y_pred))),
        "r2": r2,
        "adjusted_r2": adjusted_r2(r2, len(y_test), n_predictors),
        "y_pred": y_pred,
        "residuals": residuals,
    }
