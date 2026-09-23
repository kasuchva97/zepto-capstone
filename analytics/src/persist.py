"""Saves/reloads the complete fitted pipeline (preprocessing + estimator as
one object) so it's usable end-to-end on raw, unpreprocessed new data."""
from __future__ import annotations

import pathlib

import joblib
import pandas as pd
from sklearn.pipeline import Pipeline


def save_pipeline(pipe: Pipeline, path: pathlib.Path) -> None:
    joblib.dump(pipe, path)


def load_pipeline(path: pathlib.Path) -> Pipeline:
    return joblib.load(path)


def predict_on_raw(pipe: Pipeline, raw_rows: pd.DataFrame):
    """raw_rows must contain the pipeline's expected raw feature columns
    (pclass, sex, age, sibsp, parch, fare, embarked) -- unpreprocessed,
    possibly with missing values; the loaded pipeline handles imputation/
    encoding/scaling internally."""
    return pipe.predict(raw_rows)
