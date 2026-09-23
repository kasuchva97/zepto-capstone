"""Single point of truth for loading the Titanic dataset.

Called exactly once per pipeline run (from 01_eda.ipynb): loads via
seaborn (network/cache), immediately writes the offline CSV fallback, and
everything downstream (EDA + modeling) reads from that CSV or the returned
DataFrame -- never a second independent `sns.load_dataset` call.
"""
from __future__ import annotations

import pathlib

import pandas as pd

CSV_FALLBACK_PATH = pathlib.Path(__file__).resolve().parents[1] / "titanic.csv"


def load_titanic_and_cache(csv_path: pathlib.Path = CSV_FALLBACK_PATH) -> pd.DataFrame:
    """Loads the raw Titanic dataset via seaborn and writes the offline CSV fallback.

    Needs internet the first time (seaborn fetches + caches the CSV locally);
    later calls on the same machine reuse seaborn's own cache. If the network
    call fails outright (no internet at all, not even for seaborn's own
    first-time fetch) and the committed `titanic.csv` already exists, falls
    back to reading that instead of crashing -- this is the exact scenario
    the assignment's own offline-fallback requirement describes.
    """
    import seaborn as sns

    try:
        df = sns.load_dataset("titanic")
        df.to_csv(csv_path, index=False)
        return df
    except Exception:
        if csv_path.exists():
            return pd.read_csv(csv_path)
        raise


def load_titanic_from_csv(csv_path: pathlib.Path = CSV_FALLBACK_PATH) -> pd.DataFrame:
    """Loads the raw Titanic dataset from the committed offline CSV fallback."""
    return pd.read_csv(csv_path)
