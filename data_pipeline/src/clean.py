"""Cleans raw scraped book rows into properly typed columns.

Row-level failure policy (documented in README):
  - `title` / `category` missing or blank -> row dropped (they are the row's
    identity; there is nothing sensible to impute).
  - `price_gbp` unparseable -> left NaN, then median-imputed (numeric field).
  - `rating` unparseable (unexpected star-rating word) -> left NaN, then
    median-imputed and rounded back to the nearest int 1-5 (ordinal-numeric field).
  - `in_stock` unparseable (availability text matches neither "in stock" nor
    "out of stock") -> row dropped. It is boolean, not numeric, so median
    imputation does not apply, and guessing a stock status would fabricate data.
"""
from __future__ import annotations

import re

import pandas as pd

FX_GBP_TO_INR = 105.50  # fixed, project-defined baseline rate (not a live/dated market rate)

RATING_WORDS = {"one": 1, "two": 2, "three": 3, "four": 4, "five": 5}


def parse_price(raw: str) -> float:
    """'£47.82' -> 47.82. Returns NaN if no number can be found."""
    if not isinstance(raw, str):
        return float("nan")
    match = re.search(r"[\d.]+", raw)
    return float(match.group()) if match else float("nan")


def parse_rating(raw: str) -> float:
    """'Four' -> 4.0. Returns NaN for unrecognised words."""
    if not isinstance(raw, str):
        return float("nan")
    return float(RATING_WORDS.get(raw.strip().lower(), float("nan")))


def parse_in_stock(raw: str):
    """'In stock (22 available)' -> True, 'Out of stock' -> False, else None (unparseable)."""
    if not isinstance(raw, str):
        return None
    text = raw.strip().lower()
    if "out of stock" in text:
        return False
    if "in stock" in text:
        return True
    return None


def clean_dataframe(raw_df: pd.DataFrame) -> pd.DataFrame:
    df = raw_df.copy()

    df["title"] = df["title"].astype(str).str.strip()
    df["category"] = df["category"].astype(str).str.strip()
    before = len(df)
    df = df[(df["title"] != "") & (df["title"] != "nan") & (df["category"] != "") & (df["category"] != "nan")]
    dropped_identity = before - len(df)

    df["price_gbp"] = df["price"].apply(parse_price)
    n_price_imputed = df["price_gbp"].isna().sum()
    if n_price_imputed:
        df["price_gbp"] = df["price_gbp"].fillna(df["price_gbp"].median())

    df["rating"] = df["star_rating"].apply(parse_rating)
    n_rating_imputed = df["rating"].isna().sum()
    if n_rating_imputed:
        df["rating"] = df["rating"].fillna(df["rating"].median())
    df["rating"] = df["rating"].round().astype(int)

    df["in_stock"] = df["availability"].apply(parse_in_stock)
    before_stock = len(df)
    dropped_stock = int(df["in_stock"].isna().sum())
    df = df[df["in_stock"].notna()]
    df["in_stock"] = df["in_stock"].astype(bool)

    df["price_inr"] = (df["price_gbp"] * FX_GBP_TO_INR).round(2)

    df = df[["title", "category", "price_gbp", "price_inr", "rating", "in_stock"]].reset_index(drop=True)

    df.attrs["cleaning_report"] = {
        "rows_in": before,
        "rows_out": len(df),
        "dropped_missing_identity": int(dropped_identity),
        "dropped_unparseable_availability": dropped_stock,
        "price_values_median_imputed": int(n_price_imputed),
        "rating_values_median_imputed": int(n_rating_imputed),
        "fx_rate_gbp_to_inr": FX_GBP_TO_INR,
    }
    return df
