"""Reusable statistics for the EDA notebook: IQR outliers, central-tendency /
skew summary, boolean-masking survival-rate breakdowns, and the correlation
matrix restricted to the six specified numeric columns."""
from __future__ import annotations

import pandas as pd

CORRELATION_COLUMNS = ["survived", "pclass", "age", "sibsp", "parch", "fare"]


def iqr_outlier_bounds(series: pd.Series) -> tuple[float, float]:
    q1, q3 = series.quantile(0.25), series.quantile(0.75)
    iqr = q3 - q1
    return q1 - 1.5 * iqr, q3 + 1.5 * iqr


def iqr_outlier_count(series: pd.Series) -> int:
    lower, upper = iqr_outlier_bounds(series)
    return int(((series < lower) | (series > upper)).sum())


def central_tendency_summary(series: pd.Series) -> dict:
    mean, median = series.mean(), series.median()
    mode = series.mode().iloc[0]
    if mean > median > mode:
        skew = "right-skewed"
    elif mean < median < mode:
        skew = "left-skewed"
    else:
        skew = "approximately symmetric"
    return {"mean": mean, "median": median, "mode": mode, "skew_direction": skew}


def survival_rate_by(df: pd.DataFrame, *group_cols: str) -> pd.DataFrame:
    """Survival rate via boolean masking, grouped by 1+ columns."""
    if len(group_cols) == 1:
        col = group_cols[0]
        rows = []
        for value in sorted(df[col].dropna().unique(), key=str):
            mask = df[col] == value
            rows.append({col: value, "survival_rate": df.loc[mask, "survived"].mean(), "n": int(mask.sum())})
        return pd.DataFrame(rows)

    col_a, col_b = group_cols
    rows = []
    for va in sorted(df[col_a].dropna().unique(), key=str):
        for vb in sorted(df[col_b].dropna().unique(), key=str):
            mask = (df[col_a] == va) & (df[col_b] == vb)
            if mask.sum() == 0:
                continue
            rows.append(
                {
                    col_a: va,
                    col_b: vb,
                    "survival_rate": df.loc[mask, "survived"].mean(),
                    "n": int(mask.sum()),
                }
            )
    return pd.DataFrame(rows)


def correlation_matrix(df: pd.DataFrame) -> pd.DataFrame:
    return df[CORRELATION_COLUMNS].corr()


def zscore_standardize(df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    """EDA-stage-only sanity check (Task 6): z = (x - mean) / std over the
    full cleaned DataFrame. Not used by the modeling pipeline, which does its
    own train-only StandardScaler fit inside the ColumnTransformer."""
    out = df.copy()
    for col in columns:
        out[col] = (df[col] - df[col].mean()) / df[col].std()
    return out


def top_correlated_pairs(corr: pd.DataFrame, n: int = 2) -> list[tuple[str, str, float]]:
    pairs = []
    cols = corr.columns.tolist()
    for i in range(len(cols)):
        for j in range(i + 1, len(cols)):
            pairs.append((cols[i], cols[j], corr.iloc[i, j]))
    pairs.sort(key=lambda t: abs(t[2]), reverse=True)
    return pairs[:n]
