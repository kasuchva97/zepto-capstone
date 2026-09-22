"""The required SQL queries against the books/categories schema, plus the
pandas read_sql / merge equivalence demonstration."""
from __future__ import annotations

import sqlite3

import pandas as pd

QUERIES: dict[str, str] = {
    # SELECT + WHERE
    "affordable_in_stock_books": """
        SELECT title, price_gbp, price_inr, category_id
        FROM books
        WHERE price_gbp < 20 AND in_stock = 1;
    """,
    # ORDER BY + LIMIT
    "top_10_most_expensive_books": """
        SELECT title, price_inr
        FROM books
        ORDER BY price_inr DESC
        LIMIT 10;
    """,
    # DISTINCT
    "distinct_categories": """
        SELECT DISTINCT category_name
        FROM categories
        ORDER BY category_name;
    """,
    # IN
    "high_rated_books": """
        SELECT title, rating
        FROM books
        WHERE rating IN (4, 5)
        ORDER BY rating DESC, title;
    """,
    # BETWEEN
    "midrange_priced_books": """
        SELECT title, price_gbp
        FROM books
        WHERE price_gbp BETWEEN 20 AND 40
        ORDER BY price_gbp;
    """,
    # JOIN + window function (ROW_NUMBER) + ORDER BY -- true top-N-per-group query
    "top_5_rated_books_per_category": """
        WITH ranked AS (
            SELECT
                c.category_name,
                b.title,
                b.rating,
                b.price_inr,
                ROW_NUMBER() OVER (
                    PARTITION BY c.category_name
                    ORDER BY b.rating DESC, b.price_inr ASC
                ) AS rnk
            FROM books b
            JOIN categories c ON b.category_id = c.category_id
        )
        SELECT category_name, title, rating, price_inr
        FROM ranked
        WHERE rnk <= 5
        ORDER BY category_name, rnk;
    """,
}


def run_all_queries(conn: sqlite3.Connection) -> dict[str, pd.DataFrame]:
    return {name: pd.read_sql(sql, conn) for name, sql in QUERIES.items()}


def merge_equivalent_top_rated_per_category(books_df: pd.DataFrame, categories_df: pd.DataFrame) -> pd.DataFrame:
    """Reproduces the 'top_5_rated_books_per_category' JOIN + window-function
    query using pd.merge + groupby ranking on in-memory DataFrames only, no
    SQL, for the equivalence check. kind='mergesort' keeps the sort stable so
    ties break in the same row order SQLite's ROW_NUMBER uses (insertion/rowid
    order), matching the SQL result exactly rather than just in aggregate."""
    merged = books_df.merge(categories_df, on="category_id", how="inner")
    merged = merged.sort_values(
        ["category_name", "rating", "price_inr"], ascending=[True, False, True], kind="mergesort"
    )
    merged["rnk"] = merged.groupby("category_name").cumcount() + 1
    top5 = merged[merged["rnk"] <= 5]
    return top5[["category_name", "title", "rating", "price_inr"]].reset_index(drop=True)


if __name__ == "__main__":
    import pathlib

    db_path = pathlib.Path(__file__).resolve().parents[1] / "data" / "zepto_books.db"
    conn = sqlite3.connect(db_path)
    results = run_all_queries(conn)
    for name, df in results.items():
        print(f"\n=== {name} ({len(df)} rows) ===")
        print(df.head(10).to_string(index=False))
