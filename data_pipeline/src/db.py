"""Builds the normalized SQLite schema and loads cleaned book data into it."""
from __future__ import annotations

import sqlite3

import pandas as pd

SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS categories (
    category_id   INTEGER PRIMARY KEY AUTOINCREMENT,
    category_name TEXT UNIQUE NOT NULL
);

CREATE TABLE IF NOT EXISTS books (
    book_id     INTEGER PRIMARY KEY AUTOINCREMENT,
    title       TEXT NOT NULL,
    price_gbp   REAL NOT NULL,
    price_inr   REAL NOT NULL,
    rating      INTEGER NOT NULL CHECK (rating BETWEEN 1 AND 5),
    in_stock    INTEGER NOT NULL CHECK (in_stock IN (0, 1)),
    category_id INTEGER NOT NULL REFERENCES categories(category_id)
);
"""


def build_database(clean_df: pd.DataFrame, db_path: str) -> None:
    """(Re)creates the schema at db_path and inserts clean_df into it."""
    conn = sqlite3.connect(db_path)
    try:
        conn.executescript("DROP TABLE IF EXISTS books; DROP TABLE IF EXISTS categories;")
        conn.executescript(SCHEMA_SQL)

        categories = sorted(clean_df["category"].unique())
        conn.executemany(
            "INSERT INTO categories (category_name) VALUES (?)",
            [(c,) for c in categories],
        )

        cat_id = dict(conn.execute("SELECT category_name, category_id FROM categories").fetchall())

        rows = [
            (
                r.title,
                float(r.price_gbp),
                float(r.price_inr),
                int(r.rating),
                int(bool(r.in_stock)),
                cat_id[r.category],
            )
            for r in clean_df.itertuples(index=False)
        ]
        conn.executemany(
            "INSERT INTO books (title, price_gbp, price_inr, rating, in_stock, category_id) "
            "VALUES (?, ?, ?, ?, ?, ?)",
            rows,
        )
        conn.commit()
    finally:
        conn.close()


def get_connection(db_path: str) -> sqlite3.Connection:
    return sqlite3.connect(db_path)
