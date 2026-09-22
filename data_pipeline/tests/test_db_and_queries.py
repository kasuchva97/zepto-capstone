import sqlite3

import pandas as pd
import pytest

from src.db import build_database
from src.queries import QUERIES, merge_equivalent_top_rated_per_category, run_all_queries


@pytest.fixture
def clean_df():
    # 8 books per category (> 5) so the top-5-per-category query's LIMIT
    # behavior is actually exercised, not vacuously true.
    rows = []
    for cat, prices, ratings in [
        ("Fiction", [10.0, 25.0, 45.0, 12.0, 18.0, 33.0, 41.0, 9.0], [5, 3, 1, 5, 4, 2, 3, 5]),
        ("Mystery", [15.0, 35.0, 55.0, 20.0, 28.0, 44.0, 8.0, 60.0], [4, 4, 2, 5, 3, 1, 5, 2]),
        ("Classics", [5.0, 22.0, 60.0, 14.0, 31.0, 50.0, 19.0, 27.0], [5, 3, 5, 4, 2, 1, 5, 3]),
    ]:
        for i, (p, r) in enumerate(zip(prices, ratings)):
            rows.append(
                {
                    "title": f"{cat} Book {i}",
                    "category": cat,
                    "price_gbp": p,
                    "price_inr": round(p * 105.50, 2),
                    "rating": r,
                    "in_stock": i % 2 == 0,
                }
            )
    return pd.DataFrame(rows)


@pytest.fixture
def db_conn(tmp_path, clean_df):
    db_path = tmp_path / "test.db"
    build_database(clean_df, str(db_path))
    conn = sqlite3.connect(str(db_path))
    yield conn
    conn.close()


class TestSchema:
    def test_two_tables_with_fk(self, db_conn):
        tables = {r[0] for r in db_conn.execute("SELECT name FROM sqlite_master WHERE type='table'")}
        assert {"books", "categories"} <= tables

        fk_info = db_conn.execute("PRAGMA foreign_key_list(books)").fetchall()
        assert len(fk_info) == 1
        assert fk_info[0][2] == "categories"  # referenced table

    def test_row_counts_match_input(self, db_conn, clean_df):
        n_books = db_conn.execute("SELECT COUNT(*) FROM books").fetchone()[0]
        n_categories = db_conn.execute("SELECT COUNT(*) FROM categories").fetchone()[0]
        assert n_books == len(clean_df)
        assert n_categories == clean_df["category"].nunique()

    def test_rating_check_constraint_rejects_bad_value(self, db_conn):
        with pytest.raises(sqlite3.IntegrityError):
            db_conn.execute(
                "INSERT INTO books (title, price_gbp, price_inr, rating, in_stock, category_id) "
                "VALUES ('Bad', 1.0, 105.5, 9, 1, 1)"
            )


class TestQueries:
    def test_all_required_clauses_present_across_queries(self):
        combined = " ".join(sql.upper() for sql in QUERIES.values())
        for clause in ["WHERE", "ORDER BY", "LIMIT", "DISTINCT", "JOIN"]:
            assert clause in combined
        assert "IN (" in combined.replace(" IN(", " IN (") or "BETWEEN" in combined

    def test_queries_run_without_error_and_return_rows(self, db_conn):
        results = run_all_queries(db_conn)
        assert set(results) == set(QUERIES)
        for name, df in results.items():
            assert isinstance(df, pd.DataFrame)
            assert len(df) > 0, f"{name} returned no rows"

    def test_distinct_categories_query_matches_unique_count(self, db_conn, clean_df):
        results = run_all_queries(db_conn)
        assert len(results["distinct_categories"]) == clean_df["category"].nunique()

    def test_top_5_query_returns_exactly_5_per_category_and_is_sorted(self, db_conn):
        results = run_all_queries(db_conn)
        df = results["top_5_rated_books_per_category"]
        counts = df.groupby("category_name").size()
        assert (counts == 5).all(), "each category has 8 candidate books, so top-5 should cut off at exactly 5"
        for _, group in df.groupby("category_name"):
            ratings = group["rating"].tolist()
            assert ratings == sorted(ratings, reverse=True)

    def test_read_sql_and_merge_equivalent_for_join_query(self, db_conn):
        results = run_all_queries(db_conn)
        books_df = pd.read_sql("SELECT * FROM books;", db_conn)
        categories_df = pd.read_sql("SELECT * FROM categories;", db_conn)
        merge_result = merge_equivalent_top_rated_per_category(books_df, categories_df)
        sql_result = results["top_5_rated_books_per_category"].reset_index(drop=True)
        pd.testing.assert_frame_equal(sql_result, merge_result.reset_index(drop=True))
