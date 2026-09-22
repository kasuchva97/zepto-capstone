import math

import pandas as pd
import pytest

from src.clean import FX_GBP_TO_INR, clean_dataframe, parse_in_stock, parse_price, parse_rating


class TestParsePrice:
    def test_strips_currency_symbol(self):
        assert parse_price("£47.82") == 47.82

    def test_plain_number_string(self):
        assert parse_price("19.63") == 19.63

    def test_unparseable_returns_nan(self):
        assert math.isnan(parse_price("not a price"))

    def test_non_string_returns_nan(self):
        assert math.isnan(parse_price(None))


class TestParseRating:
    @pytest.mark.parametrize(
        "word,expected",
        [("One", 1), ("Two", 2), ("Three", 3), ("Four", 4), ("Five", 5), ("three", 3)],
    )
    def test_known_words(self, word, expected):
        assert parse_rating(word) == expected

    def test_unknown_word_returns_nan(self):
        assert math.isnan(parse_rating("Zero"))


class TestParseInStock:
    def test_in_stock_variants(self):
        assert parse_in_stock("In stock (22 available)") is True
        assert parse_in_stock("  In stock  ") is True

    def test_out_of_stock(self):
        assert parse_in_stock("Out of stock") is False

    def test_unparseable_returns_none(self):
        assert parse_in_stock("Limited availability") is None
        assert parse_in_stock(None) is None


class TestCleanDataframe:
    def _raw(self, **overrides):
        base = {
            "title": "Some Book",
            "price": "£20.00",
            "star_rating": "Three",
            "availability": "In stock (5 available)",
            "category": "Fiction",
        }
        base.update(overrides)
        return base

    def test_happy_path_types_and_fx(self):
        df = pd.DataFrame([self._raw()])
        clean = clean_dataframe(df)
        row = clean.iloc[0]
        assert row["price_gbp"] == 20.0
        assert row["price_inr"] == pytest.approx(20.0 * FX_GBP_TO_INR)
        assert row["rating"] == 3
        assert pd.api.types.is_integer_dtype(type(row["rating"]))
        assert row["in_stock"] == True  # noqa: E712 (explicit bool check)
        assert pd.api.types.is_bool_dtype(clean["in_stock"])

    def test_fx_rate_is_exactly_105_50(self):
        assert FX_GBP_TO_INR == 105.50

    def test_drops_row_with_missing_title(self):
        df = pd.DataFrame([self._raw(title=""), self._raw()])
        clean = clean_dataframe(df)
        assert len(clean) == 1
        assert clean.attrs["cleaning_report"]["dropped_missing_identity"] == 1

    def test_drops_row_with_unparseable_availability(self):
        df = pd.DataFrame([self._raw(availability="???"), self._raw()])
        clean = clean_dataframe(df)
        assert len(clean) == 1
        assert clean.attrs["cleaning_report"]["dropped_unparseable_availability"] == 1

    def test_median_imputes_unparseable_price(self):
        df = pd.DataFrame(
            [self._raw(price="£10.00"), self._raw(price="£30.00"), self._raw(price="bad")]
        )
        clean = clean_dataframe(df)
        assert len(clean) == 3  # row kept, not dropped
        assert clean.attrs["cleaning_report"]["price_values_median_imputed"] == 1
        imputed_row = clean[clean["price_gbp"] == 20.0]  # median of 10, 30
        assert len(imputed_row) == 1

    def test_median_imputes_unparseable_rating(self):
        df = pd.DataFrame(
            [self._raw(star_rating="Two"), self._raw(star_rating="Four"), self._raw(star_rating="Zero")]
        )
        clean = clean_dataframe(df)
        assert len(clean) == 3
        assert clean.attrs["cleaning_report"]["rating_values_median_imputed"] == 1
        assert clean["rating"].between(1, 5).all()

    def test_output_columns_and_row_minimum_shape(self):
        rows = [self._raw(category=c) for c in ["A", "B", "C"] for _ in range(25)]
        df = pd.DataFrame(rows)
        clean = clean_dataframe(df)
        assert set(clean.columns) == {"title", "category", "price_gbp", "price_inr", "rating", "in_stock"}
        assert len(clean) >= 60
        assert clean["category"].nunique() >= 3
