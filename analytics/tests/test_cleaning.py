import pandas as pd

from src.cleaning import clean_titanic, missing_value_report


class TestMissingValueReport:
    def test_reports_only_columns_with_missing_values_descending(self, synthetic_raw_titanic):
        report = missing_value_report(synthetic_raw_titanic)
        assert list(report.index) == ["deck", "age", "embarked", "embark_town"]
        assert report.is_monotonic_decreasing

    def test_percentages_roughly_match_expected_brackets(self, synthetic_raw_titanic):
        report = missing_value_report(synthetic_raw_titanic)
        assert report["deck"] > 30  # very high -> drop column
        assert 5 <= report["age"] <= 30  # impute bracket
        assert report["embarked"] < 5  # drop rows bracket


class TestCleanTitanic:
    def test_deck_column_dropped_high_missing(self, synthetic_raw_titanic):
        clean = clean_titanic(synthetic_raw_titanic)
        assert "deck" not in clean.columns

    def test_age_has_no_missing_after_cleaning(self, synthetic_raw_titanic):
        clean = clean_titanic(synthetic_raw_titanic)
        assert clean["age"].isna().sum() == 0

    def test_age_imputation_does_not_touch_non_missing_values(self, synthetic_raw_titanic):
        clean = clean_titanic(synthetic_raw_titanic)
        original_present = synthetic_raw_titanic[synthetic_raw_titanic["age"].notna()][["_row_id", "age"]]
        merged = original_present.merge(clean[["_row_id", "age"]], on="_row_id", suffixes=("_orig", "_clean"))
        # every row that survives cleaning AND already had an age keeps that exact value
        assert len(merged) > 0
        pd.testing.assert_series_equal(
            merged["age_orig"], merged["age_clean"], check_names=False
        )

    def test_low_missing_rows_dropped_not_imputed(self, synthetic_raw_titanic):
        clean = clean_titanic(synthetic_raw_titanic)
        assert clean["embarked"].isna().sum() == 0
        assert len(clean) == len(synthetic_raw_titanic) - 2

    def test_embark_town_column_kept_for_charting(self, synthetic_raw_titanic):
        clean = clean_titanic(synthetic_raw_titanic)
        assert "embark_town" in clean.columns
        assert clean["embark_town"].isna().sum() == 0

    def test_alive_dropped_as_target_leakage(self, synthetic_raw_titanic):
        clean = clean_titanic(synthetic_raw_titanic)
        assert "alive" not in clean.columns

    def test_other_story_columns_kept(self, synthetic_raw_titanic):
        clean = clean_titanic(synthetic_raw_titanic)
        for col in ["class", "who", "adult_male", "alone", "pclass", "sex", "fare", "sibsp", "parch"]:
            assert col in clean.columns

    def test_no_missing_values_anywhere_after_cleaning(self, synthetic_raw_titanic):
        clean = clean_titanic(synthetic_raw_titanic)
        assert clean.isna().sum().sum() == 0
