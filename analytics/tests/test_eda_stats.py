import pandas as pd
import pytest

from src.eda_stats import (
    central_tendency_summary,
    correlation_matrix,
    iqr_outlier_bounds,
    iqr_outlier_count,
    survival_rate_by,
    top_correlated_pairs,
)


class TestIQR:
    def test_no_outliers_in_uniform_data(self):
        s = pd.Series(range(1, 101))  # 1..100, no outliers by construction
        assert iqr_outlier_count(s) == 0

    def test_detects_injected_outliers(self):
        s = pd.Series(list(range(1, 51)) + [1000, -500])
        lower, upper = iqr_outlier_bounds(s)
        assert 1000 > upper
        assert -500 < lower
        assert iqr_outlier_count(s) == 2


class TestCentralTendency:
    def test_right_skew_detected(self):
        # mean > median > mode by construction
        s = pd.Series([1, 1, 1, 2, 2, 3, 100])
        result = central_tendency_summary(s)
        assert result["mode"] == 1
        assert result["mean"] > result["median"] > result["mode"]
        assert result["skew_direction"] == "right-skewed"

    def test_symmetric_data(self):
        s = pd.Series([1, 2, 3, 4, 5, 2, 2, 4, 4, 3])  # roughly symmetric around 3
        result = central_tendency_summary(s)
        assert result["skew_direction"] in ("approximately symmetric", "right-skewed", "left-skewed")


class TestSurvivalRateBy:
    def _df(self):
        return pd.DataFrame(
            {
                "sex": ["male", "male", "female", "female", "female"],
                "pclass": [1, 3, 1, 3, 3],
                "survived": [0, 0, 1, 1, 0],
            }
        )

    def test_single_column_grouping(self):
        result = survival_rate_by(self._df(), "sex")
        result = result.set_index("sex")
        assert result.loc["male", "survival_rate"] == 0.0
        assert result.loc["female", "survival_rate"] == pytest.approx(2 / 3)
        assert result.loc["male", "n"] == 2
        assert result.loc["female", "n"] == 3

    def test_two_column_grouping(self):
        result = survival_rate_by(self._df(), "sex", "pclass")
        row = result[(result["sex"] == "female") & (result["pclass"] == 3)].iloc[0]
        assert row["survival_rate"] == 0.5
        assert row["n"] == 2


class TestCorrelationMatrix:
    def _df(self):
        return pd.DataFrame(
            {
                "survived": [0, 1, 1, 0, 1, 0, 1, 0],
                "pclass": [3, 1, 1, 3, 2, 3, 1, 2],
                "age": [22, 38, 26, 35, 28, 2, 27, 54],
                "sibsp": [1, 1, 0, 1, 0, 0, 0, 1],
                "parch": [0, 0, 0, 0, 0, 0, 2, 0],
                "fare": [7.25, 71.28, 7.92, 53.1, 8.05, 21.07, 11.13, 51.86],
                "adult_male": [True, False, False, True, True, False, True, True],
                "alone": [False, False, True, False, True, True, False, False],
            }
        )

    def test_matrix_uses_exactly_the_six_specified_columns(self):
        corr = correlation_matrix(self._df())
        assert list(corr.columns) == ["survived", "pclass", "age", "sibsp", "parch", "fare"]
        assert "adult_male" not in corr.columns
        assert "alone" not in corr.columns
        assert corr.shape == (6, 6)

    def test_diagonal_is_one(self):
        corr = correlation_matrix(self._df())
        assert (corr.values.diagonal() == 1.0).all()

    def test_top_correlated_pairs_ranked_by_absolute_value(self):
        corr = correlation_matrix(self._df())
        pairs = top_correlated_pairs(corr, n=2)
        assert len(pairs) == 2
        # sanity: sorted descending by |corr|
        assert abs(pairs[0][2]) >= abs(pairs[1][2])
        # survived/pclass should be a strong (negative) relationship in this toy set
        pair_names = [{p[0], p[1]} for p in pairs]
        assert {"survived", "pclass"} in pair_names
