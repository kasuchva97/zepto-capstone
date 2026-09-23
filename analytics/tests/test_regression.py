import numpy as np
import pandas as pd
import pytest
from sklearn.model_selection import train_test_split

from src.regression import adjusted_r2, build_regression_pipeline, evaluate_regression, select_regression_columns


@pytest.fixture
def regression_df():
    rng = np.random.default_rng(3)
    n = 150
    pclass = rng.choice([1, 2, 3], n)
    fare = np.maximum(5, 100 - pclass * 25 + rng.normal(0, 10, n))  # fare correlates with pclass
    return pd.DataFrame(
        {
            "pclass": pclass,
            "sex": rng.choice(["male", "female"], n),
            "age": rng.normal(30, 10, n),
            "sibsp": rng.integers(0, 3, n),
            "parch": rng.integers(0, 3, n),
            "survived": rng.integers(0, 2, n),
            "embarked": rng.choice(["S", "C", "Q"], n),
            "fare": fare,
        }
    )


class TestAdjustedR2:
    def test_adjusted_r2_lower_than_r2_when_predictors_present(self):
        r2 = 0.5
        adj = adjusted_r2(r2, n_samples=100, n_predictors=5)
        assert adj < r2

    def test_returns_nan_when_denominator_non_positive(self):
        result = adjusted_r2(0.5, n_samples=5, n_predictors=10)
        assert np.isnan(result)


class TestRegressionPipeline:
    def test_select_regression_columns_excludes_target_leakage_columns(self, regression_df):
        df = regression_df.copy()
        df["alive"] = "no"
        out = select_regression_columns(df)
        assert "fare" in out.columns
        assert "alive" not in out.columns

    def test_pipeline_fits_and_reports_all_required_metrics(self, regression_df):
        df = select_regression_columns(regression_df)
        X, y = df.drop(columns=["fare"]), df["fare"]
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1)

        pipe = build_regression_pipeline()
        pipe.fit(X_train, y_train)
        metrics = evaluate_regression(pipe, X_test, y_test)

        for key in ["mae", "rmse", "r2", "adjusted_r2"]:
            assert key in metrics
            assert np.isfinite(metrics[key])
        assert metrics["mae"] >= 0
        assert metrics["rmse"] >= 0

    def test_recovers_strong_known_relationship(self, regression_df):
        """fare was constructed to depend strongly (negatively) on pclass;
        the fitted model should achieve a reasonably high R2 on this
        synthetic, low-noise relationship."""
        df = select_regression_columns(regression_df)
        X, y = df.drop(columns=["fare"]), df["fare"]
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1)

        pipe = build_regression_pipeline()
        pipe.fit(X_train, y_train)
        metrics = evaluate_regression(pipe, X_test, y_test)
        assert metrics["r2"] > 0.5

    def test_residuals_length_matches_test_set(self, regression_df):
        df = select_regression_columns(regression_df)
        X, y = df.drop(columns=["fare"]), df["fare"]
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1)

        pipe = build_regression_pipeline()
        pipe.fit(X_train, y_train)
        metrics = evaluate_regression(pipe, X_test, y_test)
        assert len(metrics["residuals"]) == len(y_test)
