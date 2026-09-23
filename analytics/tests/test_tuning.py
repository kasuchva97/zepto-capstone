import numpy as np
import pandas as pd
import pytest
from sklearn.model_selection import train_test_split

from src.preprocessing import select_model_columns
from src.tuning import PARAM_GRID, oob_score_of_best_estimator, tune_random_forest


@pytest.fixture
def model_ready_df():
    rng = np.random.default_rng(11)
    n = 120
    return pd.DataFrame(
        {
            "pclass": rng.choice([1, 2, 3], n),
            "sex": rng.choice(["male", "female"], n),
            "age": rng.normal(30, 10, n),
            "sibsp": rng.integers(0, 3, n),
            "parch": rng.integers(0, 3, n),
            "fare": rng.gamma(2, 15, n),
            "embarked": rng.choice(["S", "C", "Q"], n),
            "survived": rng.integers(0, 2, n),
        }
    )


class TestTuneRandomForest:
    def test_best_params_are_from_the_declared_grid(self, model_ready_df):
        df = select_model_columns(model_ready_df)
        X, y = df.drop(columns=["survived"]), df["survived"]
        X_train, _, y_train, _ = train_test_split(X, y, test_size=0.3, random_state=1, stratify=y)

        search = tune_random_forest(X_train, y_train, cv=3)

        assert search.best_params_["classifier__n_estimators"] in PARAM_GRID["classifier__n_estimators"]
        assert search.best_params_["classifier__max_depth"] in PARAM_GRID["classifier__max_depth"]
        assert search.best_params_["classifier__max_features"] in PARAM_GRID["classifier__max_features"]

    def test_oob_score_is_available_and_in_valid_range(self, model_ready_df):
        df = select_model_columns(model_ready_df)
        X, y = df.drop(columns=["survived"]), df["survived"]
        X_train, _, y_train, _ = train_test_split(X, y, test_size=0.3, random_state=1, stratify=y)

        search = tune_random_forest(X_train, y_train, cv=3)
        oob = oob_score_of_best_estimator(search)

        assert 0.0 <= oob <= 1.0

    def test_best_estimator_was_constructed_with_oob_score_true(self, model_ready_df):
        df = select_model_columns(model_ready_df)
        X, y = df.drop(columns=["survived"]), df["survived"]
        X_train, _, y_train, _ = train_test_split(X, y, test_size=0.3, random_state=1, stratify=y)

        search = tune_random_forest(X_train, y_train, cv=3)
        classifier = search.best_estimator_.named_steps["classifier"]
        assert classifier.oob_score is True
        assert hasattr(classifier, "oob_score_")  # only populated when oob_score=True was set at construction
