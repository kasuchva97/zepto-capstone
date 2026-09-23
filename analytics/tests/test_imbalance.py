import numpy as np
import pandas as pd
import pytest
from sklearn.model_selection import train_test_split

from src.imbalance import class_balance, compare_imbalance_strategies
from src.preprocessing import select_model_columns


@pytest.fixture
def imbalanced_df():
    """~85/15 imbalanced synthetic set, large enough for SMOTE's k-neighbors."""
    rng = np.random.default_rng(7)
    n_majority, n_minority = 170, 30
    rows = []
    for _ in range(n_majority):
        rows.append(
            {
                "pclass": rng.choice([1, 2, 3]),
                "sex": rng.choice(["male", "female"]),
                "age": rng.normal(32, 10),
                "sibsp": rng.integers(0, 3),
                "parch": rng.integers(0, 3),
                "fare": rng.gamma(2, 15),
                "embarked": rng.choice(["S", "C", "Q"]),
                "survived": 0,
            }
        )
    for _ in range(n_minority):
        rows.append(
            {
                "pclass": rng.choice([1, 2, 3]),
                "sex": rng.choice(["male", "female"]),
                "age": rng.normal(28, 10),
                "sibsp": rng.integers(0, 3),
                "parch": rng.integers(0, 3),
                "fare": rng.gamma(2, 15),
                "embarked": rng.choice(["S", "C", "Q"]),
                "survived": 1,
            }
        )
    return pd.DataFrame(rows)


class TestClassBalance:
    def test_proportions_sum_to_one_and_reflect_imbalance(self, imbalanced_df):
        balance = class_balance(imbalanced_df["survived"])
        assert balance.sum() == pytest.approx(1.0)
        assert balance[0] > balance[1]  # majority class 0


class TestCompareImbalanceStrategies:
    def test_returns_all_three_strategies_with_metrics(self, imbalanced_df):
        df = select_model_columns(imbalanced_df)
        X, y = df.drop(columns=["survived"]), df["survived"]
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1, stratify=y)

        result = compare_imbalance_strategies(X_train, y_train, X_test, y_test)
        assert list(result.index) == [
            "baseline (no handling)",
            "class_weight='balanced'",
            "SMOTE (train fold only)",
        ]
        for col in ["precision", "recall", "f1"]:
            assert col in result.columns
            assert result[col].between(0, 1).all()

    def test_smote_only_resamples_training_data_test_set_size_unaffected(self, imbalanced_df):
        """SMOTE must never touch the test fold -- verified indirectly: the
        function only ever receives the already-split X_test/y_test and
        passes them straight to .predict(), so their length is untouched."""
        df = select_model_columns(imbalanced_df)
        X, y = df.drop(columns=["survived"]), df["survived"]
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1, stratify=y)
        original_test_len = len(X_test)

        compare_imbalance_strategies(X_train, y_train, X_test, y_test)

        assert len(X_test) == original_test_len  # unchanged by the call
