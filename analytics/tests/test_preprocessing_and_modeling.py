import numpy as np
import pandas as pd
import pytest
from sklearn.model_selection import train_test_split

from src.modeling import build_classifier_pipelines, evaluate_all, evaluate_classifier, train_all
from src.preprocessing import CATEGORICAL_FEATURES, NUMERIC_FEATURES, build_preprocessor, select_model_columns


@pytest.fixture
def model_ready_df():
    rng = np.random.default_rng(0)
    n = 150
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
            "extra_junk_column": rng.random(n),  # should be dropped by select_model_columns
        }
    )


class TestSelectModelColumns:
    def test_keeps_only_feature_and_target_columns(self, model_ready_df):
        out = select_model_columns(model_ready_df)
        assert set(out.columns) == set(NUMERIC_FEATURES + CATEGORICAL_FEATURES + ["survived"])
        assert "extra_junk_column" not in out.columns


class TestPreprocessorNoLeakage:
    def test_scaler_stats_come_only_from_training_data(self, model_ready_df):
        df = select_model_columns(model_ready_df)
        X, y = df.drop(columns=["survived"]), df["survived"]
        X_train, X_test = train_test_split(X, test_size=0.3, random_state=1)

        pre = build_preprocessor()
        pre.fit(X_train)  # fit on train only
        train_mean_from_scaler = pre.named_transformers_["num"].named_steps["scaler"].mean_

        expected_mean = X_train[NUMERIC_FEATURES].fillna(X_train[NUMERIC_FEATURES].median()).mean().values
        np.testing.assert_allclose(train_mean_from_scaler, expected_mean, rtol=1e-6)

        # transforming test data must not refit / change the learned stats
        pre.transform(X_test)
        np.testing.assert_allclose(
            pre.named_transformers_["num"].named_steps["scaler"].mean_, train_mean_from_scaler
        )

    def test_pipeline_fit_only_calls_fit_once_on_preprocessor(self, model_ready_df):
        """A Pipeline.fit(X_train) then .predict(X_test) must never re-fit the
        preprocessor on X_test -- verified by checking learned stats are
        identical before and after predicting on the test set."""
        df = select_model_columns(model_ready_df)
        X, y = df.drop(columns=["survived"]), df["survived"]
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1, stratify=y)

        pipelines = build_classifier_pipelines()
        pipe = pipelines["Logistic Regression"]
        pipe.fit(X_train, y_train)
        mean_after_fit = pipe.named_steps["preprocessor"].named_transformers_["num"].named_steps["scaler"].mean_.copy()

        pipe.predict(X_test)
        mean_after_predict = pipe.named_steps["preprocessor"].named_transformers_["num"].named_steps["scaler"].mean_

        np.testing.assert_array_equal(mean_after_fit, mean_after_predict)


class TestModelingEndToEnd:
    def test_all_three_classifiers_train_and_evaluate(self, model_ready_df):
        df = select_model_columns(model_ready_df)
        X, y = df.drop(columns=["survived"]), df["survived"]
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1, stratify=y)

        pipelines = build_classifier_pipelines()
        assert set(pipelines) == {"Logistic Regression", "Decision Tree", "Random Forest"}
        train_all(pipelines, X_train, y_train)
        results = evaluate_all(pipelines, X_test, y_test)

        assert list(results.index) == ["Logistic Regression", "Decision Tree", "Random Forest"]
        for col in ["accuracy", "precision", "recall", "f1", "roc_auc"]:
            assert col in results.columns
            assert results[col].between(0, 1).all()

    def test_confusion_matrix_shape_and_sum_equals_test_size(self, model_ready_df):
        df = select_model_columns(model_ready_df)
        X, y = df.drop(columns=["survived"]), df["survived"]
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1, stratify=y)

        pipe = build_classifier_pipelines()["Random Forest"]
        pipe.fit(X_train, y_train)
        metrics = evaluate_classifier(pipe, X_test, y_test)
        cm = metrics["confusion_matrix"]
        assert cm.shape == (2, 2)
        assert cm.sum() == len(y_test)
