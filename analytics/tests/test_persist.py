import numpy as np
import pandas as pd
import pytest
from sklearn.model_selection import train_test_split

from src.modeling import build_classifier_pipelines
from src.persist import load_pipeline, predict_on_raw, save_pipeline
from src.preprocessing import select_model_columns


@pytest.fixture
def trained_pipeline_and_raw_data():
    rng = np.random.default_rng(5)
    n = 120
    df = pd.DataFrame(
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
    # introduce raw missing values, like real unpreprocessed new data would have
    df.loc[0:5, "age"] = np.nan
    df.loc[0:2, "embarked"] = np.nan

    model_df = select_model_columns(df)
    X, y = model_df.drop(columns=["survived"]), model_df["survived"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1, stratify=y)

    pipe = build_classifier_pipelines()["Random Forest"]
    pipe.fit(X_train, y_train)
    return pipe, X_test


class TestPersist:
    def test_round_trip_predictions_match_original(self, tmp_path, trained_pipeline_and_raw_data):
        pipe, X_test = trained_pipeline_and_raw_data
        path = tmp_path / "pipeline.joblib"
        save_pipeline(pipe, path)

        assert path.exists()
        reloaded = load_pipeline(path)

        original_preds = pipe.predict(X_test)
        reloaded_preds = predict_on_raw(reloaded, X_test)
        np.testing.assert_array_equal(original_preds, reloaded_preds)

    def test_reloaded_pipeline_handles_raw_missing_values_end_to_end(self, tmp_path, trained_pipeline_and_raw_data):
        """The saved artifact must be the FULL pipeline (imputer+encoder+
        scaler+estimator), not the bare estimator -- proven by successfully
        predicting on raw rows containing NaNs without any manual
        preprocessing step in this test."""
        pipe, X_test = trained_pipeline_and_raw_data
        path = tmp_path / "pipeline.joblib"
        save_pipeline(pipe, path)
        reloaded = load_pipeline(path)

        raw_row = X_test.iloc[[0]].copy()
        raw_row["age"] = np.nan
        raw_row["embarked"] = np.nan

        preds = predict_on_raw(reloaded, raw_row)
        assert preds.shape == (1,)
        assert preds[0] in (0, 1)
