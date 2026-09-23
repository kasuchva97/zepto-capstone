import pandas as pd
import pytest

import src.data as data_module
from src.data import load_titanic_and_cache, load_titanic_from_csv


class TestLoadTitanicAndCache:
    def test_writes_csv_and_returns_dataframe_on_success(self, tmp_path, monkeypatch):
        csv_path = tmp_path / "titanic.csv"
        fake_df = pd.DataFrame({"survived": [0, 1], "pclass": [3, 1]})

        class _FakeSeaborn:
            @staticmethod
            def load_dataset(name):
                assert name == "titanic"
                return fake_df

        monkeypatch.setitem(__import__("sys").modules, "seaborn", _FakeSeaborn())

        result = load_titanic_and_cache(csv_path)
        pd.testing.assert_frame_equal(result, fake_df)
        assert csv_path.exists()
        pd.testing.assert_frame_equal(pd.read_csv(csv_path), fake_df)

    def test_falls_back_to_committed_csv_when_network_call_fails(self, tmp_path, monkeypatch):
        """The real-world scenario this guards against: a grader with no
        internet at all runs 01_eda.ipynb; sns.load_dataset raises, but the
        committed titanic.csv is right there, so this must not crash."""
        csv_path = tmp_path / "titanic.csv"
        pd.DataFrame({"survived": [1, 0, 1], "pclass": [1, 2, 3]}).to_csv(csv_path, index=False)

        class _FailingSeaborn:
            @staticmethod
            def load_dataset(name):
                raise ConnectionError("simulated: no internet, seaborn cache also empty")

        monkeypatch.setitem(__import__("sys").modules, "seaborn", _FailingSeaborn())

        result = load_titanic_and_cache(csv_path)
        assert list(result["survived"]) == [1, 0, 1]

    def test_raises_when_network_fails_and_no_csv_fallback_exists(self, tmp_path, monkeypatch):
        csv_path = tmp_path / "titanic.csv"  # deliberately does not exist

        class _FailingSeaborn:
            @staticmethod
            def load_dataset(name):
                raise ConnectionError("simulated: no internet, no fallback")

        monkeypatch.setitem(__import__("sys").modules, "seaborn", _FailingSeaborn())

        with pytest.raises(ConnectionError):
            load_titanic_and_cache(csv_path)


class TestLoadTitanicFromCsv:
    def test_reads_back_what_was_written(self, tmp_path):
        csv_path = tmp_path / "titanic.csv"
        original = pd.DataFrame({"survived": [0, 1], "age": [22.0, None]})
        original.to_csv(csv_path, index=False)

        result = load_titanic_from_csv(csv_path)
        pd.testing.assert_frame_equal(result, original)
