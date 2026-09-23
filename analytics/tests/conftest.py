import numpy as np
import pandas as pd
import pytest


@pytest.fixture
def synthetic_raw_titanic() -> pd.DataFrame:
    """A small synthetic dataset shaped like seaborn's raw titanic load:
    same columns/dtypes, and similar missing-value proportions (deck very
    high, age moderate, embarked/embark_town a couple of rows) -- big enough
    (200 rows) that percentage-based thresholds behave the same way they do
    on the real 891-row dataset."""
    rng = np.random.default_rng(42)
    n = 200

    pclass = rng.choice([1, 2, 3], size=n, p=[0.25, 0.25, 0.5])
    sex = rng.choice(["male", "female"], size=n, p=[0.65, 0.35])
    survived = (rng.random(n) < np.where(sex == "female", 0.7, 0.2)).astype(int)
    age = rng.normal(30, 12, size=n).clip(0.5, 80)
    sibsp = rng.integers(0, 4, size=n)
    parch = rng.integers(0, 3, size=n)
    fare = rng.gamma(2, 20, size=n)
    embarked = rng.choice(["S", "C", "Q"], size=n)
    embark_town = np.select(
        [embarked == "S", embarked == "C", embarked == "Q"],
        ["Southampton", "Cherbourg", "Queenstown"],
    )
    who = np.where(age < 16, "child", np.where(sex == "male", "man", "woman"))
    adult_male = (sex == "male") & (age >= 16)
    alone = (sibsp == 0) & (parch == 0)
    pclass_label = pd.Categorical(pclass).rename_categories({1: "First", 2: "Second", 3: "Third"})
    alive = np.where(survived == 1, "yes", "no")
    deck = rng.choice(list("ABCDEFG"), size=n)

    df = pd.DataFrame(
        {
            "_row_id": np.arange(n),  # test-only helper to re-align rows after cleaning drops some
            "survived": survived,
            "pclass": pclass,
            "sex": sex,
            "age": age,
            "sibsp": sibsp,
            "parch": parch,
            "fare": fare,
            "embarked": embarked,
            "class": pclass_label,
            "who": who,
            "adult_male": adult_male,
            "deck": deck,
            "embark_town": embark_town,
            "alive": alive,
            "alone": alone,
        }
    )

    # ~19% of age missing (same bracket as the real dataset: 5-30% -> impute)
    age_missing_idx = rng.choice(n, size=int(n * 0.19), replace=False)
    df.loc[age_missing_idx, "age"] = np.nan

    # ~77% of deck missing (same bracket as real: >30% -> drop column)
    deck_missing_idx = rng.choice(n, size=int(n * 0.77), replace=False)
    df.loc[deck_missing_idx, "deck"] = np.nan

    # 2 rows missing embarked/embark_town (< 5% -> drop rows)
    df.loc[[0, 1], "embarked"] = np.nan
    df.loc[[0, 1], "embark_town"] = np.nan

    return df
