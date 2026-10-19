import pandas as pd

from src.eda import dataset_overview, iqr_outlier_flags, profile_columns


def test_dataset_overview_counts_rows_and_duplicates():
    df = pd.DataFrame({"x": [1, 2, 2], "category": ["a", "b", "b"]})
    result = dataset_overview(df)
    assert result["rows"] == 3
    assert result["columns"] == 2
    assert result["duplicate_rows"] == 1


def test_profile_columns_detects_missing_values():
    df = pd.DataFrame({"x": [1, None, 3]})
    profile = profile_columns(df)
    assert profile.loc[0, "missing_count"] == 1
    assert profile.loc[0, "missing_percent"] == 33.33


def test_iqr_outlier_flags_flags_extreme_value():
    df = pd.DataFrame({"x": [1, 2, 2, 3, 4, 100]})
    flags = iqr_outlier_flags(df)
    assert not flags.empty
    assert "x" in flags["column"].tolist()
