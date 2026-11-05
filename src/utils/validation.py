from collections.abc import Iterable

import pandas as pd


def require_columns(df: pd.DataFrame, columns: Iterable[str], name: str) -> None:
    missing = sorted(set(columns) - set(df.columns))
    if missing:
        raise ValueError(f"{name} is missing required columns: {missing}")


def assert_no_null_keys(df: pd.DataFrame, keys: Iterable[str], name: str) -> None:
    key_list = list(keys)
    null_counts = df[key_list].isna().sum()
    bad = null_counts[null_counts > 0]
    if not bad.empty:
        raise ValueError(f"{name} has null key values: {bad.to_dict()}")


def assert_unique_keys(df: pd.DataFrame, keys: Iterable[str], name: str) -> None:
    key_list = list(keys)
    duplicate_count = int(df.duplicated(key_list).sum())
    if duplicate_count:
        raise ValueError(f"{name} has {duplicate_count} duplicated key rows for {key_list}")
