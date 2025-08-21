"""Data ingestion utilities: load csv, validate keys, and split."""
import pandas as pd
from loguru import logger
from sklearn.model_selection import train_test_split
import os

def load_table(path: str) -> pd.DataFrame:
    if not os.path.exists(path):
        logger.error(f"Table not found: {path}")
        raise FileNotFoundError(path)
    df = pd.read_csv(path)
    logger.info(f"Loaded table: {path} rows={len(df)} cols={len(df.columns)}")
    return df

def validate_primary_key(df, key: str) -> bool:
    if key not in df.columns:
        logger.error(f"Primary key '{key}' not found in DataFrame columns: {list(df.columns)}")
        raise KeyError(key)
    dups = df[key].duplicated()
    nulls = df[key].isna()
    if dups.any() or nulls.any():
        logger.warning(
            f"Primary key '{key}' invalid: duplicates={int(dups.sum())}, nulls={int(nulls.sum())}"
        )
        return False
    return True

def split_df(df, test_size=0.15, val_size=0.15, randomize=True):
    if randomize:
        train_val, test = train_test_split(df, test_size=test_size, random_state=42)
        train, val = train_test_split(train_val, test_size=val_size/(1-test_size), random_state=42)
    else:
        n = len(df)
        n_test = int(n * test_size)
        n_val = int(n * val_size)
        train = df[:-n_test-n_val]
        val = df[-n_test-n_val:-n_test] if n_val>0 else df[:-n_test]
        test = df[-n_test:]
    logger.debug(f"Split sizes -> train={len(train)} val={len(val)} test={len(test)}")
    return train.reset_index(drop=True), val.reset_index(drop=True), test.reset_index(drop=True)
