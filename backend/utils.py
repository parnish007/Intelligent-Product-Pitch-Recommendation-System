"""
utils.py

Stateless utility helpers for data validation and light preprocessing.
Safe for reuse in training & inference.
"""

from typing import List, Optional, Tuple
import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer


# Missing Value Handling


def handle_missing_values(
    df: pd.DataFrame,
    numeric_cols: Optional[List[str]] = None,
    categorical_cols: Optional[List[str]] = None
) -> pd.DataFrame:
    """
    Fill missing values:
    - numeric -> median
    - categorical -> most frequent
    """
    df = df.copy()

    if numeric_cols is None:
        numeric_cols = df.select_dtypes(include=["int64", "float64"]).columns.tolist()
    if categorical_cols is None:
        categorical_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()

    if numeric_cols:
        num_imputer = SimpleImputer(strategy="median")
        df[numeric_cols] = num_imputer.fit_transform(df[numeric_cols])

    if categorical_cols:
        cat_imputer = SimpleImputer(strategy="most_frequent")
        df[categorical_cols] = cat_imputer.fit_transform(df[categorical_cols])

    return df



# Outlier Handling (EDA only)


def handle_outliers_iqr(
    df: pd.DataFrame,
    numeric_cols: Optional[List[str]] = None,
    return_summary: bool = False
):
    """
    IQR-based outlier capping.
    NOTE: Should NOT be used before train-test split.
    """
    df = df.copy()

    if numeric_cols is None:
        numeric_cols = df.select_dtypes(include=["int64", "float64"]).columns.tolist()

    summary = []

    for col in numeric_cols:
        if df[col].nunique() <= 2:
            continue

        q1, q3 = np.percentile(df[col].dropna(), [25, 75])
        iqr = q3 - q1
        lower, upper = q1 - 1.5 * iqr, q3 + 1.5 * iqr

        outliers = ((df[col] < lower) | (df[col] > upper)).sum()

        df[col] = df[col].clip(lower, upper)

        summary.append({
            "column": col,
            "outliers": int(outliers),
            "lower": float(lower),
            "upper": float(upper)
        })

    if return_summary:
        return df, pd.DataFrame(summary)

    return df


#
# Simple Helpers


YES_NO_MAP = {
    "yes": 1, "no": 0, "y": 1, "n": 0,
    "1": 1, "0": 0, 1: 1, 0: 0
}

def convert_yes_no_columns(df: pd.DataFrame, cols: List[str]) -> pd.DataFrame:
    """
    Convert yes/no style columns into binary.
    Unknown values default to 0.
    """
    df = df.copy()
    for col in cols:
        if col in df.columns:
            df[col] = (
                df[col]
                .astype(str)
                .str.strip()
                .str.lower()
                .map(YES_NO_MAP)
                .fillna(0)
                .astype(int)
            )
    return df


def validate_required_columns(
    df: pd.DataFrame,
    required_cols: List[str]
) -> Tuple[bool, List[str]]:
    missing = [c for c in required_cols if c not in df.columns]
    return len(missing) == 0, missing
