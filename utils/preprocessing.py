
"""Utilities for preparing uploaded datasets for machine learning."""

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder


def prepare_train_test_data(
    df,
    target_column,
    test_size=0.2,
    random_state=42
):
    """Encode and complete a dataset, then return its train/test split."""

    X = df.drop(columns=[target_column]).copy()
    y = df[target_column].copy()

    if X.empty:
        raise ValueError(
            "The dataset must contain at least one feature column."
        )

    # Encode categorical feature columns
    categorical_columns = X.select_dtypes(
        include=["object", "category", "string"]
    ).columns

    for column in categorical_columns:
        X[column] = X[column].fillna("Missing")
        X[column] = LabelEncoder().fit_transform(
            X[column].astype(str)
        )

    # Fill missing numeric values
    numeric_columns = X.select_dtypes(
        include="number"
    ).columns

    if len(numeric_columns) > 0:
        X[numeric_columns] = X[numeric_columns].fillna(
            X[numeric_columns].median()
        )

    # Fill any remaining missing values
    X = X.fillna(0)

    # Handle target column
    # Supports object, category, and pandas string dtypes
    if (
        pd.api.types.is_object_dtype(y)
        or pd.api.types.is_categorical_dtype(y)
        or pd.api.types.is_string_dtype(y)
    ):
        y = y.fillna("Missing").astype(str)

        label_encoder = LabelEncoder()
        y = label_encoder.fit_transform(y)

    else:
        # Numeric target
        y = y.fillna(y.median())

    return train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state
    )
