"""Utilities for preparing uploaded datasets for machine learning."""

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder


def prepare_train_test_data(df, target_column, test_size=0.2, random_state=42):
    """Encode and complete a dataset, then return its train/test split."""

    X = df.drop(columns=[target_column]).copy()
    y = df[target_column].copy()

    if X.empty:
        raise ValueError("The dataset must contain at least one feature column.")

    # Encode categorical feature columns
    categorical_columns = X.select_dtypes(
        include=["object", "category"]
    ).columns

    for column in categorical_columns:
        X[column] = X[column].fillna("Missing")
        X[column] = LabelEncoder().fit_transform(X[column].astype(str))

    # Fill missing numeric values
    numeric_columns = X.select_dtypes(include="number").columns

    X[numeric_columns] = X[numeric_columns].fillna(
        X[numeric_columns].median()
    )

    X = X.fillna(0)

    # Handle target column
    if y.dtype == "object" or y.dtype.name == "category":
        y = LabelEncoder().fit_transform(
            y.fillna("Missing").astype(str)
        )
    else:
        y = y.fillna(y.median())

    return train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state
    )