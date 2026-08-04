import pandas as pd


def load_dataset(uploaded_file):
    if uploaded_file.name.endswith(".csv"):
        return pd.read_csv(uploaded_file)
    if uploaded_file.name.endswith(".xlsx"):
        return pd.read_excel(uploaded_file)
    raise ValueError("Unsupported file format.")


def remove_duplicates(df):
    return df.drop_duplicates()


def remove_missing_values(df):
    return df.dropna()
