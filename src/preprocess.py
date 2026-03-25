import pandas as pd
import numpy as np
from config import PREDICTION_HORIZON


def parse_timestamp(df: pd.DataFrame) -> pd.DataFrame:
    """Parse reference_timestamp to datetime."""
    df = df.copy()
    df["timestamp"] = pd.to_datetime(df["reference_timestamp"], format="%d.%m.%Y %H:%M")
    df = df.drop(columns=["reference_timestamp"])
    df = df.sort_values("timestamp").reset_index(drop=True)
    return df


def handle_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    """Fill missing values using forward fill then backward fill."""
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    df[numeric_cols] = df[numeric_cols].ffill().bfill()
    return df


def create_target(df: pd.DataFrame, target_col: str = "wind_speed") -> pd.DataFrame:
    """Create target variable: wind speed PREDICTION_HORIZON steps ahead."""
    df = df.copy()
    df["target"] = df[target_col].shift(-PREDICTION_HORIZON)
    return df.dropna(subset=["target"])


def preprocess(df: pd.DataFrame) -> pd.DataFrame:
    """Full preprocessing pipeline."""
    df = parse_timestamp(df)
    df = handle_missing_values(df)
    df = create_target(df)
    return df


if __name__ == "__main__":
    from load_data import load_train_data, rename_columns

    print("Loading and preprocessing training data...")
    raw_df = rename_columns(load_train_data())
    df = preprocess(raw_df)
    print(f"Preprocessed shape: {df.shape}")
    print(f"Date range: {df['timestamp'].min()} to {df['timestamp'].max()}")
    print(f"Missing values:\n{df.isnull().sum()}")
