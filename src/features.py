import pandas as pd
import numpy as np
from config import LAG_FEATURES, ROLLING_FEATURES, WINDOW_SIZE, EXTENDED_LAGS


def temporal_cycles(df: pd.DataFrame) -> pd.DataFrame:
    """Add cyclical encoding for hour, month, day_of_year."""
    df = df.copy()
    df["hour"] = df["timestamp"].dt.hour
    df["month"] = df["timestamp"].dt.month
    df["day_of_year"] = df["timestamp"].dt.dayofyear

    df["hour_sin"] = np.sin(2 * np.pi * df["hour"] / 24)
    df["hour_cos"] = np.cos(2 * np.pi * df["hour"] / 24)
    df["month_sin"] = np.sin(2 * np.pi * df["month"] / 12)
    df["month_cos"] = np.cos(2 * np.pi * df["month"] / 12)
    df["day_of_year_sin"] = np.sin(2 * np.pi * df["day_of_year"] / 365)
    df["day_of_year_cos"] = np.cos(2 * np.pi * df["day_of_year"] / 365)

    return df.drop(columns=["hour", "month", "day_of_year"])


def lag_features(
    df: pd.DataFrame, features: list = LAG_FEATURES, n_lags: list = None
) -> pd.DataFrame:
    """Add lagged features for past n timesteps."""
    if n_lags is None:
        n_lags = EXTENDED_LAGS
    df = df.copy()
    for feat in features:
        if feat not in df.columns:
            continue
        for lag in n_lags:
            df[f"{feat}_lag_{lag}"] = df[feat].shift(lag)
    return df


def rolling_features(
    df: pd.DataFrame, features: list = ROLLING_FEATURES, windows: list = None
) -> pd.DataFrame:
    """Add rolling statistics (mean, std) over past window timesteps."""
    if windows is None:
        windows = [6, 12, 24]
    df = df.copy()
    for feat in features:
        if feat not in df.columns:
            continue
        for window in windows:
            df[f"{feat}_rolling_mean_{window}"] = (
                df[feat].rolling(window=window, min_periods=1).mean()
            )
            df[f"{feat}_rolling_std_{window}"] = (
                df[feat].rolling(window=window, min_periods=1).std()
            )
    return df


def direction_features(df: pd.DataFrame) -> pd.DataFrame:
    """Convert wind direction to vector components (u, v)."""
    if "wind_direction" not in df.columns:
        return df
    df = df.copy()
    wind_dir_rad = np.deg2rad(df["wind_direction"])
    df["wind_u"] = -df["wind_speed"] * np.sin(wind_dir_rad)
    df["wind_v"] = -df["wind_speed"] * np.cos(wind_dir_rad)
    return df


def trend_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add linear trend over past windows (pressure/temperature change)."""
    df = df.copy()
    for feat in ["pressure", "temperature"]:
        if feat not in df.columns:
            continue
        for window in [6, 12, 24]:
            df[f"{feat}_trend_{window}"] = df[feat] - df[feat].shift(window)
    return df


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Full feature engineering pipeline."""
    df = temporal_cycles(df)
    df = lag_features(df)
    df = rolling_features(df)
    df = direction_features(df)
    df = trend_features(df)
    return df


def get_feature_columns(df: pd.DataFrame, exclude: list = None) -> list:
    """Get list of feature columns (exclude target, timestamp, raw meteo)."""
    if exclude is None:
        exclude = ["timestamp", "target"]

    feature_cols = [col for col in df.columns if col not in exclude]
    exclude_raw = [
        "wind_speed",
        "wind_gust",
        "wind_direction",
        "temperature",
        "temperature_surface",
        "wind_chill",
        "pressure",
        "pressure_sea_level",
        "humidity",
        "dew_point",
    ]
    feature_cols = [col for col in feature_cols if col not in exclude_raw]
    return feature_cols


if __name__ == "__main__":
    from load_data import load_train_data, rename_columns
    from preprocess import preprocess

    print("Loading and preprocessing...")
    raw_df = rename_columns(load_train_data())
    df = preprocess(raw_df)

    print("Engineering features...")
    df = engineer_features(df)
    print(f"Features shape: {df.shape}")
    print(f"Feature columns: {get_feature_columns(df)}")
