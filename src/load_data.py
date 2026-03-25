import os
import pandas as pd
from config import (
    STATION,
    DATA_DIR,
    TRAIN_YEARS,
    TEST_YEARS,
    RAW_COLUMNS,
    COLUMN_RENAME,
)


def load_station_data(station: str = STATION, years: list = None) -> pd.DataFrame:
    """Load all CSV files for a station and concatenate."""
    station_dir = os.path.join(DATA_DIR, station)

    all_dfs = []
    for f in sorted(os.listdir(station_dir)):
        if not f.startswith(f"ogd-smn_{station.lower()}_t_historical"):
            continue

        file_year = int(f.split("_")[-1].split("-")[0])
        if years is not None and file_year not in years:
            continue

        filepath = os.path.join(station_dir, f)
        print(f"Loading {f}...")
        df = pd.read_csv(filepath, sep=";", usecols=RAW_COLUMNS, low_memory=False)
        all_dfs.append(df)

    combined = pd.concat(all_dfs, ignore_index=True)
    return combined
    """Load all CSV files for a station and concatenate."""
    station_dir = os.path.join(DATA_DIR, station)

    all_dfs = []
    for f in sorted(os.listdir(station_dir)):
        if not f.startswith(f"ogd-smn_{station.lower()}_t_historical"):
            continue

        file_year = int(f.split("_")[-1].split("-")[0])
        if years and file_year not in years:
            continue

        filepath = os.path.join(station_dir, f)
        print(f"Loading {f}...")
        df = pd.read_csv(filepath, sep=";", usecols=RAW_COLUMNS, low_memory=False)
        all_dfs.append(df)

    combined = pd.concat(all_dfs, ignore_index=True)
    return combined


def load_train_data() -> pd.DataFrame:
    """Load training data (2000-2019)."""
    return load_station_data(years=TRAIN_YEARS)


def load_test_data() -> pd.DataFrame:
    """Load test data (2020-2029)."""
    return load_station_data(years=TEST_YEARS)


def rename_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Rename columns to readable names."""
    return df.rename(columns=COLUMN_RENAME)


if __name__ == "__main__":
    print("Loading training data...")
    train_df = load_train_data()
    print(f"Training data shape: {train_df.shape}")

    print("\nLoading test data...")
    test_df = load_test_data()
    print(f"Test data shape: {test_df.shape}")
