STATION = "BOU"
PREDICTION_HORIZON = 1
DATA_DIR = "data/stations"

RAW_COLUMNS = [
    "reference_timestamp",
    "fkl010z0",
    "fkl010z1",
    "dkl010z0",
    "tre200s0",
    "tresurs0",
    "xchills0",
    "prestas0",
    "pp0qffs0",
    "ure200s0",
    "tde200s0",
]

COLUMN_RENAME = {
    "fkl010z0": "wind_speed",
    "fkl010z1": "wind_gust",
    "dkl010z0": "wind_direction",
    "tre200s0": "temperature",
    "tresurs0": "temperature_surface",
    "xchills0": "wind_chill",
    "prestas0": "pressure",
    "pp0qffs0": "pressure_sea_level",
    "ure200s0": "humidity",
    "tde200s0": "dew_point",
}

LAG_FEATURES = ["wind_speed", "pressure", "temperature", "humidity"]
ROLLING_FEATURES = ["wind_speed", "pressure", "temperature"]
WINDOW_SIZE = 6

EXTENDED_LAGS = [1, 2, 3, 4, 5, 6, 12, 18, 24]
ROLLING_WINDOWS = [6, 12, 24]

TRAIN_YEARS = list(range(2000, 2020))
TEST_YEARS = list(range(2020, 2030))

XGB_PARAMS = {
    "objective": "reg:squarederror",
    "tree_method": "hist",
    "device": "cuda",
    "n_jobs": -1,
    "random_state": 42,
}

GRID_SEARCH_PARAMS = {
    "max_depth": [6, 8],
    "n_estimators": [200, 400],
    "learning_rate": [0.1],
    "min_child_weight": [1, 3],
}
