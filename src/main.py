import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import json
import numpy as np
import pandas as pd
from sklearn.model_selection import TimeSeriesSplit, GridSearchCV
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
import xgboost as xgb

from config import (
    XGB_PARAMS,
    GRID_SEARCH_PARAMS,
    STATION,
    PREDICTION_HORIZON,
    LAG_FEATURES,
    ROLLING_FEATURES,
    WINDOW_SIZE,
    DATA_DIR,
    TRAIN_YEARS,
    TEST_YEARS,
)
from load_data import load_train_data, load_test_data, rename_columns
from preprocess import preprocess
from features import engineer_features, get_feature_columns


os.makedirs("models", exist_ok=True)
os.makedirs("results", exist_ok=True)


def prepare_data():
    print("=" * 50)
    print("STEP 1: Loading and preprocessing data")
    print("=" * 50)

    print("\nLoading training data (2000-2019)...")
    train_raw = rename_columns(load_train_data())
    print(f"  Raw training shape: {train_raw.shape}")

    train_df = preprocess(train_raw)
    print(f"  After preprocessing: {train_df.shape}")

    train_df = engineer_features(train_df)
    print(f"  After feature engineering: {train_df.shape}")

    print("\nLoading test data (2020-2029)...")
    test_raw = rename_columns(load_test_data())
    print(f"  Raw test shape: {test_raw.shape}")

    test_df = preprocess(test_raw)
    print(f"  After preprocessing: {test_df.shape}")

    test_df = engineer_features(test_df)
    print(f"  After feature engineering: {test_df.shape}")

    feature_cols = get_feature_columns(train_df)
    print(f"\nFeature count: {len(feature_cols)}")

    X_train = train_df[feature_cols].values
    y_train = train_df["target"].values
    X_test = test_df[feature_cols].values
    y_test = test_df["target"].values

    X_train = np.nan_to_num(X_train, nan=0.0)
    X_test = np.nan_to_num(X_test, nan=0.0)

    print(f"\nFinal shapes:")
    print(f"  X_train: {X_train.shape}, y_train: {y_train.shape}")
    print(f"  X_test:  {X_test.shape}, y_test:  {y_test.shape}")

    return X_train, y_train, X_test, y_test, feature_cols


def grid_search(X_train, y_train):
    print("\n" + "=" * 50)
    print("STEP 2: Grid search with time series CV")
    print("=" * 50)

    base_model = xgb.XGBRegressor(**XGB_PARAMS)
    tscv = TimeSeriesSplit(n_splits=3)

    param_combinations = (
        len(GRID_SEARCH_PARAMS["max_depth"])
        * len(GRID_SEARCH_PARAMS["n_estimators"])
        * len(GRID_SEARCH_PARAMS["learning_rate"])
        * len(GRID_SEARCH_PARAMS["min_child_weight"])
    )
    print(f"Total combinations: {param_combinations}")
    print(f"Folds: 3, Total fits: {param_combinations * 3}")

    grid = GridSearchCV(
        estimator=base_model,
        param_grid=GRID_SEARCH_PARAMS,
        cv=tscv,
        scoring="neg_mean_squared_error",
        verbose=1,
        n_jobs=1,
    )

    grid.fit(X_train, y_train)

    print(f"\nBest parameters: {grid.best_params_}")
    print(f"Best CV RMSE: {np.sqrt(-grid.best_score_):.4f}")

    return grid.best_estimator_, grid.cv_results_


def evaluate_model(model, X_test, y_test, feature_cols):
    print("\n" + "=" * 50)
    print("STEP 3: Evaluating on test set")
    print("=" * 50)

    y_pred = model.predict(X_test)

    r2 = r2_score(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    mape = np.mean(np.abs((y_test - y_pred) / (y_test + 1e-10))) * 100

    print(f"\nTest Metrics:")
    print(f"  R²:   {r2:.4f}")
    print(f"  MAE:  {mae:.4f}")
    print(f"  RMSE: {rmse:.4f}")
    print(f"  MAPE: {mape:.2f}%")

    feature_importance = pd.DataFrame(
        {"feature": feature_cols, "importance": model.feature_importances_}
    ).sort_values("importance", ascending=False)

    print(f"\nTop 15 Features:")
    print(feature_importance.head(15).to_string(index=False))

    return {
        "r2": float(r2),
        "mae": float(mae),
        "rmse": float(rmse),
        "mape": float(mape),
        "feature_importance": feature_importance.to_dict("records"),
    }


def save_results(model, metrics, cv_results):
    print("\n" + "=" * 50)
    print("STEP 4: Saving results")
    print("=" * 50)

    model.save_model("models/best_model.json")
    print("  Saved: models/best_model.json")

    with open("results/metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)
    print("  Saved: results/metrics.json")

    pd.DataFrame(cv_results).to_csv("results/cv_results.csv", index=False)
    print("  Saved: results/cv_results.csv")

    pd.DataFrame(metrics["feature_importance"]).to_csv(
        "results/feature_importance.csv", index=False
    )
    print("  Saved: results/feature_importance.csv")


def main():
    print("Wind Speed Prediction Pipeline")
    print(f"  Station: {STATION}")
    print(f"  Prediction horizon: {PREDICTION_HORIZON} timesteps (10-min)")
    print(f"  Target: 1 hour ahead")

    X_train, y_train, X_test, y_test, feature_cols = prepare_data()

    best_model, cv_results = grid_search(X_train, y_train)

    metrics = evaluate_model(best_model, X_test, y_test, feature_cols)

    save_results(best_model, metrics, cv_results)

    print("\n" + "=" * 50)
    print("PIPELINE COMPLETE")
    print("=" * 50)


if __name__ == "__main__":
    main()
