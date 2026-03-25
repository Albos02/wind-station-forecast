import os
import sys
import json
import time
import subprocess

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

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
    EXTENDED_LAGS,
    ROLLING_WINDOWS,
    DATA_DIR,
    TRAIN_YEARS,
    TEST_YEARS,
)
from load_data import load_train_data, load_test_data, rename_columns
from preprocess import preprocess
from features import engineer_features, get_feature_columns


os.makedirs("models", exist_ok=True)
os.makedirs("results", exist_ok=True)
os.makedirs("runs", exist_ok=True)


def get_git_info():
    try:
        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            cwd=os.path.dirname(__file__),
        ).stdout.strip()[:8]
        branch = subprocess.run(
            ["git", "rev-parse", "--abbrev-ref", "HEAD"],
            capture_output=True,
            text=True,
            cwd=os.path.dirname(__file__),
        ).stdout.strip()
        return commit, branch
    except:
        return "unknown", "unknown"


def log_run(commit, branch, config_summary, metrics, elapsed_time):
    run_id = f"run_{time.strftime('%Y%m%d_%H%M%S')}"
    run_dir = f"runs/{run_id}"
    os.makedirs(run_dir, exist_ok=True)
    os.makedirs(f"{run_dir}/models", exist_ok=True)

    run_entry = {
        "run_id": run_id,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "git_commit": commit,
        "git_branch": branch,
        "elapsed_seconds": round(elapsed_time, 1),
    }

    run_log_path = "runs/run_log.json"
    if os.path.exists(run_log_path):
        with open(run_log_path, "r") as f:
            runs = json.load(f)
    else:
        runs = []

    previous_best_r2 = max(
        (r.get("metrics", {}).get("r2", 0) for r in runs), default=None
    )
    is_best = metrics["r2"] > previous_best_r2 if previous_best_r2 is not None else True
    run_entry["is_best"] = is_best
    run_entry["metrics"] = {
        "r2": metrics["r2"],
        "mae": metrics["mae"],
        "rmse": metrics["rmse"],
    }

    runs.append(run_entry)

    with open(run_log_path, "w") as f:
        json.dump(runs, f, indent=2)

    with open(f"{run_dir}/config.json", "w") as f:
        json.dump(config_summary, f, indent=2)

    with open(f"{run_dir}/metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    return is_best, previous_best_r2, run_id


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


def save_results(model, metrics, cv_results, run_id=None):
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

    if run_id:
        run_dir = f"runs/{run_id}"
        model.save_model(f"{run_dir}/model.json")
        with open(f"{run_dir}/metrics.json", "w") as f:
            json.dump(metrics, f, indent=2)
        pd.DataFrame(cv_results).to_csv(f"{run_dir}/cv_results.csv", index=False)
        pd.DataFrame(metrics["feature_importance"]).to_csv(
            f"{run_dir}/feature_importance.csv", index=False
        )
        print(f"  Saved: {run_dir}/")


def main():
    start_time = time.time()

    print("Wind Speed Prediction Pipeline")
    print(f"  Station: {STATION}")
    print(f"  Prediction horizon: {PREDICTION_HORIZON} timesteps (10-min)")
    print(f"  Target: 1 hour ahead")

    X_train, y_train, X_test, y_test, feature_cols = prepare_data()

    best_model, cv_results = grid_search(X_train, y_train)

    metrics = evaluate_model(best_model, X_test, y_test, feature_cols)

    elapsed_time = time.time() - start_time

    commit, branch = get_git_info()

    config_summary = {
        "station": STATION,
        "prediction_horizon": PREDICTION_HORIZON,
        "lag_features": LAG_FEATURES,
        "extended_lags": EXTENDED_LAGS,
        "rolling_windows": ROLLING_WINDOWS,
        "grid_search": GRID_SEARCH_PARAMS,
    }

    is_best, previous_best_r2, run_id = log_run(
        commit, branch, config_summary, metrics, elapsed_time
    )

    save_results(best_model, metrics, cv_results, run_id)

    with open(f"runs/{run_id}/config.json", "w") as f:
        json.dump(config_summary, f, indent=2)

    print(f"\n  Saved: runs/run_log.json and runs/{run_id}/")

    run_log_path = "runs/run_log.json"
    with open(run_log_path, "r") as f:
        runs = json.load(f)

    print("\n" + "=" * 50)
    print("PIPELINE COMPLETE")
    print(f"  Elapsed time: {elapsed_time:.1f} seconds")
    if is_best:
        prev = f"{previous_best_r2:.4f}" if previous_best_r2 else "N/A"
        print(f"  ★ NEW BEST R²: {metrics['r2']:.4f} (previous best: {prev})")
    else:
        print(
            f"  Best R² so far: {max(r.get('metrics', {}).get('r2', 0) for r in runs):.4f}"
        )
    print("=" * 50)


if __name__ == "__main__":
    main()
