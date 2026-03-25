import os
import json
import numpy as np
import pandas as pd
import joblib
from sklearn.model_selection import TimeSeriesSplit, GridSearchCV
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
import xgboost as xgb
from config import XGB_PARAMS, GRID_SEARCH_PARAMS
from load_data import load_train_data, load_test_data, rename_columns
from preprocess import preprocess
from features import engineer_features, get_feature_columns


def prepare_data():
    """Load, preprocess, and engineer features for train and test."""
    print("Loading training data...")
    train_raw = rename_columns(load_train_data())
    train_df = preprocess(train_raw)
    train_df = engineer_features(train_df)

    print("Loading test data...")
    test_raw = rename_columns(load_test_data())
    test_df = preprocess(test_raw)
    test_df = engineer_features(test_df)

    feature_cols = get_feature_columns(train_df)

    X_train = train_df[feature_cols].values
    y_train = train_df["target"].values
    X_test = test_df[feature_cols].values
    y_test = test_df["target"].values

    X_train = np.nan_to_num(X_train, nan=0.0)
    X_test = np.nan_to_num(X_test, nan=0.0)

    print(f"X_train: {X_train.shape}, X_test: {X_test.shape}")
    return X_train, y_train, X_test, y_test, feature_cols


def grid_search(X_train, y_train):
    """Perform grid search with time series cross-validation."""
    print("\nStarting grid search...")
    base_model = xgb.XGBRegressor(**XGB_PARAMS)

    tscv = TimeSeriesSplit(n_splits=3)

    grid = GridSearchCV(
        estimator=base_model,
        param_grid=GRID_SEARCH_PARAMS,
        cv=tscv,
        scoring="neg_mean_squared_error",
        verbose=2,
        n_jobs=1,
        return_train_score=True,
    )

    grid.fit(X_train, y_train)

    print(f"\nBest params: {grid.best_params_}")
    print(f"Best CV RMSE: {np.sqrt(-grid.best_score_):.4f}")

    return grid.best_estimator_, grid.cv_results_


def evaluate_model(model, X_test, y_test, feature_cols):
    """Evaluate model on test set and compute metrics."""
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

    print(f"\nTop 10 Features:")
    print(feature_importance.head(10).to_string(index=False))

    return {
        "r2": r2,
        "mae": mae,
        "rmse": rmse,
        "mape": mape,
        "feature_importance": feature_importance,
    }


def save_results(model, metrics, cv_results, feature_cols):
    """Save model and results."""
    os.makedirs("models", exist_ok=True)
    os.makedirs("results", exist_ok=True)

    model.save_model("models/best_model.json")
    print("\nModel saved to models/best_model.json")

    with open("results/metrics.json", "w") as f:
        json.dump(metrics, f, default=str, indent=2)

    pd.DataFrame(cv_results).to_csv("results/cv_results.csv", index=False)

    metrics["feature_importance"].to_csv("results/feature_importance.csv", index=False)


if __name__ == "__main__":
    X_train, y_train, X_test, y_test, feature_cols = prepare_data()

    best_model, cv_results = grid_search(X_train, y_train)

    metrics = evaluate_model(best_model, X_test, y_test, feature_cols)

    save_results(best_model, metrics, cv_results, feature_cols)
