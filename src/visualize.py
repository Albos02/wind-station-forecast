import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import xgboost as xgb

from config import STATION
from load_data import load_test_data, rename_columns
from preprocess import preprocess
from features import engineer_features, get_feature_columns

sns.set_style("whitegrid")

model = xgb.XGBRegressor()
model.load_model("models/best_model.json")

print("Loading test data...")
test_raw = rename_columns(load_test_data())
test_df = preprocess(test_raw)
test_df = engineer_features(test_df)

feature_cols = get_feature_columns(test_df)
X_test = test_df[feature_cols].values
y_test = test_df["target"].values
X_test = np.nan_to_num(X_test, nan=0.0)

y_pred = model.predict(X_test)

fig, axes = plt.subplots(2, 2, figsize=(14, 10))

ax1 = axes[0, 0]
fi = pd.read_csv("results/feature_importance.csv")
top_15 = fi.head(15)
ax1.barh(top_15["feature"], top_15["importance"], color="steelblue")
ax1.invert_yaxis()
ax1.set_xlabel("Importance")
ax1.set_title("Top 15 Feature Importances")

ax2 = axes[0, 1]
sample_idx = np.random.choice(len(y_test), 5000, replace=False)
ax2.scatter(y_test[sample_idx], y_pred[sample_idx], alpha=0.3, s=5)
ax2.plot([0, y_test.max()], [0, y_test.max()], "r--", lw=2)
ax2.set_xlabel("Actual Wind Speed (m/s)")
ax2.set_ylabel("Predicted Wind Speed (m/s)")
ax2.set_title("Actual vs Predicted")

ax3 = axes[1, 0]
sample_start = np.random.randint(0, len(y_test) - 500)
ax3.plot(y_test[sample_start : sample_start + 500], label="Actual", alpha=0.7)
ax3.plot(y_pred[sample_start : sample_start + 500], label="Predicted", alpha=0.7)
ax3.set_xlabel("Time Step (10-min)")
ax3.set_ylabel("Wind Speed (m/s)")
ax3.set_title("Time Series: Actual vs Predicted")
ax3.legend()

ax4 = axes[1, 1]
residuals = y_test - y_pred
ax4.hist(residuals, bins=50, edgecolor="black", alpha=0.7)
ax4.axvline(0, color="red", linestyle="--", lw=2)
ax4.set_xlabel("Residual (m/s)")
ax4.set_ylabel("Frequency")
ax4.set_title("Residual Distribution")

plt.suptitle(
    f"Wind Speed Prediction - Station {STATION}", fontsize=14, fontweight="bold"
)
plt.tight_layout()
plt.show()
plt.savefig("results/prediction_plots.png", dpi=150, bbox_inches="tight")
plt.close()

print("Saved: results/prediction_plots.png")

with open("results/metrics.json", "r") as f:
    metrics = json.load(f)

print(
    f"\nMetrics: R²={metrics['r2']:.3f}, MAE={metrics['mae']:.3f}, RMSE={metrics['rmse']:.3f}"
)
