"""
Rossmann Retail Sales Forecasting & Promotion Impact Analysis
Pipeline script — produces the metrics and figures referenced in the README
and mirrors the notebook exactly (deterministic, random_state=42 throughout).
"""
import json
import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor

warnings.filterwarnings("ignore")
sns.set_style("whitegrid")

DATA_DIR = Path(__file__).parent / "data"
OUT_DIR = Path(__file__).parent / "outputs"
OUT_DIR.mkdir(exist_ok=True)

RANDOM_STATE = 42
N_SAMPLE_STORES = 100  # data/ ships pre-sampled to 100 stores (GitHub 25MB limit); set to None if you've dropped in the full 1,115-store Kaggle files

# ---------------------------------------------------------------------------
# 1. Load & merge
# ---------------------------------------------------------------------------
train = pd.read_csv(DATA_DIR / "train.csv", parse_dates=["Date"], low_memory=False)
store = pd.read_csv(DATA_DIR / "store.csv")

df = train.merge(store, on="Store", how="left")

# ---------------------------------------------------------------------------
# 2. Data quality / cleaning
# ---------------------------------------------------------------------------
# Closed-store days have zero sales by definition and add no forecasting signal
df = df[df["Open"] == 1].copy()
df = df[df["Sales"] > 0].copy()

df["CompetitionDistance"] = df["CompetitionDistance"].fillna(df["CompetitionDistance"].median())
df["CompetitionOpenSinceMonth"] = df["CompetitionOpenSinceMonth"].fillna(0)
df["CompetitionOpenSinceYear"] = df["CompetitionOpenSinceYear"].fillna(0)
df["Promo2SinceWeek"] = df["Promo2SinceWeek"].fillna(0)
df["Promo2SinceYear"] = df["Promo2SinceYear"].fillna(0)
df["PromoInterval"] = df["PromoInterval"].fillna("")
df["StateHoliday"] = df["StateHoliday"].astype(str).replace({"0": "None"})

# Reduce to a representative, reproducible sample of stores for fast local runs.
# The exact same code trains on all 1,115 stores if N_SAMPLE_STORES is set to None.
if N_SAMPLE_STORES:
    rng = np.random.RandomState(RANDOM_STATE)
    sample_stores = rng.choice(df["Store"].unique(), size=N_SAMPLE_STORES, replace=False)
    df = df[df["Store"].isin(sample_stores)].copy()

# ---------------------------------------------------------------------------
# 3. Feature engineering
# ---------------------------------------------------------------------------
df = df.sort_values(["Store", "Date"])

df["Year"] = df["Date"].dt.year
df["Month"] = df["Date"].dt.month
df["Day"] = df["Date"].dt.day
df["WeekOfYear"] = df["Date"].dt.isocalendar().week.astype(int)
df["IsWeekend"] = df["DayOfWeek"].isin([6, 7]).astype(int)

df["StoreType"] = df["StoreType"].astype("category").cat.codes
df["Assortment"] = df["Assortment"].astype("category").cat.codes
df["StateHoliday"] = df["StateHoliday"].astype("category").cat.codes

df["CompetitionOpenMonths"] = (
    (df["Year"] - df["CompetitionOpenSinceYear"]) * 12
    + (df["Month"] - df["CompetitionOpenSinceMonth"])
).clip(lower=0)

# lag / rolling features per store (previous 7 & 28-day average sales)
df["Sales_Lag7"] = df.groupby("Store")["Sales"].shift(7)
df["Sales_Roll7_Mean"] = df.groupby("Store")["Sales"].transform(lambda s: s.shift(1).rolling(7).mean())
df["Sales_Roll28_Mean"] = df.groupby("Store")["Sales"].transform(lambda s: s.shift(1).rolling(28).mean())

df = df.dropna(subset=["Sales_Lag7", "Sales_Roll7_Mean", "Sales_Roll28_Mean"]).copy()

FEATURES = [
    "Store", "DayOfWeek", "Promo", "StateHoliday", "SchoolHoliday",
    "StoreType", "Assortment", "CompetitionDistance", "CompetitionOpenMonths",
    "Promo2", "Year", "Month", "Day", "WeekOfYear", "IsWeekend",
    "Sales_Lag7", "Sales_Roll7_Mean", "Sales_Roll28_Mean",
]
TARGET = "Sales"

# ---------------------------------------------------------------------------
# 4. Train / test split — time-based (last 6 weeks held out, matching the
#    real Kaggle competition's forecast horizon)
# ---------------------------------------------------------------------------
cutoff = df["Date"].max() - pd.Timedelta(days=42)
train_df = df[df["Date"] <= cutoff]
test_df = df[df["Date"] > cutoff]

X_train, y_train = train_df[FEATURES], train_df[TARGET]
X_test, y_test = test_df[FEATURES], test_df[TARGET]

# ---------------------------------------------------------------------------
# 5. Baseline model
# ---------------------------------------------------------------------------
baseline = LinearRegression()
baseline.fit(X_train, y_train)
base_preds = baseline.predict(X_test)

# ---------------------------------------------------------------------------
# 6. Random Forest model
# ---------------------------------------------------------------------------
rf = RandomForestRegressor(
    n_estimators=200, max_depth=14, min_samples_leaf=3,
    random_state=RANDOM_STATE, n_jobs=-1,
)
rf.fit(X_train, y_train)
rf_preds = rf.predict(X_test)

# ---------------------------------------------------------------------------
# 6b. XGBoost model — gradient boosting comparison
# ---------------------------------------------------------------------------
xgb = XGBRegressor(
    n_estimators=400, max_depth=6, learning_rate=0.05,
    subsample=0.85, colsample_bytree=0.85,
    random_state=RANDOM_STATE, n_jobs=-1,
)
xgb.fit(X_train, y_train)
xgb_preds = xgb.predict(X_test)


def rmspe(y_true, y_pred):
    mask = y_true != 0
    return float(np.sqrt(np.mean(((y_true[mask] - y_pred[mask]) / y_true[mask]) ** 2)))


def evaluate(y_true, y_pred, name):
    return {
        "model": name,
        "MAE": round(mean_absolute_error(y_true, y_pred), 2),
        "RMSE": round(np.sqrt(mean_squared_error(y_true, y_pred)), 2),
        "RMSPE": round(rmspe(y_true.values, y_pred), 4),
        "R2": round(r2_score(y_true, y_pred), 4),
    }


results = [
    evaluate(y_test, base_preds, "Linear Regression (baseline)"),
    evaluate(y_test, rf_preds, "Random Forest"),
    evaluate(y_test, xgb_preds, "XGBoost"),
]

# ---------------------------------------------------------------------------
# 7. Feature importance (best model — chosen dynamically by RMSPE)
# ---------------------------------------------------------------------------
best_result = min(results[1:], key=lambda r: r["RMSPE"])  # best of RF/XGBoost
best_model = xgb if best_result["model"] == "XGBoost" else rf
importance = pd.Series(best_model.feature_importances_, index=FEATURES).sort_values(ascending=False)

# ---------------------------------------------------------------------------
# 8. Promo impact (business question, not just model accuracy)
# ---------------------------------------------------------------------------
promo_impact = df.groupby("Promo")["Sales"].mean()
promo_lift_pct = round((promo_impact[1] / promo_impact[0] - 1) * 100, 1)

storetype_sales = df.groupby("StoreType")["Sales"].mean().sort_values(ascending=False)

# ---------------------------------------------------------------------------
# 9. Save metrics
# ---------------------------------------------------------------------------
metrics = {
    "n_stores_sampled": int(df["Store"].nunique()),
    "n_rows_used": int(len(df)),
    "date_range": [str(df["Date"].min().date()), str(df["Date"].max().date())],
    "test_period": [str(cutoff.date()), str(df["Date"].max().date())],
    "results": results,
    "best_model": best_result["model"],
    "promo_lift_pct": promo_lift_pct,
    "top_features": importance.head(8).round(4).to_dict(),
}
with open(OUT_DIR / "metrics.json", "w") as f:
    json.dump(metrics, f, indent=2)

print(json.dumps(metrics, indent=2))

# ---------------------------------------------------------------------------
# 10. Figures
# ---------------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(9, 5))
importance.head(10).sort_values().plot(kind="barh", ax=ax, color="#4F46E5")
ax.set_title(f"Top 10 Sales Drivers ({best_result['model']} Feature Importance)")
plt.tight_layout()
plt.savefig(OUT_DIR / "feature_importance.png", dpi=140)
plt.close()

fig, ax = plt.subplots(figsize=(8, 5))
model_names = [r["model"] for r in results]
rmspe_vals = [r["RMSPE"] for r in results]
colors = ["#94A3B8", "#4F46E5", "#059669"]
ax.bar(model_names, rmspe_vals, color=colors[: len(model_names)])
ax.set_ylabel("RMSPE (lower is better)")
ax.set_title("Model Comparison — Forecast Error (RMSPE)")
for i, v in enumerate(rmspe_vals):
    ax.text(i, v + 0.003, f"{v:.1%}", ha="center", fontweight="bold")
plt.tight_layout()
plt.savefig(OUT_DIR / "model_comparison.png", dpi=140)
plt.close()

fig, ax = plt.subplots(figsize=(9, 5))
sample_store_id = int(df["Store"].unique()[0])
s = df[df["Store"] == sample_store_id].sort_values("Date")
ax.plot(s["Date"], s["Sales"], label="Actual", color="#4F46E5", linewidth=1)
ax.set_title(f"Daily Sales Pattern — Store {sample_store_id}")
ax.set_ylabel("Sales (€)")
plt.tight_layout()
plt.savefig(OUT_DIR / "sample_store_trend.png", dpi=140)
plt.close()

fig, ax = plt.subplots(figsize=(6, 5))
promo_impact.plot(kind="bar", ax=ax, color=["#94A3B8", "#059669"])
ax.set_xticklabels(["No Promo", "Promo"], rotation=0)
ax.set_ylabel("Avg. Daily Sales (€)")
ax.set_title(f"Promotion Impact: +{promo_lift_pct}% Avg. Sales Lift")
plt.tight_layout()
plt.savefig(OUT_DIR / "promo_impact.png", dpi=140)
plt.close()

print("\nSaved figures to", OUT_DIR)
