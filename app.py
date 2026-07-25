"""
Rossmann Retail Sales Forecasting Dashboard
Live companion to Sales_Forecasting_Rossmann.ipynb

Run locally:
    pip install -r requirements.txt
    streamlit run app.py
"""

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from xgboost import XGBRegressor

st.set_page_config(
    page_title="Retail Sales Forecasting Dashboard",
    page_icon="📈",
    layout="wide",
)

PRIMARY = "#4F46E5"
GREEN = "#059669"
GRAY = "#94A3B8"

REQUIRED_TRAIN_COLS = ["Store", "DayOfWeek", "Date", "Sales", "Customers", "Open", "Promo", "StateHoliday", "SchoolHoliday"]
REQUIRED_STORE_COLS = ["Store", "StoreType", "Assortment", "CompetitionDistance"]


@st.cache_data
def load_default():
    train = pd.read_csv("data/train.csv", parse_dates=["Date"], low_memory=False)
    store = pd.read_csv("data/store.csv")
    return train, store


@st.cache_data
def prepare(train: pd.DataFrame, store: pd.DataFrame, n_sample_stores: int | None):
    df = train.merge(store, on="Store", how="left")
    df = df[df["Open"] == 1].copy()
    df = df[df["Sales"] > 0].copy()

    df["CompetitionDistance"] = df["CompetitionDistance"].fillna(df["CompetitionDistance"].median())
    for col in ["CompetitionOpenSinceMonth", "CompetitionOpenSinceYear", "Promo2SinceWeek", "Promo2SinceYear"]:
        if col in df.columns:
            df[col] = df[col].fillna(0)
    if "Promo2" not in df.columns:
        df["Promo2"] = 0
    df["StateHoliday"] = df["StateHoliday"].astype(str).replace({"0": "None"})

    if n_sample_stores:
        rng = np.random.RandomState(42)
        stores = df["Store"].unique()
        if len(stores) > n_sample_stores:
            sample = rng.choice(stores, size=n_sample_stores, replace=False)
            df = df[df["Store"].isin(sample)].copy()

    df = df.sort_values(["Store", "Date"])
    df["Year"] = df["Date"].dt.year
    df["Month"] = df["Date"].dt.month
    df["Day"] = df["Date"].dt.day
    df["WeekOfYear"] = df["Date"].dt.isocalendar().week.astype(int)
    df["IsWeekend"] = df["DayOfWeek"].isin([6, 7]).astype(int)

    df["StoreType_enc"] = df["StoreType"].astype("category").cat.codes
    df["Assortment_enc"] = df["Assortment"].astype("category").cat.codes if "Assortment" in df.columns else 0
    df["StateHoliday_enc"] = df["StateHoliday"].astype("category").cat.codes

    if {"CompetitionOpenSinceYear", "CompetitionOpenSinceMonth"}.issubset(df.columns):
        df["CompetitionOpenMonths"] = (
            (df["Year"] - df["CompetitionOpenSinceYear"]) * 12
            + (df["Month"] - df["CompetitionOpenSinceMonth"])
        ).clip(lower=0)
    else:
        df["CompetitionOpenMonths"] = 0

    df["Sales_Lag7"] = df.groupby("Store")["Sales"].shift(7)
    df["Sales_Roll7_Mean"] = df.groupby("Store")["Sales"].transform(lambda s: s.shift(1).rolling(7).mean())
    df["Sales_Roll28_Mean"] = df.groupby("Store")["Sales"].transform(lambda s: s.shift(1).rolling(28).mean())
    df = df.dropna(subset=["Sales_Lag7", "Sales_Roll7_Mean", "Sales_Roll28_Mean"]).copy()
    return df


FEATURES = [
    "Store", "DayOfWeek", "Promo", "StateHoliday_enc", "SchoolHoliday",
    "StoreType_enc", "Assortment_enc", "CompetitionDistance", "CompetitionOpenMonths",
    "Promo2", "Year", "Month", "Day", "WeekOfYear", "IsWeekend",
    "Sales_Lag7", "Sales_Roll7_Mean", "Sales_Roll28_Mean",
]
TARGET = "Sales"


def rmspe(y_true, y_pred):
    mask = y_true != 0
    return float(np.sqrt(np.mean(((y_true[mask] - y_pred[mask]) / y_true[mask]) ** 2)))


@st.cache_resource
def train_model(df: pd.DataFrame):
    cutoff = df["Date"].max() - pd.Timedelta(days=42)
    train_df = df[df["Date"] <= cutoff]
    test_df = df[df["Date"] > cutoff]

    X_train, y_train = train_df[FEATURES], train_df[TARGET]
    X_test, y_test = test_df[FEATURES], test_df[TARGET]

    rf_model = RandomForestRegressor(n_estimators=150, max_depth=14, min_samples_leaf=3, random_state=42, n_jobs=-1)
    rf_model.fit(X_train, y_train)
    rf_preds = rf_model.predict(X_test)

    xgb_model = XGBRegressor(
        n_estimators=400, max_depth=6, learning_rate=0.05,
        subsample=0.85, colsample_bytree=0.85, random_state=42, n_jobs=-1,
    )
    xgb_model.fit(X_train, y_train)
    xgb_preds = xgb_model.predict(X_test)

    candidates = {
        "Random Forest": (rf_model, rf_preds),
        "XGBoost": (xgb_model, xgb_preds),
    }
    all_metrics = {
        name: {
            "MAE": mean_absolute_error(y_test, preds),
            "RMSE": np.sqrt(mean_squared_error(y_test, preds)),
            "RMSPE": rmspe(y_test.values, preds),
            "R2": r2_score(y_test, preds),
        }
        for name, (_, preds) in candidates.items()
    }
    best_name = min(all_metrics, key=lambda n: all_metrics[n]["RMSPE"])
    best_model, best_preds = candidates[best_name]

    return best_model, best_name, all_metrics[best_name], all_metrics, test_df.assign(Predicted=best_preds), cutoff


# ---------------------------------------------------------------------------
# Sidebar — data source + controls
# ---------------------------------------------------------------------------
st.sidebar.title("📈 Sales Forecasting")
st.sidebar.caption("Rossmann Retail Case Study — live model")

data_mode = st.sidebar.radio("Data source", ["Demo dataset (Rossmann)", "Upload my own"])

if data_mode == "Upload my own":
    train_file = st.sidebar.file_uploader("train.csv (Store, Date, Sales, Promo, ...)", type="csv")
    store_file = st.sidebar.file_uploader("store.csv (Store, StoreType, CompetitionDistance, ...)", type="csv")
    if train_file and store_file:
        train_raw = pd.read_csv(train_file, parse_dates=["Date"], low_memory=False)
        store_raw = pd.read_csv(store_file)
        st.sidebar.success("Your data is loaded ✅")
    else:
        st.sidebar.info("Upload both files to use your own data. Showing demo data meanwhile.")
        train_raw, store_raw = load_default()
else:
    train_raw, store_raw = load_default()
    st.sidebar.info("Showing the public Rossmann demo dataset (100 sampled stores).")

n_sample = st.sidebar.slider("Stores to include (for speed)", 20, 300, 100, step=20)

with st.spinner("Preparing data and training models (Random Forest + XGBoost)..."):
    df = prepare(train_raw, store_raw, n_sample)
    model, best_model_name, metrics, all_metrics, test_df, cutoff = train_model(df)

st.sidebar.markdown("---")
st.sidebar.caption(
    "[View the full notebook & methodology on GitHub]"
    "(https://github.com/NurMithu)"
)

# ---------------------------------------------------------------------------
# Header + KPIs
# ---------------------------------------------------------------------------
st.title("Retail Sales Forecasting Dashboard")
st.subheader(f"Rossmann Store Sales — Live Model (best: {best_model_name})")

k1, k2, k3, k4 = st.columns(4)
k1.metric("Stores in model", f"{df['Store'].nunique():,}")
k2.metric("Forecast Accuracy (R²)", f"{metrics['R2']:.3f}")
k3.metric("RMSPE (holdout)", f"{metrics['RMSPE']:.1%}")
k4.metric("Avg. Error (MAE)", f"€{metrics['MAE']:,.0f}")

with st.expander("Model comparison (Random Forest vs. XGBoost)"):
    comp_df = pd.DataFrame(all_metrics).T
    comp_df.index.name = "Model"
    st.dataframe(
        comp_df.style.format({"MAE": "€{:.0f}", "RMSE": "€{:.0f}", "RMSPE": "{:.1%}", "R2": "{:.3f}"})
        .highlight_min(subset=["RMSPE"], color="#D1FAE5"),
        use_container_width=True,
    )

st.markdown("---")

tab_overview, tab_forecast, tab_promo, tab_reco = st.tabs(
    ["📊 Overview", "🔮 Store Forecast", "🎯 Promotion Impact", "📋 Recommendations"]
)

# ---------------------------------------------------------------------------
with tab_overview:
    c1, c2 = st.columns(2)
    with c1:
        fig = px.histogram(df, x="Sales", nbins=60, title="Sales Distribution (all stores)",
                            color_discrete_sequence=[PRIMARY])
        st.plotly_chart(fig, use_container_width=True)
    with c2:
        dow_sales = df.groupby("DayOfWeek")["Sales"].mean().reset_index()
        fig = px.bar(dow_sales, x="DayOfWeek", y="Sales", title="Avg. Sales by Day of Week",
                     color_discrete_sequence=[PRIMARY])
        st.plotly_chart(fig, use_container_width=True)

    if "StoreType" in df.columns:
        st_sales = df.groupby("StoreType")["Sales"].mean().sort_values(ascending=False).reset_index()
        fig = px.bar(st_sales, x="StoreType", y="Sales", title="Avg. Sales by Store Type",
                     color_discrete_sequence=[PRIMARY])
        st.plotly_chart(fig, use_container_width=True)

    st.markdown("**Model feature importance**")
    importance = pd.Series(model.feature_importances_, index=FEATURES).sort_values(ascending=False).head(10)
    fig = px.bar(importance.sort_values(), orientation="h", title="Top 10 Sales Drivers",
                 color_discrete_sequence=[PRIMARY])
    fig.update_layout(showlegend=False, yaxis_title="", xaxis_title="Importance")
    st.plotly_chart(fig, use_container_width=True)

# ---------------------------------------------------------------------------
with tab_forecast:
    st.markdown("#### Forecast vs. actual for an individual store")
    store_ids = sorted(df["Store"].unique())
    selected_store = st.selectbox("Choose a store", store_ids)

    s_hist = df[df["Store"] == selected_store].sort_values("Date")
    s_test = test_df[test_df["Store"] == selected_store].sort_values("Date")

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=s_hist["Date"], y=s_hist["Sales"], name="Actual (history)",
                              line=dict(color=GRAY, width=1)))
    if len(s_test):
        fig.add_trace(go.Scatter(x=s_test["Date"], y=s_test["Sales"], name="Actual (holdout)",
                                  line=dict(color=PRIMARY, width=2)))
        fig.add_trace(go.Scatter(x=s_test["Date"], y=s_test["Predicted"], name="Model Forecast",
                                  line=dict(color=GREEN, width=2, dash="dot")))
    fig.update_layout(title=f"Store {selected_store} — Actual vs. Forecast (last 6 weeks)",
                       yaxis_title="Sales (€)")
    st.plotly_chart(fig, use_container_width=True)

    if len(s_test):
        store_mae = mean_absolute_error(s_test["Sales"], s_test["Predicted"])
        c1, c2, c3 = st.columns(3)
        c1.metric("Store Avg. Daily Sales", f"€{s_hist['Sales'].mean():,.0f}")
        c2.metric("Forecast Error (MAE)", f"€{store_mae:,.0f}")
        c3.metric("Promo Days (holdout)", int(s_test["Promo"].sum()))

    st.download_button(
        "⬇️ Download this store's forecast (CSV)",
        s_test[["Date", "Sales", "Predicted", "Promo"]].to_csv(index=False).encode("utf-8"),
        f"store_{selected_store}_forecast.csv", "text/csv",
    )

# ---------------------------------------------------------------------------
with tab_promo:
    promo_impact = df.groupby("Promo")["Sales"].mean()
    promo_lift = (promo_impact.get(1, 0) / promo_impact.get(0, 1) - 1) * 100 if promo_impact.get(0, 0) else 0

    c1, c2 = st.columns([1, 2])
    with c1:
        st.metric("Average Sales Lift from Promotions", f"+{promo_lift:.1f}%")
        st.caption("Computed on days with vs. without an active promotion, across all stores in scope.")
    with c2:
        fig = px.bar(
            x=["No Promo", "Promo"], y=[promo_impact.get(0, 0), promo_impact.get(1, 0)],
            title="Average Daily Sales: Promo vs. No Promo",
            color=["No Promo", "Promo"], color_discrete_map={"No Promo": GRAY, "Promo": GREEN},
        )
        fig.update_layout(showlegend=False, yaxis_title="Avg. Sales (€)", xaxis_title="")
        st.plotly_chart(fig, use_container_width=True)

    if "StoreType" in df.columns:
        promo_by_type = df.groupby(["StoreType", "Promo"])["Sales"].mean().reset_index()
        promo_by_type["Promo"] = promo_by_type["Promo"].map({0: "No Promo", 1: "Promo"})
        fig = px.bar(promo_by_type, x="StoreType", y="Sales", color="Promo", barmode="group",
                     title="Promotion Lift by Store Type", color_discrete_map={"No Promo": GRAY, "Promo": GREEN})
        st.plotly_chart(fig, use_container_width=True)

# ---------------------------------------------------------------------------
with tab_reco:
    st.markdown("#### Business recommendations")
    st.markdown(
        f"""
- **Deploy the {best_model_name} model as a weekly per-store forecasting routine** — it outperformed the alternative approach tested here (see the model comparison above), rather than being chosen by default. It explains
  **{metrics['R2']:.0%}** of sales variance on unseen data with a typical
  error of **€{metrics['MAE']:,.0f}/day** — accurate enough to drive
  inventory and staffing decisions, not just directional planning.
- **Promotions are the largest controllable lever**, delivering roughly
  **+{promo_lift:.0f}%** average sales lift — worth analyzing at the
  store/period level to target promo spend where it converts best, rather
  than running blanket promotions.
- **Recent momentum (7/28-day rolling average sales) is the strongest single
  predictor** — forecasts should be refreshed weekly, not left static for a
  quarter, since each store's own recent trend carries more signal than
  calendar effects alone.
- **Use the per-store forecast tab to flag stores tracking meaningfully below
  model prediction** — an early operational warning sign worth a manual
  review, similar in spirit to the churn-risk flagging in the CLV dashboard.
"""
    )
    st.link_button("📓 View the full notebook & methodology on GitHub", "https://github.com/NurMithu")

st.markdown("---")
st.caption(
    "Live model trained on the public Rossmann Store Sales dataset. "
    "Upload your own train.csv/store.csv in the sidebar to see results on your own retail data."
)
