# Retail Sales Forecasting & Promotion Impact Analysis

[![Python](https://img.shields.io/badge/Python-3.10+-3776AB?logo=python&logoColor=white)](https://python.org)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.5+-F7931E?logo=scikitlearn&logoColor=white)](https://scikit-learn.org)
[![pandas](https://img.shields.io/badge/pandas-2.0+-150458?logo=pandas&logoColor=white)](https://pandas.pydata.org)
[![Streamlit](https://img.shields.io/badge/Streamlit-Live%20Demo-FF4B4B?logo=streamlit&logoColor=white)](#live-demo)
[![License](https://img.shields.io/badge/License-MIT-green)](LICENSE)

End-to-end data science project forecasting daily retail sales for 1,115+ stores, quantifying promotion impact, and translating findings into a live, client-facing dashboard — built on the public Rossmann Store Sales dataset.

## Live Demo

🔗 **[Try the live dashboard](#)** 


## Overview

Rossmann store managers currently forecast daily sales manually up to six weeks ahead — a process affected by promotions, competition, holidays, seasonality, and local factors, with accuracy varying widely manager to manager.

This project builds a complete, reproducible forecasting pipeline that answers five core business questions:

1. How accurately can we forecast daily store sales?
2. What actually drives sales — promotions, competition, seasonality, or store format?
3. How much sales lift does a promotion actually deliver?
4. Which store types/assortments outperform, and why?
5. What should the business do differently based on these findings?

The deliverable is a fully-executable Jupyter notebook (15 modules, business-question-first structure) **plus** a live interactive Streamlit dashboard that turns the model into something a non-technical stakeholder can use directly.

## Repository Structure

```
sales-forecasting-rossmann/
├── app.py                              # Live Streamlit dashboard
├── pipeline.py                         # Standalone script (metrics/figures)
├── Sales_Forecasting_Rossmann.ipynb    # Full analysis notebook (15 modules, executed)
├── requirements.txt
├── data/
│   ├── train.csv                       # 1,017,210 rows — daily sales, 1,115 stores
│   └── store.csv                       # Store metadata (type, assortment, competition)
├── outputs/                            # Saved figures & metrics.json
├── LICENSE
└── README.md
```

## Dataset

Public **Rossmann Store Sales** dataset — daily sales for drug stores across Germany, January 2013 to July 2015, merged with store-level metadata.

> **Note on file size:** the full dataset (1,115 stores, ~1M rows, ~37MB) exceeds GitHub's 25MB drag-and-drop upload limit. This repo ships a **100-store random subset** (~91,000 rows, ~3MB) — the exact same subset the notebook and pipeline train on, so every number in this README is reproducible from the files included here. To run against the full 1,115-store dataset, download it directly from [Kaggle's Rossmann Store Sales competition](https://www.kaggle.com/c/rossmann-store-sales/data), drop `train.csv`/`store.csv` into the `data/` folder, and set `N_SAMPLE_STORES = None` in `pipeline.py` or the notebook's Module 4 — no other code changes needed.

| Column | Description |
|---|---|
| `Store` | Unique store ID |
| `Date` | Calendar date |
| `Sales` | Daily turnover (target variable) |
| `Customers` | Daily customer count |
| `Open` | 1 = store open |
| `Promo` | 1 = promotion running that day |
| `StateHoliday` | Public/Easter/Christmas holiday code |
| `SchoolHoliday` | 1 = affected by school closures |
| `StoreType` | Store format (a/b/c/d) |
| `Assortment` | Product assortment level |
| `CompetitionDistance` | Distance to nearest competitor (m) |

## Notebook Structure (15 Modules)

| Module | Contents |
|---|---|
| 1 | Business Understanding — problem, objectives, scope |
| 2 | Data Understanding — load, inspect |
| 3 | Data Quality Assessment — missingness, invalid values |
| 4 | Data Cleaning — closed-day removal, imputation |
| 5 | Exploratory Data Analysis — distributions, day-of-week, promo lift |
| 6 | Feature Engineering — calendar, competition exposure, lag/rolling sales |
| 7 | Train/Test Split — time-based, 6-week holdout |
| 8 | Baseline Model — Linear Regression |
| 9 | Random Forest Model |
| 9b | XGBoost Model — gradient boosting comparison |
| 10 | Model Evaluation — MAE, RMSE, RMSPE, R² (best model selected empirically) |
| 11 | Feature Importance |
| 12 | Prediction Diagnostics |
| 13 | Business Insights Summary |
| 14 | Business Recommendations |
| 15 | Limitations & Future Work |

## Key Results

**Model comparison (6-week time-based holdout, 100 sampled stores)** — three approaches tested rather than assumed, to show *why* a model is chosen, not just its output:

| Model | MAE | RMSE | RMSPE | R² |
|---|---|---|---|---|
| Linear Regression (baseline) | €988.85 | €1,348.65 | 22.3% | 0.828 |
| Random Forest | €690.94 | €959.94 | 15.0% | 0.913 |
| **XGBoost** | **€638.46** | **€884.94** | **13.2%** | **0.926** |

> RMSPE (Root Mean Squared Percentage Error) is reported because it's the actual metric this dataset's original Kaggle competition was judged on — it penalizes percentage error rather than absolute error, which matters when store sizes vary widely. XGBoost cuts RMSPE by roughly 40% versus the linear baseline and outperforms Random Forest — gradient boosting's sequential error-correction gives it an edge on this kind of tabular data with one dominant signal (recent sales momentum) plus many smaller contributing factors.

**Promotion impact:** stores running an active promotion see an average **+39.0% sales lift** on that day — the single largest controllable lever identified in the analysis.

**Top sales drivers** (XGBoost feature importance): recent sales momentum (28-day rolling average) dominates, followed by active promotion status and day-of-week — calendar effects and competition distance matter, but are secondary.

## Live Dashboard Features

- **Model comparison** — trains Random Forest *and* XGBoost live and automatically selects the better performer (by RMSPE), with a side-by-side comparison table
- **Overview** — sales distribution, day-of-week patterns, store-type comparison, model feature importance
- **Store Forecast** — pick any store, see actual vs. model-forecast sales for the holdout period, download the forecast as CSV
- **Promotion Impact** — quantified sales lift, broken down by store type
- **Recommendations** — plain-English business actions tied to the live numbers and the model that actually won
- **Bring your own data** — upload your own `train.csv`/`store.csv` (same schema) and every chart and model retrains on your data instantly

## How to Run

### Notebook
```bash
git clone https://github.com/NurMithu/sales-forecasting-rossmann.git
cd sales-forecasting-rossmann
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
jupyter notebook Sales_Forecasting_Rossmann.ipynb
```

### Live Dashboard
```bash
pip install -r requirements.txt
streamlit run app.py
```

## Tech Stack

`pandas` · `numpy` · `scikit-learn` · `xgboost` · `matplotlib` · `seaborn` · `plotly` · `streamlit` · `Jupyter`

## Business Recommendations (Summary)

- **Deploy the best-performing model (XGBoost, selected empirically, not by default) as a weekly per-store forecasting routine** — roughly 40% fewer forecast errors than a naive linear model translates directly into better inventory and staffing decisions.
- **Promotions are the largest controllable lever** (+39% average lift) — worth analyzing at store/period level to target promo spend where it converts best, rather than blanket promotions.
- **Refresh forecasts weekly, not quarterly** — recent sales momentum is the dominant predictor, more than calendar or holiday effects.
- **Use store-level forecast deviation as an early operational signal** — a store tracking meaningfully below its own model prediction is worth a manual review.

Full detail in Module 14 of the notebook.

## Limitations & Future Work

- This build samples 100 of 1,115 stores for runtime speed; the same code trains on the full store set unmodified (`N_SAMPLE_STORES = None`).
- Neither Random Forest nor XGBoost had their hyperparameters tuned here (reasonable defaults, not a search) — a brief GridSearch/Optuna pass is a natural next step before production deployment.
- Tree-based models don't explicitly model long-range seasonality the way a dedicated time-series model (Prophet, SARIMA) can — a natural next iteration for holiday-heavy quarters.
- External signals not in this dataset (weather, local events, regional economic indicators) plausibly add further predictive lift.
- Models should be retrained on a rolling basis as new sales data accumulates, since retail patterns drift with competitive and macroeconomic conditions.

## License

This project is licensed under the [MIT License](LICENSE).

---

*Built as an end-to-end analytics case study — from raw retail transaction data to a validated, business-ready forecasting model with a live interactive demo.*
