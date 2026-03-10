# 🚕 NYC Taxi Trip Duration & Late-Risk Prediction

> Predicting airport taxi trip durations and classifying late-risk for Manhattan → JFK/LGA rides using ~50K cleaned trip records from the NYC TLC dataset.

**Tech:** Python · scikit-learn · XGBoost · CatBoost · Random Forest · pandas · Matplotlib · Seaborn

---

## TL;DR

- Predicted NYC airport taxi trip duration using tree-based regression (**MAE ≈ 5.6 min, R² ≈ 0.78**)
- Modeled late-risk as an imbalanced classification problem (**ROC AUC ≈ 0.73**)
- Used time-based train/test splits to avoid data leakage
- Tuned probability thresholds for traveler vs. operations use cases
- Translated predictions into buffer-time guidance ("Airport Taxi Timing Advisor")

---

## The Problem

Airport travel in NYC is a classic *planning under uncertainty* problem. A single ETA prediction hides the **late-risk tail** of unusually slow trips. This project answers two practical questions for Manhattan → JFK/LGA taxi rides:

1. **How long will the trip take?** → Regression
2. **What is the probability the trip will be "late"?** → Classification

The goal is converting historical taxi data into **decision-ready departure-time guidance** that adapts to different risk preferences.

---

## Key Results

| Task | Best Model | Performance |
|------|-----------|-------------|
| **Duration Prediction** | Random Forest | MAE ≈ 5.6 min, R² ≈ 0.78 |
| **Late-Trip Classification** | CatBoost (tuned threshold) | ROC AUC ≈ 0.73 |
| **Baseline Comparison** | Historical Median | MAE ≈ 13 min (regression), F1 = 0.0 (classification) |

---

## Data & Feature Engineering

**Source:** [NYC Yellow Taxi Trip Records (Aug 2025)](https://www.nyc.gov/site/tlc/about/tlc-trip-record-data.page) — filtered from ~886K trips to ~50K clean airport rides.

All features limited to **pre-pickup information only** to prevent leakage:

| Used ✅ | Excluded ❌ |
|---------|------------|
| Pickup zone / neighborhood | Fare totals |
| Destination airport (JFK vs LGA) | Tolls |
| Pickup hour & day of week | Any post-trip variables |
| Trip distance, payment type | |
| Rush-hour indicator | |

**Train/Test Split:** Chronological 80/20 — mimics real deployment (learn from past, predict future).

---

## Regression: Trip Duration

| Model | Test MAE | Test R² | Notes |
|-------|----------|---------|-------|
| Linear Regression | ~6.6 min | ~0.68 | Misses nonlinear traffic effects |
| XGBoost (tuned) | ~5.6 min | ~0.77 | High accuracy, slightly more variance |
| **Random Forest** | **~5.6 min** | **~0.78** | **Best stability + accuracy + interpretability** |

**Key duration drivers:** Hour of day (rush-hour), airport destination (JFK longer & more variable), trip distance, pickup neighborhood.

---

## Classification: Late-Risk

**Late = trip duration > 120% of the historical median** for similar trips (zone × airport × hour × weekday). Only ~18% of trips are late → significant class imbalance.

| Model | ROC AUC | Late-Class F1 | Notes |
|-------|---------|---------------|-------|
| Always On-Time Baseline | — | 0.00 | 82% accuracy but useless |
| Logistic Regression | ~0.68 | — | Reasonable but underperforms |
| Random Forest | ~0.71 | — | Better nonlinear capture |
| **CatBoost** | **~0.73** | **Best** | **Best precision-recall balance** |

### Threshold Tuning

Same model, different thresholds, different user experience:

| Use Case | Threshold | Recall | Trade-off |
|----------|-----------|--------|-----------|
| **Traveler** (risk-averse) | 0.40 | ~0.83 | Catches most late trips — "better safe than sorry" |
| **Operations** (precision) | 0.50 | ~0.66 | Fewer false alarms, cleaner reporting |

---

## Buffer-Time Advisor

Translating predictions into actionable guidance for weekday Manhattan → JFK:

| Buffer Added | Late Risk |
|-------------|-----------|
| +0 min | ~50% |
| +10–15 min | ~25% (roughly halved) |
| +20–25 min | Low single digits |

> *Given a pickup zone, airport, and time → estimate trip duration and recommend a buffer based on your acceptable late risk.*

---

## Repository Structure

```
├── taxi_EDA.ipynb              # Exploratory data analysis & visualization
├── taxi_Regression.ipynb       # Duration prediction (Linear, XGBoost, Random Forest)
├── taxi_classification.ipynb   # Late-risk classification (Logistic Reg, RF, CatBoost)
└── README.md
```

---

## How to Run

```bash
git clone https://github.com/awhite121/NYC-Taxi-Trip-Duration-Late-Risk-Prediction.git
cd NYC-Taxi-Trip-Duration-Late-Risk-Prediction
pip install pandas numpy scikit-learn xgboost catboost matplotlib seaborn
jupyter notebook
```

---

## Limitations & Next Steps

**Limitations:** Single month of data (no seasonality), Manhattan → JFK/LGA only, no weather/events/live traffic, offline modeling.

**Next steps:** Multi-month/year extension, weather & traffic feeds, probability calibration, live Timing Advisor deployment.

---

## Author

Andrew White — [GitHub](https://github.com/awhite121)
*MSBA coursework — Advanced Machine Learning & Regression Modeling, University of Texas at Austin*
