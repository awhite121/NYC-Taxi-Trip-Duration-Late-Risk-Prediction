# NYC Taxi Trip Duration & Late Risk Prediction

## TL;DR
- Predicted NYC airport taxi trip duration using tree-based regression (MAE ≈ 5–6 minutes)
- Modeled late-risk as an imbalanced classification problem
- Used time-based train/test splits to avoid data leakage
- Tuned probability thresholds for traveler vs. operations use cases
- Translated predictions into buffer-time guidance (“Airport Taxi Timing Advisor”)

---

## Project Overview

Airport travel in New York City is a classic *planning under uncertainty* problem. Leaving at the “typical” time might be fine — or a single traffic spike can turn it into a missed flight. Single ETA predictions hide this uncertainty by ignoring the **late-risk tail** of unusually slow trips.

This project answers two practical questions for Manhattan → JFK/LGA taxi rides:

1. **How long will the trip take?** (regression)
2. **What is the probability the trip will be “late” relative to typical conditions?** (classification)

The goal is not just strong predictive accuracy, but converting historical taxi data into **decision-ready departure-time guidance** that adapts to different risk preferences (traveler vs. operations).

---

## Data Sources

- **NYC Yellow Taxi Trip Records (August 2025)**  
  Trip-level timestamps, distances, payment types, pickup/drop-off zone IDs  
- **NYC Taxi Zone Lookup**  
  Maps zone IDs to Manhattan neighborhoods and boroughs

After filtering to Manhattan → JFK/LGA trips and removing implausible records, the dataset was reduced from ~886K trips to **~50K clean airport rides**.

---

## Feature Engineering (Pre-Trip Only)

All features were limited to information available **before pickup** to avoid leakage.

**Features used**
- Pickup zone / neighborhood
- Destination airport (JFK vs LGA)
- Pickup hour and day of week
- Trip distance
- Payment type
- Rush-hour indicator

**Intentionally excluded**
Fare totals, tolls, and other post-trip variables that are direct functions of trip duration.

---

## Train/Test Strategy & Baseline

### Time-Based Split
Trips were sorted chronologically:
- Earliest 80% → training
- Most recent 20% → test

This mimics real deployment: learning from past trips to predict future ones.

### Strong Baseline
A non-ML benchmark predicted duration using the **historical median** for each:
> pickup zone × airport × pickup hour

- **Baseline MAE ≈ 13 minutes**
- Already captures geography and time effects
- Sets a meaningful bar for ML improvement

---

## Regression Modeling (Trip Duration)

Models evaluated:
- Linear Regression
- XGBoost Regressor
- **Random Forest Regressor (final model)**

| Model | Test MAE | Test R² | Notes |
|------|---------|--------|------|
| Linear Regression | ~6.6 min | ~0.68 | Strong improvement but misses nonlinear traffic effects |
| XGBoost (tuned) | ~5.6 min | ~0.77 | High accuracy with slightly more variance |
| **Random Forest** | **~5.6 min** | **~0.78** | Best balance of stability, accuracy, and interpretability |

**Why Random Forest**
- Minimal train–test gap (low overfitting)
- Smooth, interpretable feature importance
- Strong generalization across pickup zones

**Key drivers of trip duration**
- Hour of day (rush-hour effects)
- Airport destination (JFK longer and more variable)
- Trip distance
- Pickup neighborhood and weekday effects

---

## Classification Modeling (Late Risk)

### Late Label Definition
A trip is labeled **late** if its duration exceeds **120% of the historical median** for similar trips  
(pickup zone × airport × hour × weekday).

Only ~18% of trips are late → **class imbalance** is substantial.

### Baseline
An “always on-time” classifier:
- Accuracy ≈ 82%
- **F1 (late class) = 0.0**

This shows why accuracy alone is misleading.

### Models Evaluated
- Logistic Regression
- Random Forest Classifier
- **CatBoost Classifier (final model)**

CatBoost achieved the best balance across:
- ROC-AUC (~0.73)
- Precision–Recall curve
- Late-class recall and F1

---

## Threshold Tuning (Decision Framing)

Instead of fixing the cutoff at 0.50, probability thresholds were tuned by use case:

**Traveler-focused (risk-averse)**
- Threshold = 0.40
- Recall ≈ 0.83
- Catches most late trips (“better safe than sorry”)

**Operations / analytics**
- Threshold = 0.50
- Precision ≈ 0.30, Recall ≈ 0.66
- Fewer false alarms, cleaner reporting

Same model — different threshold — different user experience.

---

## Buffer-Time Guidance (“Timing Advisor”)

To translate predictions into actionable advice, buffer curves were constructed for weekday Manhattan → JFK trips:

- 0-minute buffer → ~50% late risk
- +10–15 minutes → late risk roughly halves
- +20–25 minutes → late risk drops to low single digits

This supports an **Airport Taxi Timing Advisor** concept:
> Given pickup zone, airport, and time, estimate trip duration and recommend a buffer based on acceptable late risk.

---

## Limitations & Next Steps

**Limitations**
- Single month of data (no seasonality or holidays)
- Only Manhattan → JFK/LGA
- No weather, events, or live traffic features
- Offline modeling (no production system)

**Next steps**
- Extend to multiple months/years and additional airports
- Incorporate weather and traffic feeds
- Calibrate predicted probabilities
- Deploy a live version of the Timing Advisor

---

## Takeaway

This project demonstrates an end-to-end applied ML workflow:
- Realistic baselines and leakage-aware modeling
- Regression + classification with proper evaluation
- Imbalanced learning and threshold tuning
- Translation of predictions into real decision support
