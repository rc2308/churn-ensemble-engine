# Behavioral Segmentation & Churn Prediction Engine

Clustering → XGBoost + LightGBM Ensemble → SHAP → In-Notebook Dual-Mode Dashboard.
Built entirely inside a single Google Colab notebook on the `Churn_Modelling.csv`
dataset (10,000 labeled rows, binary target `Exited`).

## Overview

This project replicates the machine-learning lifecycle used by financial institutions
(American Express / Visa / Mastercard): train on labeled history, validate with the
**Gini coefficient**, score current customers, and explain decisions with **SHAP**.

**Pipeline (Phase A — Development):**
1. Data cleaning (drop `RowNumber`/`CustomerId`/`Surname`, encode target)
2. EDA (churn distribution, correlations, behavioral patterns)
3. Feature engineering (`HasBalance`, `BalanceSalaryRatio`, `TenureByAge`; one-hot Geography/Gender)
4. Train/test split (80/20, stratified) + scaling (fit on train only)
5. **SMOTE** oversampling on the scaled training set (handles the ~20% class imbalance)
6. K-Means segmentation (K=4) → named personas, appended as a model feature
7. XGBoost (Optuna-tuned) + LightGBM → **weighted soft-voting ensemble**
8. Auto-optimized decision threshold (F2-score) + auto-optimized ensemble blend weight
9. Evaluation (Gini / AUC / PR-AUC / Recall / F1) + SHAP (global + per-segment)
10. Artifact export

**Phase B — Dual-Mode Dashboard (Gradio):**
- **Predict mode** (no labels): churn probability, persona, risk badge, per-prediction SHAP
- **Evaluate mode** (labels present): predictions + Gini / AUC / confusion matrix
- Two tabs: batch CSV upload (with downloadable scored results) and single-customer form
- Customer identifiers (`CustomerId`/`Surname`) are preserved in the output for actionability

## Customer Personas

| Cluster | Persona | Avg Balance | Churn Rate | Description |
|---|---|---|---|---|
| 0 | At-Risk Wealthy | 118,875 | 69% | High balance, highest churn — retention priority |
| 1 | Low-Balance Mass | 113 | 38% | Near-zero balance, most products |
| 2 | Wealthy Mid-Risk | 121,854 | 50% | High balance, moderate churn |
| 3 | Affluent Watchlist | 119,593 | 49% | High balance, elevated churn |

> Personas were re-labeled to match the actual post-SMOTE cluster churn rates.

## Results (labeled 20% hold-out)

| Model | Gini | AUC | PR-AUC | Recall | Precision | F1 |
|---|---|---|---|---|---|---|
| XGBoost  | 0.7306 | 0.8653 | 0.7180 | 0.8477 | 0.3966 | 0.5403 |
| LightGBM | 0.7334 | 0.8667 | 0.7212 | 0.8108 | 0.4577 | 0.5851 |
| Ensemble | 0.7339 | 0.8669 | 0.7215 | 0.8182 | 0.4464 | 0.5776 |

- **Ensemble weight (XGB):** 0.15
- **Decision threshold (F2-optimized):** 0.25
- **Headline Gini: 0.73** — exceeds the project target of ≳ 0.50.

The threshold is tuned for **recall (F2-score)**: in churn retention, missing a customer
who will leave (false negative) costs more than a wasted retention offer (false positive).

## Repository Structure


## How to Run

1. Open `notebooks/churn_ensemble_engine.ipynb` in Google Colab.
2. Run the install cells, then mount Drive and upload `Churn_Modelling.csv`.
3. Run Phase A cells top-to-bottom (training + artifact export).
4. Run Phase B cells to launch the Gradio dashboard (in-Colab public URL).

## Tech Stack

Python · pandas · NumPy · scikit-learn · XGBoost · LightGBM · imbalanced-learn (SMOTE) ·
Optuna · SHAP · Gradio

## License

`<add license, e.g. MIT>`
