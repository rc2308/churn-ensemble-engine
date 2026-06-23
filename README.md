# Behavioral Segmentation & Churn Prediction Engine for Bank Customers

Clustering → XGBoost + LightGBM Ensemble → SHAP → In-Notebook Dual-Mode Dashboard
**Fully implemented in a single Google Colab notebook.**

## Overview
An end-to-end churn analytics pipeline on the `Churn_Modelling.csv` dataset
(10,000 labeled rows, binary target `Exited`). It combines unsupervised customer
segmentation, a weighted XGBoost + LightGBM ensemble, SHAP explainability, and an
in-Colab dual-mode dashboard. The headline metric is the **Gini coefficient**,
supported by AUC, PR-AUC, Recall, and F1.

## Architecture (two phases, one notebook)
- **Phase A — Development:** Data Cleaning → EDA → Feature Engineering → K-Means
  Clustering → XGBoost + LightGBM Ensemble → SHAP → Artifact Export.
- **Phase B — Dashboard:** loads exported artifacts and serves a dual-mode Gradio app
  (predict-only, or predict + evaluate if labels are present).

## Repo Structure
