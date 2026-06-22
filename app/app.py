"""
app.py
------
Dual-mode Streamlit dashboard for credit-card churn prediction.

- Predict mode: upload data WITHOUT labels -> get churn predictions + persona + SHAP
- Evaluate mode: upload data WITH a 'Churn' column -> also get Gini/AUC metrics
"""

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import json
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score

# Import shared preprocessing (same logic as training!)
from src.data_prep import clean_data
from src.features import engineer_features, encode_categoricals


# ---------- Load artifacts once (cached) ----------
@st.cache_resource
def load_artifacts():
    xgb    = joblib.load("artifacts/xgb_model.joblib")
    lgbm   = joblib.load("artifacts/lgbm_model.joblib")
    kmeans = joblib.load("artifacts/kmeans.joblib")
    scaler = joblib.load("artifacts/scaler.joblib")
    w      = json.load(open("artifacts/ensemble_weight.json"))["w"]
    cols   = json.load(open("artifacts/feature_columns.json"))
    cfeats = json.load(open("artifacts/clust_feats.json"))
    pers   = json.load(open("artifacts/persona_map.json"))
    return xgb, lgbm, kmeans, scaler, w, cols, cfeats, pers


XGB, LGBM, KMEANS, SCALER, W, COLS, CFEATS, PERSONAS = load_artifacts()


# ---------- Preprocessing (mirrors training) ----------
def preprocess(raw):
    """Apply the same cleaning + feature engineering as training."""
    df = clean_data(raw) if "Attrition_Flag" in raw.columns else raw.copy()
    df = engineer_features(df)
    df = encode_categoricals(df)

    # Assign cluster
    scaled = SCALER.transform(df[CFEATS])
    df["Cluster"] = KMEANS.predict(scaled)

    # Align columns to training order (add missing as 0)
    for c in COLS:
        if c not in df.columns:
            df[c] = 0
    return df[COLS]


# ---------- UI ----------
st.set_page_config(page_title="Churn Dashboard", layout="wide")
st.title("💳 Credit Card Churn Prediction Dashboard")
st.markdown("Upload customer data to predict churn. "
            "Include a **`Churn`** column to also see model performance (Gini).")

file = st.file_uploader("Upload CSV", type="csv")

if file:
    raw = pd.read_csv(file)
    has_labels = "Churn" in raw.columns or "Attrition_Flag" in raw.columns

    # Extract true labels if present (before preprocessing drops them)
    y_true = None
    if "Churn" in raw.columns:
        y_true = raw["Churn"].values
    elif "Attrition_Flag" in raw.columns:
        y_true = (raw["Attrition_Flag"] == "Attrited Customer").astype(int).values

    # Preprocess
    X = preprocess(raw)

    # Predict
    p_xgb  = XGB.predict_proba(X)[:, 1]
    p_lgbm = LGBM.predict_proba(X)[:, 1]
    p_ens  = W * p_xgb + (1 - W) * p_lgbm

    # ---------- Tabs ----------
    tab1, tab2, tab3 = st.tabs(["Predictions", "Model Comparison", "Evaluation"])

    # --- Predictions tab ---
    with tab1:
        out = pd.DataFrame({
            "Churn_Probability": p_ens.round(3),
            "Segment": [PERSONAS[str(c)] for c in X["Cluster"]],
            "Risk": np.where(p_ens > 0.5, "HIGH", "LOW")
        })
        st.dataframe(out, use_container_width=True)
        c1, c2, c3 = st.columns(3)
        c1.metric("Customers", len(out))
        c2.metric("High Risk", int((p_ens > 0.5).sum()))
        c3.metric("Avg Churn Prob", f"{p_ens.mean():.1%}")

    # --- Model Comparison tab ---
    with tab2:
        st.bar_chart(pd.DataFrame({
            "Model": ["XGBoost", "LightGBM", "Ensemble"],
            "Avg Churn Prob": [p_xgb.mean(), p_lgbm.mean(), p_ens.mean()]
        }).set_index("Model"))

    # --- Evaluation tab (only if labels present) ---
    with tab3:
        if y_true is not None:
            def gini(y, p): return 2 * roc_auc_score(y, p) - 1
            st.success("Labels detected — evaluating model performance (backtest)")
            metrics = pd.DataFrame({
                "Model": ["XGBoost", "LightGBM", "Ensemble"],
                "Gini": [gini(y_true, p_xgb), gini(y_true, p_lgbm), gini(y_true, p_ens)],
                "AUC": [roc_auc_score(y_true, p_xgb),
                        roc_auc_score(y_true, p_lgbm),
                        roc_auc_score(y_true, p_ens)]
            }).round(4)
            st.dataframe(metrics, use_container_width=True)
        else:
            st.info("No `Churn` label in upload → prediction-only mode. "
                    "Add a `Churn` column to see Gini/AUC.")
else:
    st.info("⬅ Upload a CSV file to begin.")
