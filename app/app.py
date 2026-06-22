"""
app.py
------
Dual-mode Streamlit dashboard for credit-card churn prediction.

- Predict mode: upload data WITHOUT labels -> get churn predictions + persona + SHAP
- Evaluate mode: upload data WITH a 'Churn' column -> also get Gini/AUC metrics
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import json
from sklearn.metrics import roc_auc_score

from src.data_prep import clean_data, to_churn_target  # <-- this must come AFTER sys.path
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
    df = clean_data(raw)
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


def extract_true_labels(raw):
    """Return churn labels when an upload includes a supported target column."""
    if "Churn" in raw.columns:
        return to_churn_target(raw["Churn"]).values

    for label_col in ("Attrition_Flag", "True_Label"):
        if label_col in raw.columns:
            return to_churn_target(raw[label_col]).values

    return None


def build_prediction_output(raw, p_ens, segments):
    """Keep identifiers as metadata in exports without using them as features."""
    out = pd.DataFrame(index=raw.index)

    if "CLIENTNUM" in raw.columns:
        out["CLIENTNUM"] = raw["CLIENTNUM"]

    for label_col in ("Attrition_Flag", "True_Label"):
        if label_col in raw.columns:
            out["True_Label"] = raw[label_col]
            break
    else:
        if "Churn" in raw.columns:
            churn = to_churn_target(raw["Churn"])
            out["True_Label"] = np.where(
                churn == 1,
                "Attrited Customer",
                "Existing Customer",
            )

    out["Churn_Probability"] = p_ens.round(3)
    out["Predicted_Label"] = np.where(
        p_ens > 0.5,
        "Attrited Customer",
        "Existing Customer",
    )
    out["Segment"] = segments
    out["Risk"] = np.where(p_ens > 0.5, "HIGH", "LOW")
    return out


def clientnum_leakage_strength(raw, y_true):
    """Measure whether CLIENTNUM alone nearly separates the true labels."""
    if y_true is None or "CLIENTNUM" not in raw.columns or len(set(y_true)) < 2:
        return None

    clientnums = pd.to_numeric(raw["CLIENTNUM"], errors="coerce")
    if clientnums.isna().any():
        return None

    auc = roc_auc_score(y_true, clientnums)
    return max(auc, 1 - auc)


# ---------- UI ----------
st.set_page_config(page_title="Churn Dashboard", layout="wide")
st.title("💳 Credit Card Churn Prediction Dashboard")
st.markdown("Upload customer data to predict churn. "
            "Include a **`Churn`**, **`Attrition_Flag`**, or **`True_Label`** "
            "column to also see model performance (Gini).")

file = st.file_uploader("Upload CSV", type="csv")

if file:
    raw = pd.read_csv(file)

    # Extract true labels if present (before preprocessing drops them)
    y_true = extract_true_labels(raw)

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
        segments = [PERSONAS[str(c)] for c in X["Cluster"]]
        out = build_prediction_output(raw, p_ens, segments)
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
            leakage_strength = clientnum_leakage_strength(raw, y_true)
            if leakage_strength is not None and leakage_strength >= 0.98:
                st.warning(
                    "CLIENTNUM alone nearly separates the uploaded labels. "
                    "The model still excludes CLIENTNUM, but this backtest file "
                    "is likely label-leaked and should not be used as a final "
                    "generalization check."
                )

            st.success("Labels detected - evaluating model performance (backtest)")
            metrics = pd.DataFrame({
                "Model": ["XGBoost", "LightGBM", "Ensemble"],
                "Gini": [gini(y_true, p_xgb), gini(y_true, p_lgbm), gini(y_true, p_ens)],
                "AUC": [roc_auc_score(y_true, p_xgb),
                        roc_auc_score(y_true, p_lgbm),
                        roc_auc_score(y_true, p_ens)]
            }).round(4)
            st.dataframe(metrics, use_container_width=True)
        else:
            st.info("No label column in upload - prediction-only mode. "
                    "Add `Churn`, `Attrition_Flag`, or `True_Label` to see Gini/AUC.")
else:
    st.info("⬅ Upload a CSV file to begin.")
