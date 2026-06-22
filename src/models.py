"""
models.py
---------
Functions for Stage 2: training XGBoost and LightGBM, and building
a weighted soft-voting ensemble for churn prediction.
"""

import numpy as np
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.metrics import roc_auc_score


def train_xgb(X_train, y_train, random_state=42):
    """
    Train an XGBoost classifier for churn prediction.

    Handles class imbalance using scale_pos_weight, which tells the
    model to pay more attention to the minority (churned) class.

    Parameters
    ----------
    X_train : pd.DataFrame
        Training features.
    y_train : pd.Series
        Training target (Churn).
    random_state : int
        Seed for reproducibility (production-relevant for tree subsampling).

    Returns
    -------
    XGBClassifier
        The trained XGBoost model.
    """
    # Imbalance ratio: (#stayed) / (#churned) -> weights the minority class up
    scale_pos_weight = (y_train == 0).sum() / (y_train == 1).sum()

    model = XGBClassifier(
        n_estimators=300,           # number of trees
        learning_rate=0.05,         # how much each tree contributes
        max_depth=4,                # tree depth (controls complexity)
        subsample=0.8,              # use 80% of rows per tree (reduces overfit)
        colsample_bytree=0.8,       # use 80% of features per tree
        scale_pos_weight=scale_pos_weight,  # handle imbalance
        eval_metric="auc",          # optimize for ranking quality
        random_state=random_state,
        n_jobs=-1                   # use all CPU cores
    )

    model.fit(X_train, y_train)
    return model

def train_lgbm(X_train, y_train, random_state=42):
    """
    Train a LightGBM classifier for churn prediction.

    LightGBM is faster than XGBoost and often performs similarly.
    Also handles imbalance via scale_pos_weight.

    Parameters
    ----------
    X_train : pd.DataFrame
        Training features.
    y_train : pd.Series
        Training target (Churn).
    random_state : int
        Seed for reproducibility.

    Returns
    -------
    LGBMClassifier
        The trained LightGBM model.
    """
    scale_pos_weight = (y_train == 0).sum() / (y_train == 1).sum()

    model = LGBMClassifier(
        n_estimators=300,
        learning_rate=0.05,
        max_depth=4,
        subsample=0.8,
        colsample_bytree=0.8,
        scale_pos_weight=scale_pos_weight,
        random_state=random_state,
        n_jobs=-1,
        verbose=-1                  # suppress LightGBM's training logs
    )

    model.fit(X_train, y_train)
    return model

def optimize_ensemble_weight(p_xgb, p_lgbm, y_true):
    """
    Find the best weight 'w' for combining XGBoost and LightGBM.

    Tries weights from 0.0 to 1.0 and picks the one that maximizes
    the Gini coefficient on the given (validation) predictions.

    Ensemble formula: p_ensemble = w * p_xgb + (1 - w) * p_lgbm

    Parameters
    ----------
    p_xgb : np.ndarray
        XGBoost predicted probabilities.
    p_lgbm : np.ndarray
        LightGBM predicted probabilities.
    y_true : pd.Series
        True labels (to evaluate against).

    Returns
    -------
    float
        The weight 'w' that maximizes Gini.
    """
    best_w = 0.5
    best_gini = -1

    # Try weights 0.0, 0.05, 0.10, ..., 1.0
    for w in np.arange(0, 1.01, 0.05):
        p_ensemble = w * p_xgb + (1 - w) * p_lgbm
        gini = 2 * roc_auc_score(y_true, p_ensemble) - 1

        if gini > best_gini:
            best_gini = gini
            best_w = w

    return best_w

def ensemble_predict(p_xgb, p_lgbm, w):
    """
    Combine XGBoost and LightGBM probabilities into an ensemble.

    Parameters
    ----------
    p_xgb : np.ndarray
        XGBoost predicted probabilities.
    p_lgbm : np.ndarray
        LightGBM predicted probabilities.
    w : float
        Weight for XGBoost (1 - w goes to LightGBM).

    Returns
    -------
    np.ndarray
        Weighted ensemble probabilities.
    """
    return w * p_xgb + (1 - w) * p_lgbm
