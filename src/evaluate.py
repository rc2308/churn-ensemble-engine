"""
evaluate.py
-----------
Functions for evaluating churn models: Gini, supporting metrics,
comparison tables, and ROC curve plots.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    recall_score,
    f1_score,
    roc_curve,
)


def gini(y_true, y_pred):
    """
    Compute the Gini coefficient (the project's primary metric).

    Gini = 2 * AUC - 1, where AUC is the area under the ROC curve.
    Ranges from 0 (random) to 1 (perfect). This is the standard
    metric used in credit-risk competitions (e.g., AMEX).

    Parameters
    ----------
    y_true : array-like
        True binary labels.
    y_pred : array-like
        Predicted probabilities.

    Returns
    -------
    float
        The Gini coefficient.
    """
    return 2 * roc_auc_score(y_true, y_pred) - 1

def compute_metrics(y_true, y_pred, threshold=0.5):
    """
    Compute a full set of evaluation metrics for a model.

    Includes Gini (primary) plus supporting metrics that matter
    under class imbalance (PR-AUC, Recall, F1).

    Parameters
    ----------
    y_true : array-like
        True binary labels.
    y_pred : array-like
        Predicted probabilities.
    threshold : float
        Cutoff to convert probabilities to 0/1 (for Recall/F1).

    Returns
    -------
    dict
        {gini, auc, pr_auc, recall, f1}
    """
    # Convert probabilities to binary predictions for Recall/F1
    y_label = (y_pred >= threshold).astype(int)

    metrics = {
        "gini": 2 * roc_auc_score(y_true, y_pred) - 1,
        "auc": roc_auc_score(y_true, y_pred),
        "pr_auc": average_precision_score(y_true, y_pred),
        "recall": recall_score(y_true, y_label),
        "f1": f1_score(y_true, y_label),
    }
    return metrics

def comparison_table(results_dict):
    """
    Build a comparison table from multiple models' metrics.

    Parameters
    ----------
    results_dict : dict
        {model_name: metrics_dict}, e.g.,
        {"XGBoost": {...}, "LightGBM": {...}, "Ensemble": {...}}

    Returns
    -------
    pd.DataFrame
        A table with models as rows and metrics as columns.
    """
    table = pd.DataFrame(results_dict).T  # transpose: models as rows
    table = table.round(4)                 # 4 decimal places
    return table

def plot_roc_curves(y_true, predictions_dict, save_path=None):
    """
    Plot overlaid ROC curves for multiple models.

    Parameters
    ----------
    y_true : array-like
        True binary labels.
    predictions_dict : dict
        {model_name: predicted_probabilities}.
    save_path : str, optional
        If given, saves the figure to this path.
    """
    plt.figure(figsize=(7, 6))

    for name, y_pred in predictions_dict.items():
        fpr, tpr, _ = roc_curve(y_true, y_pred)
        auc = roc_auc_score(y_true, y_pred)
        g = 2 * auc - 1
        plt.plot(fpr, tpr, label=f"{name} (Gini={g:.3f})")

    # Diagonal reference line (random model)
    plt.plot([0, 1], [0, 1], "k--", label="Random (Gini=0)")

    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve Comparison")
    plt.legend()

    if save_path:
        plt.savefig(save_path, bbox_inches="tight", dpi=150)
    plt.show()
