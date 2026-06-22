"""
explain.py
----------
Functions for SHAP explainability: global feature importance
and per-cluster (per-persona) churn driver analysis.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import shap


def global_shap(model, X, save_path=None):
    """
    Compute and plot global SHAP feature importance.

    Shows which features most influence churn predictions overall,
    and returns a ranked importance table.

    Parameters
    ----------
    model : trained tree model (XGBoost or LightGBM)
    X : pd.DataFrame
        Features to explain (e.g., test set).
    save_path : str, optional
        If given, saves the summary plot here.

    Returns
    -------
    pd.Series
        Features ranked by mean absolute SHAP value (descending).
    """
    # TreeExplainer is fast and exact for tree-based models
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X)

    # Mean absolute SHAP = overall importance per feature
    importance = pd.Series(
        np.abs(shap_values).mean(axis=0),
        index=X.columns
    ).sort_values(ascending=False)

    # Summary (beeswarm) plot
    shap.summary_plot(shap_values, X, show=False)
    plt.title("Global SHAP Summary")
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, bbox_inches="tight", dpi=150)
    plt.show()

    return importance

def per_cluster_shap(model, X, cluster_labels, cluster_id, top_n=10, save_path=None):
    """
    Compute SHAP feature importance for a SINGLE customer segment.

    This reveals why a specific persona churns -- different segments
    often have different churn drivers, which is a key project insight.

    Parameters
    ----------
    model : trained tree model
    X : pd.DataFrame
        Features (same rows as cluster_labels).
    cluster_labels : array-like
        Cluster assignment for each row.
    cluster_id : int
        Which cluster to explain.
    top_n : int
        Number of top features to return.
    save_path : str, optional
        If given, saves the plot here.

    Returns
    -------
    pd.Series
        Top features for this cluster, ranked by importance.
    """
    # Select only rows belonging to this cluster
    mask = (np.array(cluster_labels) == cluster_id)
    X_cluster = X[mask]

    # SHAP for this subset
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_cluster)

    # Ranked importance for this cluster
    importance = pd.Series(
        np.abs(shap_values).mean(axis=0),
        index=X.columns
    ).sort_values(ascending=False).head(top_n)

    # Bar plot for this cluster
    shap.summary_plot(shap_values, X_cluster, plot_type="bar", show=False)
    plt.title(f"SHAP Importance — Cluster {cluster_id}")
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, bbox_inches="tight", dpi=150)
    plt.show()

    return importance
