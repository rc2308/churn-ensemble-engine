"""
clustering.py
-------------
Functions for Stage 1: customer segmentation using K-Means.
"""

import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score


def find_optimal_k(scaled_data, k_range=range(2, 8), random_state=42):
    """
    Evaluate different numbers of clusters (K) to help choose the best.

    For each K, computes:
    - Inertia (for the Elbow method): total within-cluster distance.
      Lower is better, but it always decreases, so we look for the "elbow".
    - Silhouette score: how well-separated the clusters are.
      Ranges from -1 to 1; higher is better.

    Parameters
    ----------
    scaled_data : np.ndarray
        Scaled feature matrix (output of scale_for_clustering).
    k_range : range
        The range of K values to test (default 2 to 7).
    random_state : int
        Seed so cluster initialization is reproducible.

    Returns
    -------
    results : dict
        {k: {"inertia": value, "silhouette": value}} for each K tested.
    """
    results = {}

    for k in k_range:
        km = KMeans(n_clusters=k, random_state=random_state, n_init=10)
        labels = km.fit_predict(scaled_data)

        inertia = km.inertia_
        sil = silhouette_score(scaled_data, labels)

        results[k] = {"inertia": inertia, "silhouette": sil}

    return results

def fit_kmeans(scaled_data, k, random_state=42):
    """
    Train a K-Means model with the chosen number of clusters.

    Parameters
    ----------
    scaled_data : np.ndarray
        Scaled feature matrix.
    k : int
        Chosen number of clusters.
    random_state : int
        Seed for reproducible clusters (matters in production).

    Returns
    -------
    KMeans
        The fitted K-Means model.
    """
    km = KMeans(n_clusters=k, random_state=random_state, n_init=10)
    km.fit(scaled_data)
    return km

def assign_clusters(km_model, scaled_data):
    """
    Assign each row to its cluster using a fitted K-Means model.

    Parameters
    ----------
    km_model : KMeans
        A fitted K-Means model.
    scaled_data : np.ndarray
        Scaled feature matrix (same features used in training).

    Returns
    -------
    np.ndarray
        Array of cluster labels (e.g., [0, 2, 1, 0, 3, ...]).
    """
    return km_model.predict(scaled_data)

def profile_clusters(df, cluster_labels, profile_cols):
    """
    Summarize each cluster's average characteristics.

    This helps interpret and NAME the personas (e.g., "Dormant",
    "Loyal High-Spender") by showing the average values of key
    features per cluster.

    Parameters
    ----------
    df : pd.DataFrame
        The dataset (with original feature values).
    cluster_labels : np.ndarray
        Cluster assignment for each row.
    profile_cols : list
        Columns to summarize (e.g., transaction count, utilization).

    Returns
    -------
    pd.DataFrame
        Average feature values per cluster.
    """
    df = df.copy()
    df["Cluster"] = cluster_labels

    # Average of key features, grouped by cluster
    profile = df.groupby("Cluster")[profile_cols].mean()

    return profile
