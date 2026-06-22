"""
features.py
-----------
Functions for feature engineering, encoding, and scaling.
"""

import pandas as pd
from sklearn.preprocessing import StandardScaler


def engineer_features(df):
    """
    Create new behavioral features from existing columns.

    New features:
    - Amt_per_Transaction : average spend per transaction
      (Total_Trans_Amt / Total_Trans_Ct)
    - Contacts_per_Month  : how often the customer contacts the bank
      relative to tenure (engagement / friction signal)

    Parameters
    ----------
    df : pd.DataFrame
        Cleaned dataset (output of clean_data).

    Returns
    -------
    pd.DataFrame
        Dataset with new engineered features added.
    """
    df = df.copy()

    # Average amount spent per transaction (+1 avoids division by zero)
    df["Amt_per_Transaction"] = (
        df["Total_Trans_Amt"] / (df["Total_Trans_Ct"] + 1)
    )

    # Contact frequency relative to tenure (months on book)
    df["Contacts_per_Month"] = (
        df["Contacts_Count_12_mon"] / (df["Months_on_book"] + 1)
    )

    return df

def encode_categoricals(df):
    """
    One-hot encode all categorical (text) columns.

    Converts text columns (e.g., Gender, Education_Level) into
    numeric 0/1 columns that models can use.

    Parameters
    ----------
    df : pd.DataFrame
        Dataset with possible text columns.

    Returns
    -------
    pd.DataFrame
        Dataset with categoricals one-hot encoded.
    """
    df = df.copy()

    # Find all text (object) columns
    cat_cols = df.select_dtypes(include="object").columns.tolist()

    # One-hot encode them (drop_first avoids redundant columns)
    df = pd.get_dummies(df, columns=cat_cols, drop_first=True)

    return df

def scale_for_clustering(df, feature_cols):
    """
    Scale selected features for K-Means clustering.

    K-Means uses distances, so features must be on the same scale.
    (Tree models like XGBoost do NOT need this — only clustering does.)

    Parameters
    ----------
    df : pd.DataFrame
        Dataset containing the features to scale.
    feature_cols : list
        List of column names to scale for clustering.

    Returns
    -------
    scaled : np.ndarray
        The scaled feature matrix.
    scaler : StandardScaler
        The fitted scaler (saved later to transform new data identically).
    """
    scaler = StandardScaler()
    scaled = scaler.fit_transform(df[feature_cols])
    return scaled, scaler
