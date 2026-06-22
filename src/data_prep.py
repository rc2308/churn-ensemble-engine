import pandas as pd
from sklearn.model_selection import train_test_split


ID_COLUMNS = ("CLIENTNUM",)
TEXT_TARGET_COLUMNS = ("Attrition_Flag", "True_Label")
LEAKAGE_COLUMN_MARKERS = ("Naive_Bayes",)


def load_data(path):
    df = pd.read_csv(path)
    return df


def to_churn_target(labels):
    """Convert common churn label formats to 1=attrited, 0=existing."""
    if pd.api.types.is_numeric_dtype(labels):
        return labels.astype(int)

    normalized = labels.astype(str).str.strip().str.lower()
    return normalized.isin({"1", "1.0", "true", "attrited customer", "attrited"}).astype(int)


def leakage_columns(df):
    """Return columns that must never enter the model feature matrix."""
    cols = [c for c in ID_COLUMNS if c in df.columns]
    cols.extend(
        c
        for c in df.columns
        if any(marker in c for marker in LEAKAGE_COLUMN_MARKERS)
    )
    cols.extend(c for c in TEXT_TARGET_COLUMNS if c in df.columns)
    return list(dict.fromkeys(cols))


def clean_data(df):
    df = df.copy()
    if "Churn" not in df.columns:
        for label_col in TEXT_TARGET_COLUMNS:
            if label_col in df.columns:
                df["Churn"] = to_churn_target(df[label_col])
                break

    df = df.drop(columns=leakage_columns(df), errors="ignore")

    return df


def split_data(df, target_col="Churn", test_size=0.2, random_state=42):
    X = df.drop(columns=[target_col])
    y = df[target_col]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=test_size,
        stratify=y,
        random_state=random_state
    )

    return X_train, X_test, y_train, y_test
