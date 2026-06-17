import pandas as pd
from sklearn.model_selection import train_test_split
def load_data(path):
    df = pd.read_csv(path)
    return df
def clean_data(df):
    df = df.copy()
    if "CLIENTNUM" in df.columns:
        df = df.drop(columns=["CLIENTNUM"])
    leakage_cols = [c for c in df.columns if "Naive_Bayes" in c]
    df = df.drop(columns=leakage_cols)
    df["Churn"] = (df["Attrition_Flag"] == "Attrited Customer").astype(int)
    df = df.drop(columns=["Attrition_Flag"])

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
