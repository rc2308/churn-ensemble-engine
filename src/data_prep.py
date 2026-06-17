import pandas as pd
from sklearn.model_selection import train_test_split
def load_data(path):
    df = pd.read_csv(path)
    return df
