"""
data.py
-------
Load and preprocess the wine quality dataset.
Returns numpy arrays / pandas DataFrames
with a clean train/test split.
"""

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


FEATURE_NAMES = [
    "fixed acidity", "volatile acidity", "citric acid", "residual sugar",
    "chlorides", "free sulfur dioxide", "total sulfur dioxide", "density",
    "pH", "sulphates", "alcohol",
]
TARGET = "quality"


def load_wine(path: str = "../WineQT.csv", sep: str = ",") -> pd.DataFrame:
    """Load raw wine CSV and return a DataFrame."""
    df = pd.read_csv(path, sep=sep)
    df = df.drop(columns=["Id"], errors="ignore")  # errors="ignore" in case some versions lack it
    return df


def get_Xy(df: pd.DataFrame):
    """Split DataFrame into feature matrix X and target vector y."""
    X = df[FEATURE_NAMES].values.astype(float)
    y = df[TARGET].values.astype(int)
    return X, y


def split_and_scale(X, y, test_size=0.2, random_state=0):
    """
    Perform a stratified train/test split then standardize X.
    Returns X_train, X_test, y_train, y_test, scaler.

    KNN is distance-based, so scaling is critical.
    Scaler is fit on training data only to prevent data leakage.
    """
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    return X_train, X_test, y_train, y_test, scaler