"""
Utility functions for data splitting and preprocessing.
"""

import pandas as pd
from sklearn.model_selection import train_test_split


def split_data(df: pd.DataFrame, test_size: float = 0.2, val_size: float = 0.25, random_state: int = 42):
    """
    Split data into train (60%), validation (20%), and test (20%) sets.
    
    Uses hierarchical splitting:
    1. First split: 80% train+val, 20% test
    2. Second split: 75% train, 25% val (of the 80%)
    
    This ensures test set is never seen during training or hyperparameter tuning.
    
    Args:
        df: Full dataset
        test_size: Proportion for test set (default 0.2 = 20%)
        val_size: Proportion for validation within train+val (default 0.25 = 20% of total)
        random_state: Random seed for reproducibility
    
    Returns:
        Tuple of (X_train, X_val, X_test, y_train, y_val, y_test)
    """
    # Separate features and target
    if "quality" in df.columns:
        X = df.drop("quality", axis=1)
        y = df["quality"]
    else:
        raise ValueError("'quality' column not found in dataset")
    
    # First split: 80% train+val, 20% test
    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y,
        test_size=test_size,
        random_state=random_state,
        shuffle=True
    )
    
    # Second split: split temp into train (75%) and validation (25%)
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp,
        test_size=val_size,
        random_state=random_state,
        shuffle=True
    )
    
    # Verify split proportions
    total = len(df)
    print(f"Data Split Summary:")
    print(f"   Total samples: {total}")
    print(f"   Training set:   {len(X_train)} ({len(X_train)/total*100:.1f}%)")
    print(f"   Validation set: {len(X_val)} ({len(X_val)/total*100:.1f}%)")
    print(f"   Test set:       {len(X_test)} ({len(X_test)/total*100:.1f}%)")
    
    return X_train, X_val, X_test, y_train, y_val, y_test


def preprocess_features(X_train: pd.DataFrame, X_val: pd.DataFrame, X_test: pd.DataFrame):
    """
    Preprocess features (handle any data issues).
    
    Args:
        X_train: Training features
        X_val: Validation features
        X_test: Test features
    
    Returns:
        Tuple of (X_train, X_val, X_test) preprocessed
    """
    # Handle missing values if any
    X_train = X_train.fillna(X_train.mean())
    X_val = X_val.fillna(X_train.mean())  # Use train mean for consistency
    X_test = X_test.fillna(X_train.mean())
    
    return X_train, X_val, X_test
