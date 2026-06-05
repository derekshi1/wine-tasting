"""
random_forest.py
----------------
Random Forest classifier for wine quality.

Tunes number of trees and max features via 5-fold CV.
Uses class_weight="balanced" to handle class imbalance natively.
"""

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV


PARAM_GRID = {
    "n_estimators": [100, 200, 500],
    "max_features": ["sqrt", "log2"],
}


def train_random_forest(X_train, y_train, cv: int = 5) -> dict:
    """
    Tune n_estimators and max_features via grid search CV.
    class_weight="balanced" down-weights majority classes (5, 6) automatically.
    """
    grid = GridSearchCV(
        RandomForestClassifier(class_weight="balanced", random_state=0),
        PARAM_GRID, cv=cv, scoring="accuracy", n_jobs=-1
    )
    grid.fit(X_train, y_train)

    best = grid.best_estimator_
    best_params = grid.best_params_
    print(f"  Best params: {best_params}  |  CV accuracy: {grid.best_score_:.4f}")

    return {
        "model":        best,
        "best_params":  best_params,
        "cv_score":     grid.best_score_,
        "importances":  best.feature_importances_,
        "feature_mask": None,
    }