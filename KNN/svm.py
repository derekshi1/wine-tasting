"""
svm.py
------
Support Vector Machine classifier for wine quality.

Tunes kernel (linear, RBF) and regularization parameter C via 5-fold CV.
Multi-class handled automatically via one-vs-one (scikit-learn default).
"""

import numpy as np
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV


PARAM_GRID = {
    "C":      [0.1, 1, 10, 100],
    "kernel": ["linear", "rbf"],
}


def train_svm(X_train, y_train, cv: int = 5) -> dict:
    """
    Tune C and kernel via grid search CV, then fit on full training set.
    """
    grid = GridSearchCV(
        SVC(), PARAM_GRID, cv=cv, scoring="accuracy", n_jobs=-1
    )
    grid.fit(X_train, y_train)

    best = grid.best_estimator_
    best_params = grid.best_params_
    print(f"  Best params: {best_params}  |  CV accuracy: {grid.best_score_:.4f}")

    return {
        "model":      best,
        "best_params": best_params,
        "cv_score":   grid.best_score_,
        "feature_mask": None,
    }