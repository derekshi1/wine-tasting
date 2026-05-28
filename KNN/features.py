"""
features.py
-----------
Forward and backward stepwise feature selection for KNN classifiers.
"""

import numpy as np
from sklearn.model_selection import cross_val_score
from sklearn.neighbors import KNeighborsClassifier


def _cv_score(X, y, features: list, k: int, cv: int = 5) -> float:
    """Return mean CV accuracy for a given feature subset and K."""
    knn = KNeighborsClassifier(n_neighbors=k)
    scores = cross_val_score(knn, X[:, features], y, cv=cv, scoring="accuracy")
    return scores.mean()


def forward_stepwise(X, y, k: int, cv: int = 5) -> list:
    """
    Forward stepwise selection.

    Starts with an empty set and greedily adds the feature that
    gives the largest CV accuracy gain at each step.

    Returns
    -------
    best_features : list of column indices giving the best subset found.
    """
    n_features = X.shape[1]
    remaining = list(range(n_features))
    selected = []
    best_overall_score = -np.inf
    best_overall_subset = []

    while remaining:
        scores = {}
        for feat in remaining:
            candidate = selected + [feat]
            scores[feat] = _cv_score(X, y, candidate, k, cv)

        best_feat = max(scores, key=scores.get)
        selected = selected + [best_feat]
        remaining.remove(best_feat)

        if scores[best_feat] > best_overall_score:
            best_overall_score = scores[best_feat]
            best_overall_subset = selected.copy()

    return best_overall_subset


def backward_stepwise(X, y, k: int, cv: int = 5) -> list:
    """
    Backward stepwise selection.

    Starts with all features and greedily removes the feature whose
    removal costs the least in CV accuracy.

    Returns
    -------
    best_features : list of column indices giving the best subset found.
    """
    n_features = X.shape[1]
    selected = list(range(n_features))
    best_overall_score = _cv_score(X, y, selected, k, cv)
    best_overall_subset = selected.copy()

    while len(selected) > 1:
        scores = {}
        for feat in selected:
            candidate = [f for f in selected if f != feat]
            scores[feat] = _cv_score(X, y, candidate, k, cv)

        # Remove the feature whose removal hurts the least
        drop_feat = max(scores, key=scores.get)
        selected = [f for f in selected if f != drop_feat]

        if scores[drop_feat] > best_overall_score:
            best_overall_score = scores[drop_feat]
            best_overall_subset = selected.copy()

    return best_overall_subset