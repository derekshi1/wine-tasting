"""
training.py
-----------
CV-based tuning of K and model fitting for each variant.
"""

import numpy as np
from sklearn.model_selection import cross_val_score
from sklearn.decomposition import PCA

from model import make_knn_pipeline, K_GRID
from features import forward_stepwise, backward_stepwise


# ── helpers ──────────────────────────────────────────────────────────────────

def tune_k(X, y, cv: int = 5) -> tuple[int, np.ndarray]:
    """
    Search over K_GRID using k-fold CV accuracy.

    Returns
    -------
    best_k      : K with highest mean CV accuracy
    cv_scores   : mean CV accuracy for every K in K_GRID (for plotting)
    """
    cv_scores = np.array([
        cross_val_score(
            make_knn_pipeline(k), X, y, cv=cv, scoring="accuracy"
        ).mean()
        for k in K_GRID
    ])
    best_k = K_GRID[np.argmax(cv_scores)]
    return best_k, cv_scores


def tune_pca_components(X, y, cv: int = 5) -> tuple[int, np.ndarray]:
    """
    Search over number of PCA components (1 … p) using k-fold CV accuracy
    with a fixed K=10 (reasonable default for component search).
    The best number of components is then passed to tune_k.

    Returns
    -------
    best_n  : number of PCA components with highest mean CV accuracy
    scores  : mean CV accuracy for each component count
    """
    n_features = X.shape[1]
    component_range = list(range(1, n_features + 1))
    scores = np.array([
        cross_val_score(
            make_knn_pipeline(n_neighbors=10, n_components=n),
            X, y, cv=cv, scoring="accuracy"
        ).mean()
        for n in component_range
    ])
    best_n = component_range[np.argmax(scores)]
    return best_n, scores


# ── per-variant training ─────────────────────────────────────────────────────

def train_baseline(X_train, y_train, cv: int = 5) -> dict:
    """
    Baseline: all features, CV-tuned K.
    """
    best_k, cv_scores = tune_k(X_train, y_train, cv)
    model = make_knn_pipeline(best_k)
    model.fit(X_train, y_train)
    return {
        "model": model,
        "best_k": best_k,
        "cv_scores": cv_scores,
        "features": list(range(X_train.shape[1])),  # all features
        "feature_mask": None,
    }


def train_stepwise(X_train, y_train, cv: int = 5) -> dict:
    """
    Feature selection: run forward and backward stepwise selection,
    pick whichever subset scores higher in CV, then tune K on that subset.
    """
    # Use a fixed K=10 during selection to keep runtime reasonable
    k_for_selection = 10

    print("  Running forward stepwise selection…")
    fwd_features = forward_stepwise(X_train, y_train, k=k_for_selection, cv=cv)

    print("  Running backward stepwise selection…")
    bwd_features = backward_stepwise(X_train, y_train, k=k_for_selection, cv=cv)

    # Choose whichever subset has higher CV accuracy
    fwd_score = cross_val_score(
        make_knn_pipeline(k_for_selection),
        X_train[:, fwd_features], y_train, cv=cv, scoring="accuracy"
    ).mean()

    bwd_score = cross_val_score(
        make_knn_pipeline(k_for_selection),
        X_train[:, bwd_features], y_train, cv=cv, scoring="accuracy"
    ).mean()

    if fwd_score >= bwd_score:
        best_features = fwd_features
        selection_method = "forward"
    else:
        best_features = bwd_features
        selection_method = "backward"

    print(f"  Best method: {selection_method}  |  features: {best_features}")

    # Now tune K on the chosen feature subset
    best_k, cv_scores = tune_k(X_train[:, best_features], y_train, cv)
    model = make_knn_pipeline(best_k)
    model.fit(X_train[:, best_features], y_train)

    return {
        "model": model,
        "best_k": best_k,
        "cv_scores": cv_scores,
        "features": best_features,
        "feature_mask": best_features,
        "selection_method": selection_method,
        "fwd_features": fwd_features,
        "bwd_features": bwd_features,
    }


def train_pca(X_train, y_train, cv: int = 5) -> dict:
    """
    PCA on all 11 features, then CV-tune both number of components and K.
    """
    print("  Tuning number of PCA components…")
    best_n, component_scores = tune_pca_components(X_train, y_train, cv)
    print(f"  Best number of components: {best_n}")

    # Project training data with best_n components, then tune K
    pca = PCA(n_components=best_n)
    X_pca = pca.fit_transform(X_train)
    best_k, cv_scores = tune_k(X_pca, y_train, cv)

    # Build the full pipeline (PCA + KNN) and fit on original training data
    model = make_knn_pipeline(best_k, n_components=best_n)
    model.fit(X_train, y_train)

    return {
        "model": model,
        "best_k": best_k,
        "best_n_components": best_n,
        "cv_scores": cv_scores,
        "component_scores": component_scores,
        "features": list(range(X_train.shape[1])),
        "feature_mask": None,
    }

def train_distance_weighted(X_train, y_train, cv: int = 5) -> dict:
    """
    Distance-weighted KNN: closer neighbors contribute more to the vote.
    All features, CV-tuned K.
    """
    cv_scores = np.array([
        cross_val_score(
            make_knn_pipeline(k, weights="distance"), X_train, y_train,
            cv=cv, scoring="accuracy"
        ).mean()
        for k in K_GRID
    ])
    best_k = K_GRID[np.argmax(cv_scores)]
    model = make_knn_pipeline(best_k, weights="distance")
    model.fit(X_train, y_train)
    return {
        "model": model,
        "best_k": best_k,
        "cv_scores": cv_scores,
        "features": list(range(X_train.shape[1])),
        "feature_mask": None,
    }