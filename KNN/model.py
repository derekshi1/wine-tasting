"""
model.py
--------
KNN model factory.  Three variants are supported:

    'baseline'   - all features, CV-tuned K
    'stepwise'   - forward / backward feature selection, CV-tuned K
    'pca'        - PCA on all features, CV-tuned K

Each variant is returned as a scikit-learn Pipeline so that the
feature-transformation step and the classifier are bundled together.
"""

from sklearn.neighbors import KNeighborsClassifier
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline


# Candidate values of K to search over
K_GRID = list(range(1, 31))


def make_knn_pipeline(n_neighbors: int, n_components: int = None, weights: str = "uniform") -> Pipeline:
    """
    Build a KNN pipeline.

    Parameters
    ----------
    n_neighbors  : K for KNeighborsClassifier
    n_components : If provided, prepend a PCA step that keeps this many
                   components.  Pass None for baseline / stepwise variants.
    """
    steps = []
    if n_components is not None:
        steps.append(("pca", PCA(n_components=n_components)))
    steps.append(("knn", KNeighborsClassifier(n_neighbors=n_neighbors, weights=weights)))
    return Pipeline(steps)