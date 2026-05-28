"""
evaluation.py
-------------
Compute and display metrics for each model variant.

Metrics:
  - Test accuracy
  - Weighted F1-score  (handles class imbalance in wine quality scores)
  - Cohen's Kappa      (agreement beyond chance; useful for ordinal targets)
  - Confusion matrix   (visual, good for presentations)
"""

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    cohen_kappa_score,
    confusion_matrix,
    ConfusionMatrixDisplay,
    r2_score
)
import matplotlib.pyplot as plt


def compute_metrics(y_true, y_pred) -> dict:
    """Return a dict of evaluation metrics."""
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "f1_weighted": f1_score(y_true, y_pred, average="weighted", zero_division=0),
        "kappa": cohen_kappa_score(y_true, y_pred),
        "r2": r2_score(y_true, y_pred),
    }


def print_metrics(name: str, metrics: dict, result: dict) -> None:
    """Pretty-print metrics and model configuration."""
    print(f"\n{'=' * 50}")
    print(f"  {name}")
    print(f"{'=' * 50}")
    print(f"  Best K           : {result['best_k']}")
    if "best_n_components" in result:
        print(f"  PCA components   : {result['best_n_components']}")
    if "selection_method" in result:
        print(f"  Selection method : {result['selection_method']}")
        print(f"  Selected features: {result['features']}")
    print(f"  Test accuracy    : {metrics['accuracy']:.4f}")
    print(f"  Weighted F1      : {metrics['f1_weighted']:.4f}")
    print(f"  Cohen's Kappa    : {metrics['kappa']:.4f}")
    print(f"  R squared    : {metrics['r2']:.4f}")


def plot_cv_k(results: dict, save_path: str = "cv_k_curves.png") -> None:
    """
    Plot CV accuracy vs. K for each model variant.
    """
    fig, axes = plt.subplots(1, len(results), figsize=(5 * len(results), 4), sharey=True)
    if len(results) == 1:
        axes = [axes]

    for ax, (name, result) in zip(axes, results.items()):
        from model import K_GRID
        ax.plot(K_GRID, result["cv_scores"], marker="o", markersize=4)
        ax.axvline(result["best_k"], color="red", linestyle="--", label=f"Best K={result['best_k']}")
        ax.set_xlabel("K (number of neighbors)")
        ax.set_ylabel("CV Accuracy")
        ax.set_title(name)
        ax.legend()

    fig.suptitle("CV Accuracy vs. K", fontsize=13)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"\nSaved CV-K curves → {save_path}")

def plot_confusion_matrices(results, y_tests, y_preds, save_path="confusion_matrices.png"):
    all_labels = list(range(11))  # force 0–10
    n = len(results)
    fig, axes = plt.subplots(1, n, figsize=(5 * n, 4))
    if n == 1:
        axes = [axes]

    for ax, name in zip(axes, results.keys()):
        cm = confusion_matrix(y_tests[name], y_preds[name], labels=all_labels)
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=all_labels)
        disp.plot(ax=ax, colorbar=False)
        ax.set_title(name)

    fig.suptitle("Confusion Matrices", fontsize=13)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()

def plot_pca_components(component_scores: np.ndarray, best_n: int,
                         save_path: str = "pca_components.png") -> None:
    """Bar chart of CV accuracy vs. number of PCA components."""
    n_range = list(range(1, len(component_scores) + 1))
    plt.figure(figsize=(6, 4))
    plt.bar(n_range, component_scores, color="steelblue", alpha=0.7)
    plt.axvline(best_n, color="red", linestyle="--", label=f"Best n={best_n}")
    plt.xlabel("Number of PCA Components")
    plt.ylabel("CV Accuracy (K=10)")
    plt.title("PCA: CV Accuracy vs. Components")
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"Saved PCA component plot → {save_path}")