"""
main.py
-------
End-to-end pipeline:
  1. Load and split the wine quality dataset
  2. Train models:
       KNN  - Baseline, Stepwise, PCA, Tree Selection, SMOTE+DW
       SVM  - Grid search over kernel and C
       RF   - Grid search over n_estimators and max_features
  3. Evaluate all models with accuracy, weighted F1, and Cohen's Kappa
  4. Save plots: CV-K curves, confusion matrices, PCA component chart


Usage
-----
  python main.py                        # uses winequality-red.csv in CWD
  python main.py --data path/to/file.csv [--sep ";"] [--cv 5] [--seed 0]
"""

import argparse
import numpy as np

from data import load_wine, get_Xy, split_and_scale
from training import train_baseline, train_stepwise, train_pca, train_distance_weighted, train_tree_selection, train_smote_distance
from svm import train_svm
from randomforest import train_random_forest
from evaluation import (
    compute_metrics,
    print_metrics,
    plot_cv_k,
    plot_confusion_matrices,
    plot_pca_components,
)


def parse_args():
    parser = argparse.ArgumentParser(description="KNN Wine Quality Classifier")
    parser.add_argument("--data", default="../WineQT.csv")
    parser.add_argument("--sep", default=",")
    parser.add_argument("--cv", type=int, default=5, help="Number of CV folds")
    parser.add_argument("--seed", type=int, default=0, help="Random seed")
    return parser.parse_args()


def main():
    args = parse_args()

    # ── 1. Data ───────────────────────────────────────────────────────────────
    print("Loading data…")
    df = load_wine(args.data, sep=args.sep)
    X, y = get_Xy(df)
    X_train, X_test, y_train, y_test, _ = split_and_scale(
        X, y, random_state=args.seed
    )
    print(f"  Train: {X_train.shape}   Test: {X_test.shape}")
    print(f"  Quality classes present: {sorted(np.unique(y))}")

    # ── 2. Train ──────────────────────────────────────────────────────────────
    print("\n[1/8] Training baseline model…")
    baseline = train_baseline(X_train, y_train, cv=args.cv)

    print("\n[2/8] Training stepwise feature-selection model…")
    stepwise = train_stepwise(X_train, y_train, cv=args.cv)

    print("\n[3/8] Training tree-based feature selection model…")
    tree_result = train_tree_selection(X_train, y_train, cv=args.cv)

    print("\n[4/8] Training PCA model…")
    pca_result = train_pca(X_train, y_train, cv=args.cv)

    print("\n[5/8] Training distance-weighted KNN…")
    distance_weighted = train_distance_weighted(X_train, y_train, cv=args.cv)

    print("\n[6/8] Training SMOTE + distance-weighted model…")
    smote_result = train_smote_distance(X_train, y_train, cv=args.cv)

    print("\n[7/8] SVM…")
    svm_result = train_svm(X_train, y_train, cv=args.cv)
 
    print("\n[8/8] Random Forest…")
    rf_result = train_random_forest(X_train, y_train, cv=args.cv)
 

    # ── 3. Predict ────────────────────────────────────────────────────────────
    def predict(result):
        """Apply feature mask if one exists, then predict."""
        mask = result.get("feature_mask")
        X_in = X_test[:, mask] if mask is not None else X_test
        return result["model"].predict(X_in)
    
    knn_results = {
        "Baseline": baseline,
        "Stepwise":  stepwise,
        "Tree Sel":   tree_result,
        "PCA":       pca_result,
        "Distance Weighted": distance_weighted,
        "SMOTE+DW":   smote_result,
    }
    
    all_results = {
        **knn_results,
        "SVM":    svm_result,
        "Rand Forest": rf_result,
    }

    preds   = {name: predict(result) for name, result in all_results.items()}
    y_tests = {name: y_test for name in all_results}

    # ── 4. Evaluate ───────────────────────────────────────────────────────────

    all_metrics = {}
    for name, result in all_results.items():
        m = compute_metrics(y_test, preds[name])
        all_metrics[name] = m
        print_metrics(name, m, result)

    # ── 5. Summary table ──────────────────────────────────────────────────────
    print(f"\n{'=' * 50}")
    print(f"  Summary")
    print(f"{'=' * 50}")
    print(f"  {'Model':<12} {'Accuracy':>10} {'F1 (wtd)':>10} {'Kappa':>10}")
    print(f"  {'-'*44}")
    for name, m in all_metrics.items():
        print(f"  {name:<12} {m['accuracy']:>10.4f} {m['f1_weighted']:>10.4f} {m['kappa']:>10.4f}")

    # ── 6. Plots ──────────────────────────────────────────────────────────────
    plot_cv_k(knn_results)
    plot_confusion_matrices(all_results, y_tests, preds)
    if "component_scores" in pca_result:
        plot_pca_components(pca_result["component_scores"], pca_result["best_n_components"])


if __name__ == "__main__":
    main()