#!/usr/bin/env python3
"""
Faster hyperparameter tuning using RandomizedSearchCV.
Tests random combinations instead of all possible combinations.
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np
from sklearn.model_selection import RandomizedSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from load_data import load_wine_data
from random_forest.utils import split_data, preprocess_features
from synthetic_data_gen.mixup_data_augmentation import MixupAugmentation


def tune_hyperparameters_random(X_train, y_train, X_val, y_val, n_iter=100):
    """
    Use Randomized Search to find good hyperparameters (faster than grid search).
    
    Args:
        X_train: Training features
        y_train: Training targets
        X_val: Validation features
        y_val: Validation targets
        n_iter: Number of random combinations to test
    
    Returns:
        Tuple of (best_model, best_params, search_results)
    """
    print("🔍 Starting randomized hyperparameter search...")
    print("=" * 70)
    
    # Define parameter distributions
    param_dist = {
        'n_estimators': [50, 75, 100, 125, 150, 175, 200],
        'max_depth': [10, 12, 15, 18, 20, 22, 25, None],
        'min_samples_split': [2, 3, 5, 7, 10],
        'min_samples_leaf': [1, 2, 3, 4],
        'max_features': ['sqrt', 'log2', None],
        'bootstrap': [True, False],
        'oob_score': [True, False]
    }
    
    print(f"📊 Parameter Distribution Space:")
    print(f"   n_estimators: {param_dist['n_estimators']}")
    print(f"   max_depth: {param_dist['max_depth']}")
    print(f"   min_samples_split: {param_dist['min_samples_split']}")
    print(f"   min_samples_leaf: {param_dist['min_samples_leaf']}")
    print(f"   max_features: {param_dist['max_features']}")
    print(f"   bootstrap: {param_dist['bootstrap']}")
    print(f"   oob_score: {param_dist['oob_score']}")
    
    print(f"\n   Testing {n_iter} random combinations with 5-fold CV")
    print(f"   Total models to train: {n_iter * 5} fits\n")
    
    # Initialize base model
    rf = RandomForestRegressor(random_state=42, n_jobs=-1)
    
    # Randomized search
    print(f"🚀 Running Randomized Search...\n")
    
    random_search = RandomizedSearchCV(
        rf,
        param_distributions=param_dist,
        n_iter=n_iter,
        cv=5,
        scoring='r2',
        n_jobs=-1,
        verbose=1,
        random_state=42
    )
    
    random_search.fit(X_train, y_train)
    
    # Extract results
    results_df = pd.DataFrame(random_search.cv_results_)
    best_params = random_search.best_params_
    best_model = random_search.best_estimator_
    best_score = random_search.best_score_
    
    print("\n" + "=" * 70)
    print("✅ RANDOMIZED SEARCH COMPLETE")
    print("=" * 70)
    
    print(f"\n🏆 Best Parameters Found:")
    for param, value in sorted(best_params.items()):
        print(f"   {param:<25} = {value}")
    
    print(f"\n📊 Best Validation R² (5-fold CV): {best_score:.4f}")
    
    # Evaluate on validation set
    val_pred = best_model.predict(X_val)
    val_rmse = np.sqrt(mean_squared_error(y_val, val_pred))
    val_mae = mean_absolute_error(y_val, val_pred)
    val_r2 = r2_score(y_val, val_pred)
    
    print(f"\n📈 Best Model Performance on Validation Set:")
    print(f"   RMSE: {val_rmse:.4f}")
    print(f"   MAE:  {val_mae:.4f}")
    print(f"   R²:   {val_r2:.4f}")
    
    return best_model, best_params, results_df


def evaluate_model(model, X_train, y_train, X_val, y_val, X_test, y_test, model_name="Model"):
    """
    Evaluate model on all sets.
    
    Args:
        model: Trained model
        X_train, y_train: Training data
        X_val, y_val: Validation data
        X_test, y_test: Test data
        model_name: Name for output
    
    Returns:
        Dictionary of metrics
    """
    print(f"\n📊 {model_name} - Full Evaluation:")
    print("=" * 70)
    
    metrics = {}
    
    for X, y, set_name in [
        (X_train, y_train, "Training"),
        (X_val, y_val, "Validation"),
        (X_test, y_test, "Test (Unseen)")
    ]:
        pred = model.predict(X)
        rmse = np.sqrt(mean_squared_error(y, pred))
        mae = mean_absolute_error(y, pred)
        r2 = r2_score(y, pred)
        
        metrics[set_name] = {"RMSE": rmse, "MAE": mae, "R2": r2}
        
        print(f"\n{set_name}:")
        print(f"   RMSE: {rmse:.4f}")
        print(f"   MAE:  {mae:.4f}")
        print(f"   R²:   {r2:.4f}")
    
    return metrics


def get_top_parameters(results_df, top_n=10):
    """
    Get top N parameter combinations by validation score.
    
    Args:
        results_df: RandomizedSearchCV results dataframe
        top_n: Number of top results to show
    
    Returns:
        DataFrame of top results
    """
    print(f"\n🏅 Top {top_n} Parameter Combinations (by Validation R²):")
    print("=" * 90)
    
    # Sort by mean test score
    param_cols = [col for col in results_df.columns if col.startswith('param_')]
    top_results = results_df.nlargest(top_n, 'mean_test_score')[
        param_cols + ['mean_test_score', 'std_test_score']
    ]
    
    print(f"{'Rank':<5} {'n_est':<8} {'max_d':<8} {'min_spl':<8} {'min_leaf':<8} {'R² (mean)':<12} {'R² (std)':<10}")
    print("-" * 90)
    
    for idx, (i, row) in enumerate(top_results.iterrows(), 1):
        n_est = int(row['param_n_estimators']) if pd.notna(row['param_n_estimators']) else 0
        max_d = str(row['param_max_depth']) if pd.notna(row['param_max_depth']) else "None"
        min_spl = int(row['param_min_samples_split']) if pd.notna(row['param_min_samples_split']) else 0
        min_leaf = int(row['param_min_samples_leaf']) if pd.notna(row['param_min_samples_leaf']) else 0
        
        print(f"{idx:<5} {n_est:<8} {max_d:<8} {min_spl:<8} {min_leaf:<8} "
              f"{row['mean_test_score']:<12.4f} {row['std_test_score']:<10.4f}")
    
    return top_results


def main():
    """Main hyperparameter tuning pipeline."""
    print("🍷 Wine Quality Prediction - Random Forest Hyperparameter Tuning (Fast)")
    print("=" * 70)
    
    # Load and preprocess data
    print("\n📂 Loading and preprocessing data...")
    df = load_wine_data()
    X_train, X_val, X_test, y_train, y_val, y_test = split_data(df)
    X_train, X_val, X_test = preprocess_features(X_train, X_val, X_test)
    print("   ✅ Complete")
    
    # Augment training data with Mixup
    print("\n🔄 Augmenting training data with Mixup (1x)...")
    augmentor = MixupAugmentation(random_state=42)
    X_train_aug, y_train_aug = augmentor.augment_dataset(X_train, y_train, augmentation_factor=1.0)
    
    # ==================== BASELINE MODEL ====================
    print("\n" + "=" * 70)
    print("📊 BASELINE MODEL (Original Parameters)")
    print("=" * 70)
    
    baseline_model = RandomForestRegressor(
        n_estimators=100,
        max_depth=20,
        min_samples_split=5,
        min_samples_leaf=2,
        random_state=42,
        n_jobs=-1
    )
    baseline_model.fit(X_train_aug, y_train_aug)
    baseline_metrics = evaluate_model(baseline_model, X_train_aug, y_train_aug, X_val, y_val, X_test, y_test, "Baseline")
    
    # ==================== HYPERPARAMETER TUNING ====================
    print("\n" + "=" * 70)
    print("🔍 HYPERPARAMETER TUNING (Randomized Search)")
    print("=" * 70)
    
    tuned_model, best_params, results_df = tune_hyperparameters_random(
        X_train_aug, y_train_aug, X_val, y_val, n_iter=100
    )
    
    # Get top parameters
    top_results = get_top_parameters(results_df, top_n=10)
    
    # ==================== TUNED MODEL EVALUATION ====================
    print("\n" + "=" * 70)
    print("📊 TUNED MODEL EVALUATION")
    print("=" * 70)
    
    tuned_metrics = evaluate_model(tuned_model, X_train_aug, y_train_aug, X_val, y_val, X_test, y_test, "Tuned")
    
    # ==================== COMPARISON ====================
    print("\n" + "=" * 70)
    print("📋 BASELINE vs TUNED MODEL COMPARISON (Test Set)")
    print("=" * 70)
    
    print(f"\n{'Metric':<15} {'Baseline':<15} {'Tuned':<15} {'Change':<15}")
    print("-" * 60)
    
    for metric in ["RMSE", "MAE", "R2"]:
        baseline_val = baseline_metrics["Test (Unseen)"][metric]
        tuned_val = tuned_metrics["Test (Unseen)"][metric]
        
        if metric == "R2":
            change = tuned_val - baseline_val
            arrow = "↑" if change > 0 else "↓" if change < 0 else "="
        else:
            change = baseline_val - tuned_val
            arrow = "↓" if change > 0 else "↑" if change < 0 else "="
        
        change_pct = (abs(change) / abs(baseline_val) * 100) if baseline_val != 0 else 0
        
        print(f"{metric:<15} {baseline_val:<15.4f} {tuned_val:<15.4f} {arrow} {change_pct:.2f}%")
    
    # ==================== FEATURE IMPORTANCE ====================
    print("\n" + "=" * 70)
    print("🎯 FEATURE IMPORTANCE (Top 10) - Tuned Model")
    print("=" * 70)
    
    importances = tuned_model.feature_importances_
    feature_importance_df = pd.DataFrame({
        "feature": X_train_aug.columns,
        "importance": importances
    }).sort_values("importance", ascending=False)
    
    print(f"\n{'Feature':<25} {'Importance':<15}")
    print("-" * 40)
    for idx, row in feature_importance_df.head(10).iterrows():
        print(f"{row['feature']:<25} {row['importance']:<15.4f}")
    
    # ==================== SUMMARY ====================
    print("\n" + "=" * 70)
    print("📝 SUMMARY & RECOMMENDATIONS")
    print("=" * 70)
    
    test_improvement = tuned_metrics["Test (Unseen)"]["R2"] - baseline_metrics["Test (Unseen)"]["R2"]
    rmse_improvement = baseline_metrics["Test (Unseen)"]["RMSE"] - tuned_metrics["Test (Unseen)"]["RMSE"]
    
    print(f"\n✅ Test R² Improvement: {test_improvement:+.4f} ({abs(test_improvement/baseline_metrics['Test (Unseen)']['R2']*100):.2f}%)")
    print(f"✅ Test RMSE Improvement: {rmse_improvement:+.4f}")
    
    print(f"\n🏆 Best Parameters Found:")
    for param, value in sorted(best_params.items()):
        print(f"   {param:<25} = {value}")
    
    print(f"\n📊 Final Test Set Performance:")
    print(f"   Baseline R²: {baseline_metrics['Test (Unseen)']['R2']:.4f}")
    print(f"   Tuned R²:    {tuned_metrics['Test (Unseen)']['R2']:.4f}")
    print(f"   Baseline RMSE: {baseline_metrics['Test (Unseen)']['RMSE']:.4f}")
    print(f"   Tuned RMSE:    {tuned_metrics['Test (Unseen)']['RMSE']:.4f}")
    
    if test_improvement > 0.05:
        print(f"\n🎉 Significant improvement from tuning!")
    elif test_improvement > 0:
        print(f"\n✓ Modest improvement from tuning.")
    else:
        print(f"\n⚠️  No improvement from tuning on test set.")
    
    print("\n" + "=" * 70)


if __name__ == "__main__":
    main()
