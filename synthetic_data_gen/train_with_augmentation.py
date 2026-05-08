#!/usr/bin/env python3
"""
Train Random Forest on augmented data using Mixup.
Compares performance with original dataset.
"""

import sys
from pathlib import Path
import pandas as pd

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from load_data import load_wine_data
from random_forest.model import WineQualityRF
from random_forest.utils import split_data, preprocess_features
from synthetic_data_gen.mixup_data_augmentation import MixupAugmentation


def main():
    """Main training pipeline with data augmentation."""
    print("🍷 Wine Quality Prediction - Random Forest with Mixup Augmentation")
    print("=" * 70)
    
    # Load data
    print("\n📂 Loading data...")
    df = load_wine_data()
    print(f"   Loaded {len(df)} samples with {df.shape[1]} features")
    
    # Split BEFORE augmentation (using original data)
    print("\n🔀 Splitting data hierarchically (using original data)...")
    X_train, X_val, X_test, y_train, y_val, y_test = split_data(df)
    
    # Preprocess
    print("\n🧹 Preprocessing features...")
    X_train, X_val, X_test = preprocess_features(X_train, X_val, X_test)
    print("   ✅ Preprocessing complete")
    
    # ==================== TRAIN ON ORIGINAL DATA ====================
    print("\n" + "=" * 70)
    print("📊 MODEL 1: ORIGINAL DATA (No Augmentation)")
    print("=" * 70)
    
    model_original = WineQualityRF(
        n_estimators=100,
        max_depth=20,
        min_samples_split=5,
        min_samples_leaf=2,
        random_state=42
    )
    
    model_original.train(X_train, y_train)
    
    print("\n📈 EVALUATION - Original Data:")
    train_metrics_orig = model_original.evaluate(X_train, y_train, "Training")
    val_metrics_orig = model_original.evaluate(X_val, y_val, "Validation")
    test_metrics_orig = model_original.evaluate(X_test, y_test, "Test (Unseen)")
    
    # ==================== AUGMENT DATA ====================
    print("\n" + "=" * 70)
    print("🔄 AUGMENTING TRAINING DATA WITH MIXUP")
    print("=" * 70)
    
    augmentor = MixupAugmentation(random_state=42)
    
    # Augment training set by 1x (double the size)
    X_train_aug, y_train_aug = augmentor.augment_dataset(
        X_train, y_train,
        augmentation_factor=1.0
    )
    
    # Validate synthetic data
    augmentor.validate_synthetic_data(X_train, X_train_aug[len(X_train):])
    
    # ==================== TRAIN ON AUGMENTED DATA ====================
    print("\n" + "=" * 70)
    print("📊 MODEL 2: AUGMENTED DATA (1x Mixup)")
    print("=" * 70)
    
    model_augmented = WineQualityRF(
        n_estimators=100,
        max_depth=20,
        min_samples_split=5,
        min_samples_leaf=2,
        random_state=42
    )
    
    model_augmented.train(X_train_aug, y_train_aug)
    
    print("\n📈 EVALUATION - Augmented Data:")
    train_metrics_aug = model_augmented.evaluate(X_train_aug, y_train_aug, "Training (Augmented)")
    val_metrics_aug = model_augmented.evaluate(X_val, y_val, "Validation")
    test_metrics_aug = model_augmented.evaluate(X_test, y_test, "Test (Unseen)")
    
    # ==================== COMPARISON ====================
    print("\n" + "=" * 70)
    print("📋 COMPARISON: Original vs Augmented")
    print("=" * 70)
    
    print("\n🎯 TEST SET PERFORMANCE (Unseen Data):")
    print(f"\n{'Metric':<15} {'Original':<15} {'Augmented':<15} {'Change':<15}")
    print("-" * 60)
    
    metrics = ["RMSE", "MAE", "R2"]
    for metric in metrics:
        orig_val = test_metrics_orig[metric]
        aug_val = test_metrics_aug[metric]
        
        if metric == "R2":
            change = aug_val - orig_val
            change_pct = (change / orig_val * 100) if orig_val != 0 else 0
        else:
            change = orig_val - aug_val  # Lower is better for RMSE/MAE
            change_pct = (change / orig_val * 100) if orig_val != 0 else 0
        
        arrow = "↑" if change_pct > 0 else "↓" if change_pct < 0 else "="
        
        print(f"{metric:<15} {orig_val:<15.4f} {aug_val:<15.4f} {arrow} {abs(change_pct):.1f}%")
    
    print("\n" + "=" * 70)
    print("📊 OVERFITTING ANALYSIS:")
    print("=" * 70)
    
    overfit_orig = test_metrics_orig["RMSE"] - train_metrics_orig["RMSE"]
    overfit_aug = test_metrics_aug["RMSE"] - train_metrics_aug["RMSE"]
    
    print(f"\nOriginal Model:")
    print(f"   Test-Train RMSE diff: {overfit_orig:.4f}")
    print(f"   Status: {'⚠️  Overfitting' if overfit_orig > 0.2 else '✅ Good'}")
    
    print(f"\nAugmented Model:")
    print(f"   Test-Train RMSE diff: {overfit_aug:.4f}")
    print(f"   Status: {'⚠️  Overfitting' if overfit_aug > 0.2 else '✅ Good'}")
    
    # ==================== FEATURE IMPORTANCE ====================
    print("\n" + "=" * 70)
    print("🎯 FEATURE IMPORTANCE (Top 10)")
    print("=" * 70)
    
    print("\nOriginal Model:")
    model_original.get_feature_importance(X_train.columns, top_n=10)
    
    print("\nAugmented Model:")
    model_augmented.get_feature_importance(X_train_aug.columns[:len(X_train.columns)], top_n=10)
    
    # ==================== SUMMARY ====================
    print("\n" + "=" * 70)
    print("📝 SUMMARY & RECOMMENDATIONS")
    print("=" * 70)
    
    improvement = test_metrics_aug["R2"] - test_metrics_orig["R2"]
    
    print(f"\n✅ Test R² improvement: {improvement:+.4f}")
    
    if improvement > 0.05:
        print("   🎉 Significant improvement! Augmentation is beneficial.")
        print("   💡 Consider increasing augmentation factor (2x or higher)")
    elif improvement > 0:
        print("   ✓ Slight improvement. Augmentation helps.")
    else:
        print("   ⚠️  No improvement or slight decrease.")
        print("   💡 Try: different augmentation factors, different alpha distributions")
    
    print(f"\n📊 Final Test Set Metrics (Augmented):")
    print(f"   RMSE: {test_metrics_aug['RMSE']:.4f}")
    print(f"   MAE:  {test_metrics_aug['MAE']:.4f}")
    print(f"   R²:   {test_metrics_aug['R2']:.4f}")


if __name__ == "__main__":
    main()
