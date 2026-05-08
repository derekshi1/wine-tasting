#!/usr/bin/env python3
"""
Main training script for Random Forest wine quality regressor.
Uses hierarchical cross-validation with 60% train, 20% validation, 20% test.
"""

import sys
from pathlib import Path
import pandas as pd

# Add parent directory to path to import load_data
sys.path.insert(0, str(Path(__file__).parent.parent))

from load_data import load_wine_data
from model import WineQualityRF
from utils import split_data, preprocess_features


def main():
    """Main training pipeline."""
    print("🍷 Wine Quality Prediction - Random Forest Regressor")
    print("=" * 60)
    
    # Load data
    print("\n📂 Loading data...")
    df = load_wine_data()
    print(f"   Loaded {len(df)} samples with {df.shape[1]} features")
    
    # Split data hierarchically
    print("\n🔀 Splitting data hierarchically...")
    X_train, X_val, X_test, y_train, y_val, y_test = split_data(df)
    
    # Preprocess
    print("\n🧹 Preprocessing features...")
    X_train, X_val, X_test = preprocess_features(X_train, X_val, X_test)
    print("   ✅ Preprocessing complete")
    
    # Initialize and train model
    print("\n🤖 Initializing Random Forest model...")
    model = WineQualityRF(
        n_estimators=100,
        max_depth=20,
        min_samples_split=5,
        min_samples_leaf=2,
        random_state=42
    )
    
    # Train on training set
    model.train(X_train, y_train)
    
    # Evaluate on all sets
    print("\n" + "=" * 60)
    print("📈 EVALUATION RESULTS")
    print("=" * 60)
    
    train_metrics = model.evaluate(X_train, y_train, "Training")
    val_metrics = model.evaluate(X_val, y_val, "Validation")
    test_metrics = model.evaluate(X_test, y_test, "Test (Unseen)")
    
    # Feature importance
    print("\n" + "=" * 60)
    feature_importance = model.get_feature_importance(X_train.columns, top_n=10)
    
    # Summary
    print("\n" + "=" * 60)
    print("📋 SUMMARY")
    print("=" * 60)
    print(f"\n✅ Training complete!")
    print(f"   Training RMSE:   {train_metrics['RMSE']:.4f}")
    print(f"   Validation RMSE: {val_metrics['RMSE']:.4f}")
    print(f"   Test RMSE:       {test_metrics['RMSE']:.4f}")
    print(f"\n   Training R²:     {train_metrics['R2']:.4f}")
    print(f"   Validation R²:   {val_metrics['R2']:.4f}")
    print(f"   Test R²:         {test_metrics['R2']:.4f}")
    
    # Check for overfitting
    print("\n🔍 Overfitting Analysis:")
    rmse_diff = test_metrics['RMSE'] - train_metrics['RMSE']
    if rmse_diff > 0.2:
        print(f"   ⚠️  Possible overfitting detected (Test-Train RMSE diff: {rmse_diff:.4f})")
    else:
        print(f"   ✅ Good generalization (Test-Train RMSE diff: {rmse_diff:.4f})")
    
    return model, test_metrics


if __name__ == "__main__":
    main()
