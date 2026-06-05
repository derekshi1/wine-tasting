#!/usr/bin/env python3
"""
Train Random Forest on augmented data using Mixup.
With robust strategies to improve test set generalization:
  1. Drop spurious 'Id' feature (row index, leaks nothing)
  2. Winsorize extreme values in heavy-tailed features
  3. Add interaction features (alcohol/density, citric_acid/volatile_acidity)
  4. Moderate regularization (not over-pruned)
  5. Augment TRAINING and VALIDATION sets to 3x their original sizes
  6. Hyperparameter tuning evaluated on augmented validation set
"""

import sys
import os
from pathlib import Path
import pandas as pd
import numpy as np

os.environ.setdefault("MPLCONFIGDIR", str(Path(__file__).parent / "plots" / ".matplotlib_cache"))
import matplotlib.pyplot as plt

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from load_data import load_wine_data
from random_forest.model import WineQualityRF
from random_forest.utils import split_data
from synthetic_data_gen.mixup_data_augmentation import MixupAugmentation

from sklearn.model_selection import RandomizedSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score


TARGET_AUGMENTED_SIZE_MULTIPLIER = 3.0
MIXUP_SYNTHETIC_FACTOR = TARGET_AUGMENTED_SIZE_MULTIPLIER - 1.0
PLOTS_DIR = Path(__file__).parent / "plots" / "model_validation"


def prepare_plot_dir() -> Path:
    """Create the model validation plot directory."""
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    return PLOTS_DIR


def save_tuning_validation_plots(tuning_results: pd.DataFrame, output_dir: Path) -> None:
    """
    Save readable plots that show validation performance across tuning candidates.
    """
    if tuning_results.empty or "validation_r2" not in tuning_results.columns:
        return

    plot_df = tuning_results.copy()
    plot_df = plot_df.sort_values("validation_r2", ascending=False).reset_index(drop=True)
    plot_df["rank"] = np.arange(1, len(plot_df) + 1)
    plot_df["max_depth_label"] = plot_df["param_max_depth"].fillna("None").astype(str)
    top_df = plot_df.head(15).iloc[::-1]

    plt.style.use("seaborn-v0_8-whitegrid")

    fig, ax = plt.subplots(figsize=(11, 7))
    bars = ax.barh(
        top_df["rank"].astype(str),
        top_df["validation_r2"],
        color="#3b82f6",
        edgecolor="#1e3a8a",
        linewidth=0.8,
    )
    best_score = plot_df.loc[0, "validation_r2"]
    ax.axvline(best_score, color="#111827", linestyle="--", linewidth=1.3, label=f"Best R2 = {best_score:.3f}")
    ax.set_title("Top Random Search Candidates by Validation R2", fontsize=16, weight="bold", pad=14)
    ax.set_xlabel("Validation R2 on Augmented Validation Data", fontsize=12)
    ax.set_ylabel("Candidate Rank", fontsize=12)
    ax.legend(loc="lower right", frameon=True)
    ax.tick_params(axis="both", labelsize=10)

    for bar, (_, row) in zip(bars, top_df.iterrows()):
        label = (
            f"trees={int(row['param_n_estimators'])}, "
            f"depth={row['max_depth_label']}, "
            f"leaf={int(row['param_min_samples_leaf'])}"
        )
        ax.text(
            bar.get_width() + 0.003,
            bar.get_y() + bar.get_height() / 2,
            label,
            va="center",
            fontsize=9,
            color="#111827",
        )

    ax.set_xlim(max(0, top_df["validation_r2"].min() - 0.04), min(1, top_df["validation_r2"].max() + 0.12))
    fig.tight_layout()
    fig.savefig(output_dir / "tuning_top_validation_r2.png", dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(9, 7))
    scatter = ax.scatter(
        plot_df["mean_test_score"],
        plot_df["validation_r2"],
        c=plot_df["param_min_samples_leaf"].astype(float),
        s=85,
        alpha=0.82,
        cmap="viridis",
        edgecolor="#111827",
        linewidth=0.4,
    )
    ax.scatter(
        plot_df.loc[0, "mean_test_score"],
        plot_df.loc[0, "validation_r2"],
        marker="*",
        s=260,
        color="#ef4444",
        edgecolor="#111827",
        linewidth=0.8,
        label="Selected best",
        zorder=3,
    )
    ax.set_title("Cross-Validation Score vs Validation Score", fontsize=16, weight="bold", pad=14)
    ax.set_xlabel("Mean 5-Fold CV R2", fontsize=12)
    ax.set_ylabel("Validation R2", fontsize=12)
    ax.tick_params(axis="both", labelsize=10)
    ax.legend(frameon=True)
    colorbar = fig.colorbar(scatter, ax=ax)
    colorbar.set_label("min_samples_leaf", fontsize=11)
    fig.tight_layout()
    fig.savefig(output_dir / "tuning_cv_vs_validation_r2.png", dpi=180)
    plt.close(fig)

    summary = (
        plot_df.groupby("max_depth_label", dropna=False)["validation_r2"]
        .agg(["mean", "max", "count"])
        .reset_index()
    )
    summary["depth_order"] = summary["max_depth_label"].replace({"None": 999}).astype(float)
    summary = summary.sort_values("depth_order")

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(summary["max_depth_label"], summary["mean"], marker="o", linewidth=2.2, label="Mean validation R2")
    ax.plot(summary["max_depth_label"], summary["max"], marker="s", linewidth=2.2, label="Best validation R2")
    ax.set_title("Validation Performance by max_depth", fontsize=16, weight="bold", pad=14)
    ax.set_xlabel("max_depth", fontsize=12)
    ax.set_ylabel("Validation R2", fontsize=12)
    ax.tick_params(axis="both", labelsize=10)
    ax.legend(frameon=True)
    fig.tight_layout()
    fig.savefig(output_dir / "tuning_validation_r2_by_max_depth.png", dpi=180)
    plt.close(fig)


def evaluate_pruning_curve(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_val: pd.DataFrame,
    y_val: pd.Series,
    output_dir: Path,
) -> pd.DataFrame:
    """
    Train a small cost-complexity pruning sweep and plot validation performance.
    """
    ccp_alphas = [0.0, 0.0001, 0.00025, 0.0005, 0.001, 0.002, 0.005, 0.01]
    rows = []

    print("\n🌳 Evaluating pruning curve on original validation data...")
    for alpha in ccp_alphas:
        model = RandomForestRegressor(
            n_estimators=200,
            max_depth=10,
            min_samples_split=2,
            min_samples_leaf=1,
            max_features="log2",
            min_impurity_decrease=0.001,
            ccp_alpha=alpha,
            random_state=42,
            n_jobs=-1,
        )
        model.fit(X_train, y_train)
        pred = model.predict(X_val)
        rows.append({
            "ccp_alpha": alpha,
            "validation_rmse": np.sqrt(mean_squared_error(y_val, pred)),
            "validation_mae": mean_absolute_error(y_val, pred),
            "validation_r2": r2_score(y_val, pred),
            "avg_tree_depth": np.mean([tree.get_depth() for tree in model.estimators_]),
            "avg_leaves": np.mean([tree.get_n_leaves() for tree in model.estimators_]),
        })

    pruning_df = pd.DataFrame(rows)
    pruning_df.to_csv(output_dir / "pruning_validation_curve.csv", index=False)

    best_idx = pruning_df["validation_r2"].idxmax()
    best_row = pruning_df.loc[best_idx]

    plt.style.use("seaborn-v0_8-whitegrid")
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.6))

    axes[0].plot(pruning_df["ccp_alpha"], pruning_df["validation_r2"], marker="o", linewidth=2.2, color="#2563eb")
    axes[0].scatter(best_row["ccp_alpha"], best_row["validation_r2"], s=130, color="#ef4444", zorder=3, label="Best")
    axes[0].set_title("Validation R2 vs Pruning Strength", fontsize=14, weight="bold", pad=12)
    axes[0].set_xlabel("ccp_alpha", fontsize=11)
    axes[0].set_ylabel("Validation R2", fontsize=11)
    axes[0].legend(frameon=True)

    axes[1].plot(pruning_df["ccp_alpha"], pruning_df["validation_rmse"], marker="o", linewidth=2.2, color="#059669")
    axes[1].scatter(best_row["ccp_alpha"], best_row["validation_rmse"], s=130, color="#ef4444", zorder=3, label="Best R2")
    axes[1].set_title("Validation RMSE vs Pruning Strength", fontsize=14, weight="bold", pad=12)
    axes[1].set_xlabel("ccp_alpha", fontsize=11)
    axes[1].set_ylabel("Validation RMSE", fontsize=11)
    axes[1].legend(frameon=True)

    for ax in axes:
        ax.tick_params(axis="both", labelsize=10)
        ax.ticklabel_format(axis="x", style="plain")

    fig.suptitle("Cost-Complexity Pruning Curve on Validation Data", fontsize=16, weight="bold", y=1.03)
    fig.tight_layout()
    fig.savefig(output_dir / "pruning_validation_curve.png", dpi=180, bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(pruning_df["avg_tree_depth"], pruning_df["validation_r2"], marker="o", linewidth=2.2, color="#7c3aed")
    for _, row in pruning_df.iterrows():
        ax.annotate(
            f"{row['ccp_alpha']:.4g}",
            (row["avg_tree_depth"], row["validation_r2"]),
            textcoords="offset points",
            xytext=(6, 5),
            fontsize=9,
        )
    ax.set_title("Validation R2 as Trees Become More Pruned", fontsize=16, weight="bold", pad=14)
    ax.set_xlabel("Average Tree Depth", fontsize=12)
    ax.set_ylabel("Validation R2", fontsize=12)
    ax.tick_params(axis="both", labelsize=10)
    fig.tight_layout()
    fig.savefig(output_dir / "pruning_tree_depth_vs_validation_r2.png", dpi=180)
    plt.close(fig)

    return pruning_df


def save_validation_metric_comparison(
    original_metrics: dict,
    augmented_metrics: dict,
    holdout_val_metrics: dict,
    output_dir: Path,
) -> None:
    """Plot validation RMSE/MAE/R2 for the main model variants."""
    comparison_df = pd.DataFrame([
        {"model": "Pruned original RF", **original_metrics},
        {"model": "Tuned augmented RF", **augmented_metrics},
        {"model": "Tuned RF on original holdout", **holdout_val_metrics},
    ])[["model", "RMSE", "MAE", "R2"]]
    comparison_df.to_csv(output_dir / "validation_model_comparison.csv", index=False)

    fig, axes = plt.subplots(1, 3, figsize=(14, 5.2))
    metrics = [("RMSE", "lower is better"), ("MAE", "lower is better"), ("R2", "higher is better")]
    colors = ["#2563eb", "#059669", "#f59e0b"]

    for ax, (metric, subtitle) in zip(axes, metrics):
        ax.bar(comparison_df["model"], comparison_df[metric], color=colors, edgecolor="#111827", linewidth=0.6)
        ax.set_title(f"{metric} ({subtitle})", fontsize=13, weight="bold", pad=10)
        ax.set_ylabel(metric, fontsize=11)
        ax.tick_params(axis="x", rotation=25, labelsize=9)
        ax.tick_params(axis="y", labelsize=10)
        for label in ax.get_xticklabels():
            label.set_ha("right")
        for idx, value in enumerate(comparison_df[metric]):
            ax.text(idx, value, f"{value:.3f}", ha="center", va="bottom", fontsize=9)

    fig.suptitle("Validation Performance of Main Random Forest Variants", fontsize=16, weight="bold", y=1.03)
    fig.tight_layout()
    fig.savefig(output_dir / "validation_model_comparison.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


def drop_spurious_features(X: pd.DataFrame) -> pd.DataFrame:
    """
    Drop columns that have no predictive value.
    - 'Id' is just a row index, not a real feature.
    """
    cols_to_drop = [col for col in ['Id'] if col in X.columns]
    if cols_to_drop:
        print(f"   Dropping spurious columns: {cols_to_drop}")
        return X.drop(columns=cols_to_drop)
    return X


def winsorize_feature(X: pd.DataFrame, col: str, lower_q: float = 0.01, upper_q: float = 0.99) -> pd.Series:
    """
    Clip extreme values in a feature to reduce influence of outliers.
    Heavy-tailed features (chlorides, residual sugar, sulfur dioxides) can
    cause trees to split on outlier values, hurting generalization.
    """
    low = X[col].quantile(lower_q)
    high = X[col].quantile(upper_q)
    return X[col].clip(low, high)


def add_interaction_features(X: pd.DataFrame) -> pd.DataFrame:
    """
    Add domain-relevant interaction features that capture relationships
    known to be important for wine quality.
    """
    X = X.copy()
    
    # alcohol / density: higher alcohol generally lowers density, this ratio
    # captures the 'body' of the wine more directly than either alone
    if 'alcohol' in X.columns and 'density' in X.columns:
        X['alcohol_to_density'] = X['alcohol'] / X['density']
    
    # citric_acid / volatile_acidity: freshness vs vinegar character
    if 'citric acid' in X.columns and 'volatile acidity' in X.columns:
        # Add small epsilon to avoid division by zero
        X['acid_balance'] = X['citric acid'] / (X['volatile acidity'] + 1e-6)
    
    # free + bound sulfur dioxide (bound = total - free)
    if 'free sulfur dioxide' in X.columns and 'total sulfur dioxide' in X.columns:
        X['bound_sulfur_dioxide'] = X['total sulfur dioxide'] - X['free sulfur dioxide']
    
    # sulphates * alcohol: multiplicative preservative effect
    if 'sulphates' in X.columns and 'alcohol' in X.columns:
        X['sulphates_times_alcohol'] = X['sulphates'] * X['alcohol']
    
    original_cols = {'fixed acidity', 'volatile acidity', 'citric acid', 'residual sugar',
                     'chlorides', 'free sulfur dioxide', 'total sulfur dioxide', 'density',
                     'pH', 'sulphates', 'alcohol', 'quality', 'Id'}
    new_features = [c for c in X.columns if c not in original_cols]
    print(f"   Added interaction features: {new_features}")
    return X


def preprocess_training_pipeline(X_train: pd.DataFrame, X_val: pd.DataFrame, X_test: pd.DataFrame):
    """
    Full preprocessing pipeline for train/val/test:
    1. Drop spurious features
    2. Winsorize heavy-tailed features
    3. Add interaction features
    
    Fits on training data, transforms all sets consistently.
    """
    # Drop spurious columns
    X_train = drop_spurious_features(X_train)
    X_val = drop_spurious_features(X_val)
    X_test = drop_spurious_features(X_test)
    
    print(f"\n   Features after dropping spurious: {list(X_train.columns)}")
    
    # Winsorize heavy-tailed features (fit on train, transform all)
    heavy_tailed_cols = ['chlorides', 'residual sugar', 'free sulfur dioxide', 'total sulfur dioxide']
    for col in heavy_tailed_cols:
        if col in X_train.columns:
            lower = X_train[col].quantile(0.01)
            upper = X_train[col].quantile(0.99)
            X_train[col] = X_train[col].clip(lower, upper)
            X_val[col] = X_val[col].clip(lower, upper)
            X_test[col] = X_test[col].clip(lower, upper)
    
    print(f"   Winsorized heavy-tailed features: {[c for c in heavy_tailed_cols if c in X_train.columns]}")
    
    # Add interaction features
    X_train = add_interaction_features(X_train)
    X_val = add_interaction_features(X_val)
    X_test = add_interaction_features(X_test)
    
    # Handle any remaining NaNs from edge case in interactions
    X_train = X_train.fillna(0)
    X_val = X_val.fillna(0)
    X_test = X_test.fillna(0)
    
    return X_train, X_val, X_test


def tune_hyperparameters_with_augmented_val(
    X_train: pd.DataFrame, y_train: pd.Series,
    X_val: pd.DataFrame, y_val: pd.Series,
    n_iter: int = 75
):
    """
    Hyperparameter tuning using RandomizedSearchCV with 5-fold CV on AUGMENTED data.
    Evaluates on augmented validation set to select best params.
    This gives more robust parameter selection than just CV on training.
    """
    print("\n" + "=" * 70)
    print("🔍 HYPERPARAMETER TUNING (on augmented training + augmented validation)")
    print("=" * 70)
    
    param_dist = {
        'n_estimators': [100, 150, 200, 250, 300],
        'max_depth': [8, 10, 12, 15, 18, 20, None],
        'min_samples_split': [2, 3, 5, 7, 10],
        'min_samples_leaf': [1, 2, 3, 4, 5],
        'max_features': ['sqrt', 'log2', 0.3, 0.5],
        'bootstrap': [True, False],
    }
    
    print(f"📊 Parameter Space: {n_iter} combinations via Random Search")
    print(f"   Training set size: {len(X_train)} | Validation set size: {len(X_val)}")
    
    rf = RandomForestRegressor(random_state=42, n_jobs=-1)
    
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
    
    print(f"\n🚀 Fitting on augmented training data ({len(X_train)} samples)...")
    random_search.fit(X_train, y_train)
    
    # Refit: evaluate each CV-best model on augmented validation set
    print(f"\n🔄 Selecting best via augmented validation set ({len(X_val)} samples)...")
    
    val_r2_scores = []
    val_rmse_scores = []
    val_mae_scores = []
    for candidate_params in random_search.cv_results_['params']:
        m = RandomForestRegressor(**candidate_params, random_state=42, n_jobs=-1)
        m.fit(X_train, y_train)
        val_pred = m.predict(X_val)
        val_r2_scores.append(r2_score(y_val, val_pred))
        val_rmse_scores.append(np.sqrt(mean_squared_error(y_val, val_pred)))
        val_mae_scores.append(mean_absolute_error(y_val, val_pred))
    
    # Find best params by validation R²
    best_idx = np.argmax(val_r2_scores)
    best_val_r2 = val_r2_scores[best_idx]
    best_val_params = random_search.cv_results_['params'][best_idx]
    
    # Train final model with best params
    best_model = RandomForestRegressor(**best_val_params, random_state=42, n_jobs=-1)
    best_model.fit(X_train, y_train)
    
    print("\n" + "=" * 70)
    print("✅ TUNING COMPLETE")
    print("=" * 70)
    print(f"\n🏆 Best Parameters (by augmented validation R² = {best_val_r2:.4f}):")
    for param, value in sorted(best_val_params.items()):
        print(f"   {param:<25} = {value}")
    
    tuning_results = pd.DataFrame(random_search.cv_results_)
    tuning_results["validation_r2"] = val_r2_scores
    tuning_results["validation_rmse"] = val_rmse_scores
    tuning_results["validation_mae"] = val_mae_scores
    tuning_results = tuning_results.sort_values("validation_r2", ascending=False).reset_index(drop=True)

    return best_model, best_val_params, tuning_results


def augment_split(
    X: pd.DataFrame,
    y: pd.Series,
    split_name: str,
    random_state: int
) -> tuple[pd.DataFrame, pd.Series]:
    """
    Augment one data split to 3x its original size using Mixup.
    """
    print("\n" + "=" * 70)
    print(f"🔄 AUGMENTING {split_name.upper()} DATA WITH MIXUP")
    print("=" * 70)
    print(
        f"   Target size: {TARGET_AUGMENTED_SIZE_MULTIPLIER:.1f}x "
        f"({MIXUP_SYNTHETIC_FACTOR:.1f}x synthetic + original)"
    )
    
    augmentor = MixupAugmentation(random_state=random_state)
    return augmentor.augment_dataset(
        X,
        y,
        augmentation_factor=MIXUP_SYNTHETIC_FACTOR
    )


def evaluate_model(model, X_train, y_train, X_val, y_val, X_test, y_test, model_name="Model"):
    """
    Evaluate model on all three sets.
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


def main():
    """Main training pipeline with advanced strategies for test set generalization."""
    print("🍷 Wine Quality Prediction - Random Forest with Advanced Regularization")
    print("=" * 70)
    output_dir = prepare_plot_dir()
    print(f"\n📁 Validation plots will be saved to: {output_dir}")
    
    # ==================== LOAD DATA ====================
    print("\n📂 Loading data...")
    df = load_wine_data()
    print(f"   Loaded {len(df)} samples with {df.shape[1]} features")
    
    # ==================== SPLIT ====================
    print("\n🔀 Splitting data hierarchically...")
    X_train, X_val, X_test, y_train, y_val, y_test = split_data(df)
    
    # ==================== ADVANCED PREPROCESSING ====================
    print("\n🔧 Advanced Feature Engineering:")
    X_train, X_val, X_test = preprocess_training_pipeline(X_train, X_val, X_test)
    
    # ==================== MODEL 1: ORIGINAL DATA (Enhanced) ====================
    print("\n" + "=" * 70)
    print("📊 MODEL 1: ORIGINAL DATA (Feature-Engineered, Moderately Pruned)")
    print("=" * 70)
    
    model_original = WineQualityRF(
        n_estimators=300,
        max_depth=15,
        min_samples_split=5,
        min_samples_leaf=3,
        max_features='sqrt',
        min_impurity_decrease=0.001,
        ccp_alpha=0.0005,
        random_state=42
    )
    model_original.train(X_train, y_train)
    
    print("\n📈 EVALUATION - Original Data (Enhanced):")
    train_metrics_orig = model_original.evaluate(X_train, y_train, "Training")
    val_metrics_orig = model_original.evaluate(X_val, y_val, "Validation")
    test_metrics_orig = model_original.evaluate(X_test, y_test, "Test (Unseen)")

    pruning_df = evaluate_pruning_curve(X_train, y_train, X_val, y_val, output_dir)
    best_pruning_row = pruning_df.loc[pruning_df["validation_r2"].idxmax()]
    print(
        f"   Best pruning alpha on validation: {best_pruning_row['ccp_alpha']:.4g} "
        f"(R2={best_pruning_row['validation_r2']:.4f}, "
        f"RMSE={best_pruning_row['validation_rmse']:.4f})"
    )
    
    # ==================== AUGMENT DATA ====================
    X_train_aug, y_train_aug = augment_split(
        X_train,
        y_train,
        split_name="training",
        random_state=42
    )
    X_val_aug, y_val_aug = augment_split(
        X_val,
        y_val,
        split_name="validation",
        random_state=43
    )
    
    # ==================== HYPERPARAMETER TUNING on AUGMENTED DATA ====================
    tuned_model, best_params, tuning_results = tune_hyperparameters_with_augmented_val(
        X_train_aug, y_train_aug,
        X_val_aug, y_val_aug,
        n_iter=75
    )
    tuning_results.to_csv(output_dir / "tuning_validation_results.csv", index=False)
    save_tuning_validation_plots(tuning_results, output_dir)
    
    # ==================== MODEL 2: AUGMENTED DATA (Tuned) ====================
    print("\n" + "=" * 70)
    print("📊 MODEL 2: AUGMENTED DATA (Tuned Hyperparameters)")
    print("=" * 70)
    
    print("\n📈 EVALUATION - Augmented Data (Tuned):")
    train_metrics_aug = evaluate_model(
        tuned_model, X_train_aug, y_train_aug, X_val_aug, y_val_aug, X_test, y_test,
        "Augmented (Tuned)"
    )
    
    holdout_val_pred = tuned_model.predict(X_val)
    holdout_val_metrics = {
        "RMSE": np.sqrt(mean_squared_error(y_val, holdout_val_pred)),
        "MAE": mean_absolute_error(y_val, holdout_val_pred),
        "R2": r2_score(y_val, holdout_val_pred),
    }
    
    print("\nValidation (Original Holdout Portion):")
    print(f"   RMSE: {holdout_val_metrics['RMSE']:.4f}")
    print(f"   MAE:  {holdout_val_metrics['MAE']:.4f}")
    print(f"   R²:   {holdout_val_metrics['R2']:.4f}")

    save_validation_metric_comparison(
        val_metrics_orig,
        train_metrics_aug["Validation"],
        holdout_val_metrics,
        output_dir,
    )
    
    # ==================== COMPARISON ====================
    print("\n" + "=" * 70)
    print("📋 HELD-OUT TEST COMPARISON: Original vs 3x Augmented")
    print("=" * 70)
    
    print("\nHeld-out test data was never augmented or used for tuning.")
    print(f"\n{'Metric':<15} {'Original':<15} {'3x Augmented':<15} {'Change':<15}")
    print("-" * 60)
    
    for metric in ["RMSE", "MAE", "R2"]:
        orig_val = test_metrics_orig[metric]
        aug_val = train_metrics_aug["Test (Unseen)"][metric]
        
        if metric == "R2":
            change = aug_val - orig_val
            change_pct = (change / orig_val * 100) if orig_val != 0 else 0
        else:
            change = orig_val - aug_val
            change_pct = (change / orig_val * 100) if orig_val != 0 else 0
        
        arrow = "↑" if change_pct > 0 else "↓" if change_pct < 0 else "="
        print(f"{metric:<15} {orig_val:<15.4f} {aug_val:<15.4f} {arrow} {abs(change_pct):.1f}%")
    
    # ==================== OVERFITTING ANALYSIS ====================
    print("\n" + "=" * 70)
    print("📊 OVERFITTING ANALYSIS:")
    print("=" * 70)
    
    overfit_orig = test_metrics_orig["RMSE"] - train_metrics_orig["RMSE"]
    overfit_aug = train_metrics_aug["Test (Unseen)"]["RMSE"] - train_metrics_aug["Training"]["RMSE"]
    
    for name, diff in [("Original", overfit_orig), ("Augmented", overfit_aug)]:
        print(f"\n{name} Model:")
        print(f"   Test-Train RMSE diff: {diff:.4f}")
        if diff > 0.3:
            print(f"   Status: ⚠️  Overfitting (diff > 0.3)")
        elif diff > 0.15:
            print(f"   Status: ⚠️  Mild overfitting")
        else:
            print(f"   Status: ✅ Good generalization")
    
    # ==================== TREE COMPLEXITY ====================
    print("\n" + "=" * 70)
    print("🌳 TREE COMPLEXITY ANALYSIS")
    print("=" * 70)
    
    for name, model in [("Original", model_original.model), ("Augmented (Tuned)", tuned_model)]:
        depths = [tree.get_depth() for tree in model.estimators_]
        leaves = [tree.get_n_leaves() for tree in model.estimators_]
        avg_depth = np.mean(depths)
        avg_leaves = np.mean(leaves)
        print(f"\n{name}:")
        print(f"   Avg tree depth:       {avg_depth:.1f} (max: {max(depths)})")
        print(f"   Avg leaves per tree:  {avg_leaves:.0f}")
    
    # ==================== FEATURE IMPORTANCE ====================
    print("\n" + "=" * 70)
    print("🎯 FEATURE IMPORTANCE (Top 10) - Tuned Model")
    print("=" * 70)
    
    importances = tuned_model.feature_importances_
    feature_names = X_train_aug.columns[:len(X_train_aug.columns)]
    feature_importance_df = pd.DataFrame({
        "feature": feature_names,
        "importance": importances
    }).sort_values("importance", ascending=False)
    
    print(f"\n{'Feature':<30} {'Importance':<15}")
    print("-" * 45)
    for idx, row in feature_importance_df.head(10).iterrows():
        print(f"{row['feature']:<30} {row['importance']:<15.4f}")
    
    # ==================== SUMMARY ====================
    print("\n" + "=" * 70)
    print("📝 FINAL SUMMARY")
    print("=" * 70)
    
    test_r2 = train_metrics_aug["Test (Unseen)"]["R2"]
    test_rmse = train_metrics_aug["Test (Unseen)"]["RMSE"]
    
    print(f"\n🎯 Best Test Performance:")
    print(f"   RMSE: {test_rmse:.4f}")
    print(f"   R²:   {test_r2:.4f}")
    
    print(f"\n🔧 Key Strategies Applied:")
    print(f"   1. ✅ Dropped spurious 'Id' feature")
    print(f"   2. ✅ Winsorized heavy-tailed features (chlorides, residual sugar, SO₂)")
    print(f"   3. ✅ Added interaction features (alcohol/density, acid balance, etc.)")
    print(f"   4. ✅ Augmented training set to {TARGET_AUGMENTED_SIZE_MULTIPLIER:.1f}x for more data diversity")
    print(f"   5. ✅ Augmented validation set to {TARGET_AUGMENTED_SIZE_MULTIPLIER:.1f}x for model selection")
    print(f"   6. ✅ Moderate regularization (not over-pruned)")
    
    improvement_vs_baseline = test_r2 - test_metrics_orig["R2"]
    print(f"\n📊 R² Improvement over Original (no-augmentation) model: {improvement_vs_baseline:+.4f}")
    
    if improvement_vs_baseline > 0.02:
        print("   🎉 Significant improvement!")
    elif improvement_vs_baseline > 0:
        print("   ✓ Improvement detected")
    else:
        print("   ⚠️ Slight decrease. Consider further tuning.")

    print(f"\n📁 Saved validation plots and CSV summaries to: {output_dir}")


if __name__ == "__main__":
    main()
