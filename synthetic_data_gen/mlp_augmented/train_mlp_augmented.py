#!/usr/bin/env python3
"""
Train a simple MLP on lots of Mixup-augmented wine data.

The validation and test sets are not augmented. They stay as original held-out
data so the reported performance is easy to interpret.
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import Pipeline
from sklearn.compose import TransformedTargetRegressor
from sklearn.preprocessing import StandardScaler

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from load_data import load_wine_data
from random_forest.utils import split_data
from synthetic_data_gen.mixup_data_augmentation import MixupAugmentation


def preprocess_splits(
    X_train: pd.DataFrame,
    X_val: pd.DataFrame,
    X_test: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Preprocess and add a few stable wine-domain interaction features."""
    X_train = X_train.copy()
    X_val = X_val.copy()
    X_test = X_test.copy()

    for X in (X_train, X_val, X_test):
        if "Id" in X.columns:
            X.drop(columns="Id", inplace=True)

    heavy_tailed_cols = [
        "chlorides",
        "residual sugar",
        "free sulfur dioxide",
        "total sulfur dioxide",
    ]
    for col in heavy_tailed_cols:
        if col in X_train.columns:
            lower = X_train[col].quantile(0.01)
            upper = X_train[col].quantile(0.99)
            X_train[col] = X_train[col].clip(lower, upper)
            X_val[col] = X_val[col].clip(lower, upper)
            X_test[col] = X_test[col].clip(lower, upper)

    X_train = add_interaction_features(X_train)
    X_val = add_interaction_features(X_val)
    X_test = add_interaction_features(X_test)

    train_means = X_train.mean(numeric_only=True)
    X_train = X_train.fillna(train_means)
    X_val = X_val.fillna(train_means)
    X_test = X_test.fillna(train_means)

    return X_train, X_val, X_test


def add_interaction_features(X: pd.DataFrame) -> pd.DataFrame:
    """Add the same simple interaction features used in the RF experiment."""
    X = X.copy()

    if "alcohol" in X.columns and "density" in X.columns:
        X["alcohol_to_density"] = X["alcohol"] / X["density"]

    if "citric acid" in X.columns and "volatile acidity" in X.columns:
        X["acid_balance"] = X["citric acid"] / (X["volatile acidity"] + 1e-6)

    if "free sulfur dioxide" in X.columns and "total sulfur dioxide" in X.columns:
        X["bound_sulfur_dioxide"] = X["total sulfur dioxide"] - X["free sulfur dioxide"]

    if "sulphates" in X.columns and "alcohol" in X.columns:
        X["sulphates_times_alcohol"] = X["sulphates"] * X["alcohol"]

    return X


def make_mlp(random_state: int) -> TransformedTargetRegressor:
    """Build a small MLP pipeline with feature and target scaling."""
    feature_model = Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            (
                "mlp",
                MLPRegressor(
                    hidden_layer_sizes=(64, 32),
                    activation="relu",
                    solver="adam",
                    alpha=0.001,
                    batch_size=64,
                    learning_rate="adaptive",
                    learning_rate_init=0.001,
                    max_iter=1000,
                    early_stopping=True,
                    validation_fraction=0.15,
                    n_iter_no_change=30,
                    random_state=random_state,
                ),
            ),
        ]
    )
    return TransformedTargetRegressor(
        regressor=feature_model,
        transformer=StandardScaler(),
    )


def augment_training_data(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    augmentation_factor: float,
    random_state: int,
    augmentation_mode: str,
    noise_scale: float,
) -> tuple[pd.DataFrame, pd.Series]:
    """Create synthetic training rows."""
    if augmentation_factor <= 0:
        return X_train.reset_index(drop=True), y_train.reset_index(drop=True)

    if augmentation_mode == "jitter":
        return jitter_augmentation(
            X_train,
            y_train,
            augmentation_factor=augmentation_factor,
            random_state=random_state,
            noise_scale=noise_scale,
        )

    if augmentation_mode == "random-mixup":
        augmentor = MixupAugmentation(random_state=random_state)
        return augmentor.augment_dataset(
            X_train,
            y_train,
            augmentation_factor=augmentation_factor,
        )

    return quality_aware_mixup(
        X_train,
        y_train,
        augmentation_factor=augmentation_factor,
        random_state=random_state,
    )


def jitter_augmentation(
    X: pd.DataFrame,
    y: pd.Series,
    augmentation_factor: float,
    random_state: int,
    noise_scale: float,
) -> tuple[pd.DataFrame, pd.Series]:
    """
    Add small feature noise to resampled training rows while preserving labels.

    This is less aggressive than Mixup for ordinal wine scores: it expands local
    neighborhoods around real wines without creating synthetic in-between labels.
    """
    rng = np.random.default_rng(random_state)
    X_reset = X.reset_index(drop=True)
    y_reset = y.reset_index(drop=True)
    n_synthetic = int(len(X_reset) * augmentation_factor)

    sampled_indices = rng.integers(0, len(X_reset), size=n_synthetic)
    X_synthetic = X_reset.iloc[sampled_indices].reset_index(drop=True).copy()
    y_synthetic = y_reset.iloc[sampled_indices].reset_index(drop=True).copy()

    feature_std = X_reset.std(axis=0).replace(0, 1.0)
    noise = rng.normal(
        loc=0.0,
        scale=(feature_std * noise_scale).to_numpy(),
        size=X_synthetic.shape,
    )
    X_synthetic = X_synthetic + noise

    feature_min = X_reset.min(axis=0)
    feature_max = X_reset.max(axis=0)
    X_synthetic = X_synthetic.clip(lower=feature_min, upper=feature_max, axis=1)

    X_augmented = pd.concat([X_reset, X_synthetic], ignore_index=True)
    y_augmented = pd.concat([y_reset, y_synthetic], ignore_index=True)

    print("\nDataset Augmentation Summary:")
    print(f"   Mode:           jitter")
    print(f"   Noise scale:    {noise_scale:.3f} feature std")
    print(f"   Original size:  {len(X_reset)}")
    print(f"   Synthetic size: {len(X_synthetic)}")
    print(f"   Augmented size: {len(X_augmented)}")
    print(f"   Growth factor:  {len(X_augmented) / len(X_reset):.2f}x")

    return X_augmented, y_augmented


def quality_aware_mixup(
    X: pd.DataFrame,
    y: pd.Series,
    augmentation_factor: float,
    random_state: int,
) -> tuple[pd.DataFrame, pd.Series]:
    """
    Mix only wines with the same integer quality.

    Random Mixup creates many synthetic labels between classes and can pull the
    MLP toward mean-ish predictions. Same-quality Mixup expands local feature
    neighborhoods while keeping labels on the original rating scale.
    """
    rng = np.random.default_rng(random_state)
    X_reset = X.reset_index(drop=True)
    y_reset = y.reset_index(drop=True)
    n_synthetic = int(len(X_reset) * augmentation_factor)
    quality_values = y_reset.value_counts(normalize=True).sort_index()
    qualities = quality_values.index.to_numpy()
    quality_probs = quality_values.to_numpy()
    grouped_indices = {
        quality: y_reset.index[y_reset == quality].to_numpy()
        for quality in qualities
    }

    print(f"Generating {n_synthetic} quality-aware synthetic samples via Mixup...")

    X_synthetic = []
    y_synthetic = []
    for i in range(n_synthetic):
        quality = rng.choice(qualities, p=quality_probs)
        candidates = grouped_indices[quality]
        idx1, idx2 = rng.choice(candidates, size=2, replace=len(candidates) < 2)
        alpha = rng.uniform(0, 1)

        x_new = alpha * X_reset.iloc[idx1].to_numpy() + (1 - alpha) * X_reset.iloc[idx2].to_numpy()
        X_synthetic.append(x_new)
        y_synthetic.append(float(quality))

        if (i + 1) % max(1, n_synthetic // 10) == 0:
            print(f"   {i + 1}/{n_synthetic} ({(i + 1) / n_synthetic * 100:.0f}%)")

    X_synthetic = pd.DataFrame(X_synthetic, columns=X_reset.columns)
    y_synthetic = pd.Series(y_synthetic, name=y.name)
    X_augmented = pd.concat([X_reset, X_synthetic], ignore_index=True)
    y_augmented = pd.concat([y_reset, y_synthetic], ignore_index=True)

    print("\nDataset Augmentation Summary:")
    print(f"   Original size:  {len(X_reset)}")
    print(f"   Synthetic size: {len(X_synthetic)}")
    print(f"   Augmented size: {len(X_augmented)}")
    print(f"   Growth factor:  {len(X_augmented) / len(X_reset):.2f}x")

    return X_augmented, y_augmented


def evaluate(model: Pipeline, X: pd.DataFrame, y: pd.Series) -> dict[str, float]:
    """Return standard regression metrics."""
    pred = model.predict(X)
    return {
        "RMSE": float(np.sqrt(mean_squared_error(y, pred))),
        "MAE": float(mean_absolute_error(y, pred)),
        "R2": float(r2_score(y, pred)),
    }


def print_metrics(title: str, metrics: dict[str, float]) -> None:
    print(f"\n{title}:")
    print(f"   RMSE: {metrics['RMSE']:.4f}")
    print(f"   MAE:  {metrics['MAE']:.4f}")
    print(f"   R2:   {metrics['R2']:.4f}")


def compare_test_metrics(
    baseline_metrics: dict[str, float],
    augmented_metrics: dict[str, float],
) -> None:
    print("\n" + "=" * 70)
    print("HELD-OUT TEST COMPARISON")
    print("=" * 70)
    print(f"{'Metric':<10} {'Baseline':<12} {'Augmented':<12} {'Change':<12}")
    print("-" * 48)

    for metric in ("RMSE", "MAE", "R2"):
        baseline = baseline_metrics[metric]
        augmented = augmented_metrics[metric]
        if metric == "R2":
            improvement = augmented - baseline
        else:
            improvement = baseline - augmented

        pct = improvement / abs(baseline) * 100 if baseline else 0.0
        direction = "up" if improvement > 0 else "down" if improvement < 0 else "flat"
        print(
            f"{metric:<10} {baseline:<12.4f} {augmented:<12.4f} "
            f"{direction} {abs(pct):.1f}%"
        )


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a simple MLP with Mixup augmentation.")
    parser.add_argument(
        "--augmentation-factor",
        type=float,
        default=9.0,
        help="Synthetic rows to add as a multiple of training size. Default 9.0 gives 10x total training data.",
    )
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument(
        "--augmentation-mode",
        choices=["jitter", "quality-aware-mixup", "random-mixup"],
        default="jitter",
        help="Synthetic data strategy. Jitter is the default because Mixup over-smoothed the MLP.",
    )
    parser.add_argument(
        "--noise-scale",
        type=float,
        default=0.03,
        help="Feature noise std for jitter augmentation as a fraction of each feature std.",
    )
    parser.add_argument(
        "--results-path",
        type=Path,
        default=Path(__file__).resolve().parent / "results.json",
        help="Where to save the validation/test metrics JSON.",
    )
    args = parser.parse_args()

    print("Wine Quality MLP with Mixup Augmentation")
    print("=" * 70)

    df = load_wine_data()
    X_train, X_val, X_test, y_train, y_val, y_test = split_data(
        df,
        random_state=args.random_state,
    )
    X_train, X_val, X_test = preprocess_splits(X_train, X_val, X_test)

    print("\nTraining baseline MLP on original data...")
    baseline_model = make_mlp(random_state=args.random_state)
    baseline_model.fit(X_train, y_train)

    print(f"\nAugmenting training data with {args.augmentation_mode}...")
    X_train_aug, y_train_aug = augment_training_data(
        X_train,
        y_train,
        augmentation_factor=args.augmentation_factor,
        random_state=args.random_state,
        augmentation_mode=args.augmentation_mode,
        noise_scale=args.noise_scale,
    )

    total_factor = len(X_train_aug) / len(X_train)
    print(f"\nTraining augmented MLP on {len(X_train_aug)} rows ({total_factor:.1f}x total)...")
    augmented_model = make_mlp(random_state=args.random_state)
    augmented_model.fit(X_train_aug, y_train_aug)

    baseline_val = evaluate(baseline_model, X_val, y_val)
    baseline_test = evaluate(baseline_model, X_test, y_test)
    augmented_val = evaluate(augmented_model, X_val, y_val)
    augmented_test = evaluate(augmented_model, X_test, y_test)

    print("\n" + "=" * 70)
    print("BASELINE MLP")
    print("=" * 70)
    print_metrics("Validation", baseline_val)
    print_metrics("Test", baseline_test)

    print("\n" + "=" * 70)
    print("AUGMENTED MLP")
    print("=" * 70)
    print_metrics("Validation", augmented_val)
    print_metrics("Test", augmented_test)

    compare_test_metrics(baseline_test, augmented_test)

    results = {
        "augmentation_factor": args.augmentation_factor,
        "augmentation_mode": args.augmentation_mode,
        "noise_scale": args.noise_scale,
        "total_training_growth_factor": total_factor,
        "split_sizes": {
            "train_original": len(X_train),
            "train_augmented": len(X_train_aug),
            "validation": len(X_val),
            "test": len(X_test),
        },
        "baseline": {
            "validation": baseline_val,
            "test": baseline_test,
        },
        "augmented": {
            "validation": augmented_val,
            "test": augmented_test,
        },
    }
    args.results_path.parent.mkdir(parents=True, exist_ok=True)
    args.results_path.write_text(json.dumps(results, indent=2) + "\n")
    print(f"\nSaved results to {args.results_path}")


if __name__ == "__main__":
    main()
