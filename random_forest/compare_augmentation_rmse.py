#!/usr/bin/env python3
"""
Compare Random Forest RMSE across Mixup training-data augmentation multipliers.

The validation and test sets stay fixed and unaugmented. Only the training split
is expanded, so RMSE values are comparable across multipliers.
"""

import os
import sys
from pathlib import Path

os.environ.setdefault(
    "MPLCONFIGDIR",
    str(Path(__file__).parent.parent / "synthetic_data_gen" / "plots" / ".matplotlib_cache"),
)

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import root_mean_squared_error

sys.path.insert(0, str(Path(__file__).parent.parent))

from load_data import load_wine_data
from random_forest.utils import split_data
from synthetic_data_gen.mixup_data_augmentation import MixupAugmentation
from synthetic_data_gen.train_with_augmentation import preprocess_training_pipeline


MULTIPLIERS = [1, 2, 3, 4, 8]
DEFAULT_CCP_ALPHA = 0.0005
OUTPUT_DIR = Path(__file__).parent.parent / "synthetic_data_gen" / "plots" / "model_validation"


def augment_training_set(X_train: pd.DataFrame, y_train: pd.Series, multiplier: int):
    """Return the original training data plus Mixup rows up to total multiplier."""
    if multiplier == 1:
        return X_train.reset_index(drop=True), y_train.reset_index(drop=True)

    augmentor = MixupAugmentation(random_state=42 + multiplier)
    return augmentor.augment_dataset(
        X_train.reset_index(drop=True),
        y_train.reset_index(drop=True),
        augmentation_factor=multiplier - 1,
    )


def train_random_forest(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    alpha: float = DEFAULT_CCP_ALPHA,
) -> RandomForestRegressor:
    """Train the Random Forest configuration used for the augmentation comparison."""
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
        bootstrap=True,\
    )
    model.fit(X_train, y_train)
    return model


def rmse(model: RandomForestRegressor, X: pd.DataFrame, y: pd.Series) -> float:
    """Compute RMSE for a trained model."""
    predictions = model.predict(X)
    return root_mean_squared_error(y, predictions)


def save_rmse_plot(results: pd.DataFrame, output_dir: Path) -> Path:
    """Save augmentation multiplier vs RMSE plot."""
    plt.style.use("seaborn-v0_8-whitegrid")
    fig, ax = plt.subplots(figsize=(9, 5.8))

    if "validation_rmse" in results.columns:
        ax.plot(
            results["augmentation_multiplier"],
            results["validation_rmse"],
            marker="o",
            linewidth=2.2,
            label="Validation RMSE",
            color="#2563eb",
        )
    ax.plot(
        results["augmentation_multiplier"],
        results["test_rmse"],
        marker="s",
        linewidth=2.2,
        label="Test RMSE",
        color="#059669",
    )

    best_test_idx = results["test_rmse"].idxmin()
    best_test = results.loc[best_test_idx]
    ax.scatter(
        best_test["augmentation_multiplier"],
        best_test["test_rmse"],
        s=130,
        color="#ef4444",
        edgecolor="#111827",
        linewidth=0.7,
        zorder=3,
        label=f"Best test RMSE: {best_test['augmentation_multiplier']:.0f}x",
    )

    ax.set_title("Random Forest RMSE by Training Data Augmentation", fontsize=15, weight="bold", pad=12)
    ax.set_xlabel("Training Data Multiplier", fontsize=12)
    ax.set_ylabel("RMSE", fontsize=12)
    ax.set_xticks(results["augmentation_multiplier"])
    rmse_columns = [col for col in ["validation_rmse", "test_rmse"] if col in results.columns]
    min_rmse = results[rmse_columns].min().min()
    max_rmse = results[rmse_columns].max().max()
    rmse_range = max_rmse - min_rmse
    padding = max(rmse_range * 0.25, 0.01)
    ax.set_ylim(min_rmse - padding, max_rmse + padding)
    ax.tick_params(axis="both", labelsize=10)
    ax.legend(frameon=True)

    for _, row in results.iterrows():
        ax.annotate(
            f"{row['test_rmse']:.3f}",
            (row["augmentation_multiplier"], row["test_rmse"]),
            textcoords="offset points",
            xytext=(0, -16),
            ha="center",
            fontsize=9,
            color="#064e3b",
        )

    fig.tight_layout()
    plot_path = output_dir / "augmentation_multiplier_vs_rmse.png"
    fig.savefig(plot_path, dpi=180)
    plt.close(fig)
    return plot_path


def main() -> pd.DataFrame:
    print("Wine Quality Random Forest - Augmentation Multiplier RMSE Comparison")
    print("=" * 72)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("\nLoading data...")
    df = load_wine_data()

    print("\nSplitting data...")
    X_train, X_val, X_test, y_train, y_val, y_test = split_data(df)

    print("\nPreprocessing features...")
    X_train, X_val, X_test = preprocess_training_pipeline(X_train, X_val, X_test)

    rows = []
    for multiplier in MULTIPLIERS:
        print("\n" + "=" * 72)
        print(f"Training Random Forest with {multiplier}x total training data")
        print("=" * 72)

        X_train_aug, y_train_aug = augment_training_set(X_train, y_train, multiplier)
        model = train_random_forest(X_train_aug, y_train_aug)

        row = {
            "augmentation_multiplier": multiplier,
            "train_rows": len(X_train_aug),
            "validation_rmse": rmse(model, X_val, y_val),
            "test_rmse": rmse(model, X_test, y_test),
        }
        rows.append(row)

        print(f"Rows trained on:   {row['train_rows']}")
        print(f"Validation RMSE:   {row['validation_rmse']:.4f}")
        print(f"Test RMSE:         {row['test_rmse']:.4f}")

    results = pd.DataFrame(rows)
    csv_path = OUTPUT_DIR / "augmentation_multiplier_rmse.csv"
    results.to_csv(csv_path, index=False)
    plot_path = save_rmse_plot(results, OUTPUT_DIR)

    print("\n" + "=" * 72)
    print("RMSE Summary")
    print("=" * 72)
    print(results.to_string(index=False, formatters={
        "validation_rmse": "{:.4f}".format,
        "test_rmse": "{:.4f}".format,
    }))
    print(f"\nSaved CSV:   {csv_path}")
    print(f"Saved plot:  {plot_path}")

    return results


if __name__ == "__main__":
    main()
