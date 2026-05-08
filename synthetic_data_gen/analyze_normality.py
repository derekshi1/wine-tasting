#!/usr/bin/env python3
"""
Analyze feature distributions and check for normality.
Generates plots to visualize if Gaussian sampling is appropriate.
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))



def check_normality(df: pd.DataFrame) -> pd.DataFrame:
    """
    Test each numeric feature for normality using Shapiro-Wilk test.
    
    Args:
        df: Dataset
    
    Returns:
        DataFrame with normality test results
    """
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    results = []
    
    print("Normality Test Results (Shapiro-Wilk)")
    print("=" * 70)
    print(f"{'Feature':<20} {'Statistic':<12} {'P-Value':<12} {'Normal?':<10}")
    print("=" * 70)
    
    for col in numeric_cols:
        # Skip if all values are constant
        if df[col].std() == 0:
            continue
        
        stat, p_value = stats.shapiro(df[col])
        is_normal = p_value > 0.05  # 5% significance level
        
        results.append({
            "feature": col,
            "statistic": stat,
            "p_value": p_value,
            "is_normal": is_normal
        })
        
        status = "✅ YES" if is_normal else "❌ NO"
        print(f"{col:<20} {stat:<12.4f} {p_value:<12.6f} {status:<10}")
    
    print("=" * 70)
    
    results_df = pd.DataFrame(results)
    normal_count = results_df["is_normal"].sum()
    total_count = len(results_df)
    
    print(f"\n📈 Summary: {normal_count}/{total_count} features are normally distributed")
    
    return results_df


def plot_distributions(df: pd.DataFrame, output_dir: Path):
    """
    Create distribution plots for all numeric features.
    
    Args:
        df: Dataset
        output_dir: Directory to save plots
    """
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    
    # Create output directory
    output_dir.mkdir(exist_ok=True)
    
    print(f"\n📈 Generating distribution plots...")
    
    for col in numeric_cols:
        fig, axes = plt.subplots(1, 2, figsize=(12, 4))
        
        # Histogram with KDE
        axes[0].hist(df[col], bins=30, density=True, alpha=0.7, color="skyblue", edgecolor="black")
        df[col].plot.kde(ax=axes[0], color="red", linewidth=2, label="KDE")
        axes[0].set_title(f"{col} - Distribution", fontsize=12, fontweight="bold")
        axes[0].set_xlabel(col)
        axes[0].set_ylabel("Density")
        axes[0].legend()
        axes[0].grid(alpha=0.3)
        
        # Q-Q plot (to check normality visually)
        stats.probplot(df[col], dist="norm", plot=axes[1])
        axes[1].set_title(f"{col} - Q-Q Plot", fontsize=12, fontweight="bold")
        axes[1].grid(alpha=0.3)
        
        plt.tight_layout()
        
        # Save figure
        filename = output_dir / f"{col.replace(' ', '_')}_distribution.png"
        plt.savefig(filename, dpi=100, bbox_inches="tight")
        plt.close()
        
        print(f"   ✅ Saved: {col}")
    
    print(f"\n📁 All plots saved to: {output_dir}")


def plot_summary_heatmap(df: pd.DataFrame, output_dir: Path):
    """
    Create a heatmap showing skewness and kurtosis for all features.
    
    Args:
        df: Dataset
        output_dir: Directory to save plots
    """
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    
    skewness = []
    kurtosis = []
    features = []
    
    for col in numeric_cols:
        features.append(col)
        skewness.append(stats.skew(df[col]))
        kurtosis.append(stats.kurtosis(df[col]))
    
    summary_df = pd.DataFrame({
        "Skewness": skewness,
        "Kurtosis": kurtosis
    }, index=features)
    
    # For normal distribution:
    # - Skewness should be close to 0
    # - Kurtosis should be close to 0
    
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(summary_df, annot=True, fmt=".2f", cmap="RdYlGn_r", center=0,
                cbar_kws={"label": "Value"}, ax=ax, vmin=-2, vmax=2)
    ax.set_title("Feature Normality Indicators\n(Skewness & Kurtosis near 0 = more normal)", 
                 fontsize=12, fontweight="bold")
    plt.tight_layout()
    
    filename = output_dir / "normality_heatmap.png"
    plt.savefig(filename, dpi=100, bbox_inches="tight")
    plt.close()
    
    print(f"   ✅ Saved: normality_heatmap.png")


def main():
    """Main analysis pipeline."""
    print("🍷 Feature Normality Analysis")
    print("=" * 70)
    
    # Load data
    df = pd.read_csv("/Users/derek/Documents/wine-tasting/WineQT.csv")
    numeric_df = df.select_dtypes(include=[np.number])
    
    # Check normality
    print("\n🔍 Running normality tests...")
    normality_results = check_normality(numeric_df)
    
    # Create output directory
    output_dir = Path(__file__).parent / "plots"
    
    # Generate plots
    plot_distributions(numeric_df, output_dir)
    plot_summary_heatmap(numeric_df, output_dir)
    
    # Recommendations
    print("\n" + "=" * 70)
    print("💡 RECOMMENDATIONS FOR SYNTHETIC DATA GENERATION")
    print("=" * 70)
    
    print("\n" + "=" * 70)
    
    return normality_results


if __name__ == "__main__":
    main()
