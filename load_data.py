#!/usr/bin/env python3
"""
Data loading and exploration script for Wine Quality dataset.
"""

import pandas as pd
from pathlib import Path


def load_wine_data(filepath: str = "WineQT.csv") -> pd.DataFrame:
    """
    Load the wine quality dataset from CSV file.
    
    Args:
        filepath: Path to the CSV file (default: WineQT.csv in current directory)
    
    Returns:
        DataFrame with wine data
    """
    if not Path(filepath).exists():
        raise FileNotFoundError(f"Data file not found: {filepath}")
    
    df = pd.read_csv(filepath)
    return df


def explore_data(df: pd.DataFrame) -> None:
    """
    Print basic information about the dataset.
    
    Args:
        df: DataFrame to explore
    """
    print("🍷 Wine Quality Dataset Overview")
    print("=" * 50)
    
    # Basic info
    print(f"\n📊 Dataset Shape: {df.shape[0]} rows × {df.shape[1]} columns")
    
    # Columns and types
    print("\n📋 Columns and Data Types:")
    print(df.dtypes)
    
    # Missing values
    print("\n🔍 Missing Values:")
    missing = df.isnull().sum()
    if missing.sum() == 0:
        print("   ✅ No missing values")
    else:
        print(missing[missing > 0])
    
    # Basic statistics
    print("\n📈 Statistical Summary:")
    print(df.describe())
    
    # First few rows
    print("\n👁️ First 5 Rows:")
    print(df.head())
    
    # Data info
    print("\n💾 Memory Usage:")
    print(f"   {df.memory_usage(deep=True).sum() / 1024:.2f} KB")


def get_quality_distribution(df: pd.DataFrame) -> None:
    """Print distribution of wine quality ratings."""
    if "quality" in df.columns:
        print("\n⭐ Quality Distribution:")
        print(df["quality"].value_counts().sort_index())
    else:
        print("\n⚠️ 'quality' column not found in dataset")


def get_feature_correlations(df: pd.DataFrame, top_n: int = 10) -> None:
    """
    Print features most correlated with quality.
    
    Args:
        df: DataFrame to analyze
        top_n: Number of top correlations to show
    """
    if "quality" not in df.columns:
        print("⚠️ 'quality' column not found")
        return
    
    numeric_df = df.select_dtypes(include=["number"])
    correlations = numeric_df.corr()["quality"].sort_values(ascending=False)
    
    print(f"\n🔗 Top {top_n} Features Correlated with Quality:")
    for feature, corr in correlations.head(top_n + 1).items():
        if feature != "quality":
            print(f"   {feature:20} {corr:+.4f}")


def main():
    """Main entry point."""
    try:
        # Load data
        df = load_wine_data()
        
        # Explore
        explore_data(df)
        get_quality_distribution(df)
        get_feature_correlations(df)
        
        print("\n✅ Data loading complete!")
        
    except FileNotFoundError as e:
        print(f"❌ Error: {e}")
    except Exception as e:
        print(f"❌ Unexpected error: {e}")


if __name__ == "__main__":
    main()
