"""
Mixup augmentation for synthetic data generation.
Creates synthetic samples by interpolating between pairs of existing samples.
"""

import pandas as pd
import numpy as np
from typing import Tuple


class MixupAugmentation:
    """Generate synthetic data using mixup interpolation."""
    
    def __init__(self, random_state: int = 42):
        """
        Initialize MixupAugmentation.
        
        Args:
            random_state: Random seed for reproducibility
        """
        self.random_state = random_state
        np.random.seed(random_state)
    
    def generate_synthetic_samples(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        n_synthetic: int = 1000,
        alpha_distribution: str = "uniform"
    ) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Generate synthetic samples using mixup.
        
        For each synthetic sample:
        1. Randomly select two samples from the data
        2. Draw alpha from distribution (0, 1)
        3. Create synthetic: X_synthetic = alpha * X1 + (1 - alpha) * X2
        4. Label: y_synthetic = alpha * y1 + (1 - alpha) * y2
        
        Args:
            X: Feature matrix (n_samples, n_features)
            y: Target values (n_samples,)
            n_synthetic: Number of synthetic samples to generate
            alpha_distribution: How to sample alpha ("uniform" or "beta")
        
        Returns:
            Tuple of (X_synthetic, y_synthetic)
        """
        n_samples = len(X)
        X_synthetic_list = []
        y_synthetic_list = []
        
        print(f" Generating {n_synthetic} synthetic samples via Mixup...")
        
        for i in range(n_synthetic):
            # Randomly select two samples
            idx1, idx2 = np.random.choice(n_samples, size=2, replace=True)
            
            # Sample alpha
            if alpha_distribution == "uniform":
                alpha = np.random.uniform(0, 1)
            elif alpha_distribution == "beta":
                alpha = np.random.beta(1.0, 1.0)  # Beta(1,1) is uniform
            else:
                raise ValueError(f"Unknown distribution: {alpha_distribution}")
            
            # Mixup
            x_synthetic = alpha * X.iloc[idx1].values + (1 - alpha) * X.iloc[idx2].values
            y_synthetic = alpha * y.iloc[idx1] + (1 - alpha) * y.iloc[idx2]
            
            X_synthetic_list.append(x_synthetic)
            y_synthetic_list.append(y_synthetic)
            
            # Progress indicator
            if (i + 1) % max(1, n_synthetic // 10) == 0:
                print(f"   {i + 1}/{n_synthetic} ({(i + 1) / n_synthetic * 100:.0f}%)")
        
        # Convert to DataFrames
        X_synthetic = pd.DataFrame(
            np.array(X_synthetic_list),
            columns=X.columns
        )
        y_synthetic = pd.Series(y_synthetic_list, name=y.name)
        
        print(f"✅ Generated {n_synthetic} synthetic samples")
        
        return X_synthetic, y_synthetic
    
    def augment_dataset(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        augmentation_factor: float = 1.0
    ) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Augment dataset by combining original with synthetic data.
        
        Args:
            X: Feature matrix
            y: Target values
            augmentation_factor: Multiplier for synthetic data size
                                (1.0 = add as many synthetic as original)
        
        Returns:
            Tuple of (X_augmented, y_augmented)
        """
        n_synthetic = int(len(X) * augmentation_factor)
        
        X_synthetic, y_synthetic = self.generate_synthetic_samples(X, y, n_synthetic)
        
        # Combine original and synthetic
        X_augmented = pd.concat([X, X_synthetic], ignore_index=True)
        y_augmented = pd.concat([y, y_synthetic], ignore_index=True)
        
        print(f"\n📊 Dataset Augmentation Summary:")
        print(f"   Original size:  {len(X)}")
        print(f"   Synthetic size: {len(X_synthetic)}")
        print(f"   Augmented size: {len(X_augmented)}")
        print(f"   Growth factor:  {len(X_augmented) / len(X):.2f}x")
        
        return X_augmented, y_augmented
    
    @staticmethod
    def validate_synthetic_data(
        X_original: pd.DataFrame,
        X_synthetic: pd.DataFrame
    ) -> None:
        """
        Validate that synthetic data is reasonable.
        Check if synthetic features fall within original ranges.
        
        Args:
            X_original: Original feature matrix
            X_synthetic: Synthetic feature matrix
        """
        print(f"\n✓ Synthetic Data Validation:")
        print(f"{'Feature':<25} {'Original Range':<20} {'Synthetic Range':<20} {'Status'}")
        print("=" * 75)
        
        for col in X_original.columns:
            orig_min, orig_max = X_original[col].min(), X_original[col].max()
            syn_min, syn_max = X_synthetic[col].min(), X_synthetic[col].max()
            
            # Check if synthetic is within reasonable bounds
            # Allow small margin (5%) beyond original range
            margin = (orig_max - orig_min) * 0.05
            in_range = (syn_min >= orig_min - margin) and (syn_max <= orig_max + margin)
            
            status = "✅" if in_range else "⚠️"
            
            orig_range = f"[{orig_min:.2f}, {orig_max:.2f}]"
            syn_range = f"[{syn_min:.2f}, {syn_max:.2f}]"
            
            print(f"{col:<25} {orig_range:<20} {syn_range:<20} {status}")
