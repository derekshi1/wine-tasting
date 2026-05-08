"""
Random Forest Regressor for wine quality prediction.
"""

import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import numpy as np


class WineQualityRF:
    """Random Forest model for wine quality prediction."""
    
    def __init__(self, n_estimators: int = 100, random_state: int = 42, **kwargs):
        """
        Initialize Random Forest Regressor.
        
        Args:
            n_estimators: Number of trees in forest
            random_state: Random seed for reproducibility
            **kwargs: Additional parameters for RandomForestRegressor
        """
        self.model = RandomForestRegressor(
            n_estimators=n_estimators,
            random_state=random_state,
            n_jobs=-1,  # Use all cores
            **kwargs
        )
        self.is_trained = False
    
    def train(self, X_train: pd.DataFrame, y_train: pd.Series):
        """
        Train the model on training data.
        
        Args:
            X_train: Training features
            y_train: Training targets
        """
        print("🚂 Training Random Forest model...")
        self.model.fit(X_train, y_train)
        self.is_trained = True
        print("✅ Training complete!")
    
    def predict(self, X):
        """
        Make predictions on new data.
        
        Args:
            X: Features to predict on
        
        Returns:
            Predictions
        """
        if not self.is_trained:
            raise ValueError("Model not trained yet. Call train() first.")
        return self.model.predict(X)
    
    def evaluate(self, X, y, set_name: str = ""):
        """
        Evaluate model on given data.
        
        Args:
            X: Features
            y: True targets
            set_name: Name of set (e.g., 'Training', 'Validation', 'Test')
        
        Returns:
            Dictionary of metrics
        """
        predictions = self.predict(X)
        
        mse = mean_squared_error(y, predictions)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(y, predictions)
        r2 = r2_score(y, predictions)
        
        metrics = {
            "MSE": mse,
            "RMSE": rmse,
            "MAE": mae,
            "R2": r2
        }
        
        if set_name:
            print(f"\n📊 {set_name} Set Performance:")
        else:
            print(f"\n📊 Performance Metrics:")
        
        print(f"   MSE:  {mse:.4f}")
        print(f"   RMSE: {rmse:.4f}")
        print(f"   MAE:  {mae:.4f}")
        print(f"   R²:   {r2:.4f}")
        
        return metrics
    
    def get_feature_importance(self, feature_names, top_n: int = 10):
        """
        Get feature importance from trained model.
        
        Args:
            feature_names: Names of features
            top_n: Number of top features to show
        
        Returns:
            DataFrame with feature importance
        """
        if not self.is_trained:
            raise ValueError("Model not trained yet.")
        
        importances = self.model.feature_importances_
        feature_importance_df = pd.DataFrame({
            "feature": feature_names,
            "importance": importances
        }).sort_values("importance", ascending=False)
        
        print(f"\n🎯 Top {top_n} Feature Importance:")
        for idx, row in feature_importance_df.head(top_n).iterrows():
            print(f"   {row['feature']:20} {row['importance']:.4f}")
        
        return feature_importance_df
