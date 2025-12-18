"""
Tabular Model for Property Valuation
Handles numeric and categorical features like square footage, bedrooms, location, etc.
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
import joblib
import os
from typing import Dict, List, Optional


class TabularModel:
    """
    Gradient Boosting model for tabular property features.
    Features: bedrooms, bathrooms, sqft, lot_size, year_built, location, etc.
    """

    def __init__(self, model_path: Optional[str] = None):
        self.model = GradientBoostingRegressor(
            n_estimators=200,
            learning_rate=0.1,
            max_depth=5,
            min_samples_split=5,
            min_samples_leaf=2,
            subsample=0.8,
            random_state=42
        )
        self.scaler = StandardScaler()
        self.label_encoders = {}
        self.feature_names = None
        self.categorical_features = ['property_type', 'neighborhood', 'condition', 'view_quality']
        self.numeric_features = ['bedrooms', 'bathrooms', 'sqft_living', 'sqft_lot',
                                'floors', 'year_built', 'year_renovated', 'latitude', 'longitude']

        if model_path and os.path.exists(model_path):
            self.load(model_path)

    def _preprocess_features(self, df: pd.DataFrame, fit: bool = False) -> np.ndarray:
        """Preprocess features: encode categoricals and scale numerics"""
        df = df.copy()

        # Handle categorical features
        for col in self.categorical_features:
            if col in df.columns:
                if fit:
                    self.label_encoders[col] = LabelEncoder()
                    df[col] = self.label_encoders[col].fit_transform(df[col].astype(str))
                else:
                    if col in self.label_encoders:
                        # Handle unseen categories
                        df[col] = df[col].astype(str).apply(
                            lambda x: x if x in self.label_encoders[col].classes_
                            else self.label_encoders[col].classes_[0]
                        )
                        df[col] = self.label_encoders[col].transform(df[col])

        # Handle missing values
        df = df.fillna(df.median(numeric_only=True))

        # Select and order features
        all_features = self.numeric_features + self.categorical_features
        available_features = [f for f in all_features if f in df.columns]
        X = df[available_features].values

        # Scale features
        if fit:
            X = self.scaler.fit_transform(X)
            self.feature_names = available_features
        else:
            X = self.scaler.transform(X)

        return X

    def train(self, X: pd.DataFrame, y: np.ndarray, validation_split: float = 0.2) -> Dict:
        """
        Train the tabular model

        Args:
            X: DataFrame with property features
            y: Target values (property prices)
            validation_split: Fraction of data for validation

        Returns:
            Dictionary with training metrics
        """
        # Preprocess features
        X_processed = self._preprocess_features(X, fit=True)

        # Split data
        X_train, X_val, y_train, y_val = train_test_split(
            X_processed, y, test_size=validation_split, random_state=42
        )

        # Train model
        self.model.fit(X_train, y_train)

        # Evaluate
        train_score = self.model.score(X_train, y_train)
        val_score = self.model.score(X_val, y_val)

        train_predictions = self.model.predict(X_train)
        val_predictions = self.model.predict(X_val)

        train_mae = np.mean(np.abs(y_train - train_predictions))
        val_mae = np.mean(np.abs(y_val - val_predictions))

        train_mape = np.mean(np.abs((y_train - train_predictions) / y_train)) * 100
        val_mape = np.mean(np.abs((y_val - val_predictions) / y_val)) * 100

        metrics = {
            'train_r2': train_score,
            'val_r2': val_score,
            'train_mae': train_mae,
            'val_mae': val_mae,
            'train_mape': train_mape,
            'val_mape': val_mape
        }

        return metrics

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """
        Make predictions on new data

        Args:
            X: DataFrame with property features

        Returns:
            Array of predicted prices
        """
        X_processed = self._preprocess_features(X, fit=False)
        return self.model.predict(X_processed)

    def save(self, path: str):
        """Save model to disk"""
        os.makedirs(os.path.dirname(path) if os.path.dirname(path) else '.', exist_ok=True)
        joblib.dump({
            'model': self.model,
            'scaler': self.scaler,
            'label_encoders': self.label_encoders,
            'feature_names': self.feature_names,
            'categorical_features': self.categorical_features,
            'numeric_features': self.numeric_features
        }, path)

    def load(self, path: str):
        """Load model from disk"""
        data = joblib.load(path)
        self.model = data['model']
        self.scaler = data['scaler']
        self.label_encoders = data['label_encoders']
        self.feature_names = data['feature_names']
        self.categorical_features = data['categorical_features']
        self.numeric_features = data['numeric_features']
