"""
ML-based ARV (After Repair Value) prediction service using ensemble models.
Provides confidence intervals and model agreement scoring.
"""
import pandas as pd
import numpy as np
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor, ExtraTreesRegressor
from sklearn.linear_model import Ridge
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import mean_absolute_error, r2_score, mean_absolute_percentage_error
from sklearn.preprocessing import LabelEncoder, StandardScaler
import pickle
import os
from pathlib import Path
from typing import Dict, Tuple, Optional
import re


class MLARVPredictor:
    """
    Enhanced ARV predictor using ensemble of ML models.
    Provides primary ARV estimate with confidence intervals.
    """

    def __init__(self):
        self.models = {}
        self.scalers = {}
        self.encoders = {}
        self.feature_columns = []
        self.is_trained = False

    def _clean_price(self, price_str) -> Optional[float]:
        """Convert price string to numeric"""
        if pd.isna(price_str):
            return None
        cleaned = re.sub(r'[,$]', '', str(price_str))
        try:
            return float(cleaned)
        except:
            return None

    def _engineer_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create enhanced features for better predictions"""
        df = df.copy()

        # Basic features
        df['sqft'] = df.get('Building Sqft', 0)
        df['beds'] = df.get('Bedrooms', 0)
        df['baths'] = df.get('Total Bathrooms', 2.0)
        df['lot_size'] = df.get('Lot Size Sqft', 10000)
        df['year_built'] = df.get('Effective Year Built', 1990)
        df['assessed_value'] = df.get('Total Assessed Value', 0)

        # Derived features
        df['age'] = 2025 - df['year_built']
        df['age_squared'] = df['age'] ** 2  # Non-linear age effect
        df['sqft_per_bed'] = df['sqft'] / df['beds'].replace(0, 1)
        df['sqft_per_bath'] = df['sqft'] / df['baths'].replace(0, 1)
        df['lot_to_building_ratio'] = df['lot_size'] / df['sqft'].replace(0, 1)
        df['assessed_per_sqft'] = df['assessed_value'] / df['sqft'].replace(0, 1)

        # Property quality indicators
        df['is_large'] = (df['sqft'] > 2500).astype(int)
        df['is_new'] = (df['age'] < 10).astype(int)
        df['is_renovated'] = df.get('Year Remodeled', 0) > 0
        df['has_large_lot'] = (df['lot_size'] > 15000).astype(int)

        # Location encoding (city)
        if 'City' in df.columns:
            if 'city' not in self.encoders:
                self.encoders['city'] = LabelEncoder()
                self.encoders['city'].fit(df['City'].fillna('Unknown'))

            df['city_encoded'] = df['City'].fillna('Unknown').apply(
                lambda x: self.encoders['city'].transform([x])[0]
                if x in self.encoders['city'].classes_ else -1
            )
        else:
            df['city_encoded'] = 0

        return df

    def prepare_training_data(self, csv_paths: list) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Load and prepare training data from multiple county CSV files.

        Args:
            csv_paths: List of paths to county CSV files

        Returns:
            X: Feature matrix
            y: Target values (ARV)
        """
        print("\n" + "="*80)
        print("PREPARING TRAINING DATA")
        print("="*80)

        dfs = []
        for path in csv_paths:
            if os.path.exists(path):
                df = pd.read_csv(path)
                dfs.append(df)
                print(f"  [OK] Loaded {len(df)} properties from {os.path.basename(path)}")
            else:
                print(f"  [SKIP] File not found: {path}")

        if not dfs:
            raise ValueError("No valid CSV files found for training")

        # Combine all data
        df_all = pd.concat(dfs, ignore_index=True)
        print(f"\n  Total properties: {len(df_all)}")

        # Filter for valid training samples
        df_all = df_all[df_all['Last Sale Amount'].notna()].copy()
        df_all = df_all[df_all['Building Sqft'] > 0].copy()
        df_all = df_all[df_all['Bedrooms'] > 0].copy()
        df_all = df_all[df_all['Total Assessed Value'] > 0].copy()

        print(f"  Properties with complete data: {len(df_all)}")

        # Engineer features
        df_all = self._engineer_features(df_all)

        # Define feature columns
        self.feature_columns = [
            'sqft', 'beds', 'baths', 'lot_size', 'age', 'age_squared',
            'assessed_value', 'assessed_per_sqft', 'sqft_per_bed', 'sqft_per_bath',
            'lot_to_building_ratio', 'is_large', 'is_new', 'is_renovated',
            'has_large_lot', 'city_encoded'
        ]

        X = df_all[self.feature_columns].fillna(0)
        y = df_all['Last Sale Amount']

        print(f"\n  Features: {len(self.feature_columns)}")
        print(f"  Target: Last Sale Amount (actual ARV)")
        print(f"\n  ARV Statistics:")
        print(f"    Mean: ${y.mean():,.0f}")
        print(f"    Median: ${y.median():,.0f}")
        print(f"    Min: ${y.min():,.0f}")
        print(f"    Max: ${y.max():,.0f}")

        return X, y

    def train_ensemble(self, X: pd.DataFrame, y: pd.Series):
        """
        Train ensemble of models for robust predictions.
        """
        print("\n" + "="*80)
        print("TRAINING ENSEMBLE MODELS")
        print("="*80)

        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

        # Scale features
        self.scalers['standard'] = StandardScaler()
        X_train_scaled = self.scalers['standard'].fit_transform(X_train)
        X_test_scaled = self.scalers['standard'].transform(X_test)

        # Define ensemble models
        model_configs = {
            'gradient_boosting': GradientBoostingRegressor(
                n_estimators=300,
                learning_rate=0.05,
                max_depth=6,
                min_samples_split=10,
                min_samples_leaf=5,
                subsample=0.8,
                random_state=42
            ),
            'random_forest': RandomForestRegressor(
                n_estimators=200,
                max_depth=12,
                min_samples_split=10,
                min_samples_leaf=5,
                random_state=42,
                n_jobs=-1
            ),
            'extra_trees': ExtraTreesRegressor(
                n_estimators=200,
                max_depth=12,
                min_samples_split=10,
                min_samples_leaf=5,
                random_state=42,
                n_jobs=-1
            ),
            'ridge': Ridge(alpha=10.0, random_state=42)
        }

        # Train each model
        performances = {}
        for name, model in model_configs.items():
            print(f"\n  Training {name}...")

            if name == 'ridge':
                # Ridge needs scaled features
                model.fit(X_train_scaled, y_train)
                y_pred_train = model.predict(X_train_scaled)
                y_pred_test = model.predict(X_test_scaled)
            else:
                # Tree-based models use raw features
                model.fit(X_train, y_train)
                y_pred_train = model.predict(X_train)
                y_pred_test = model.predict(X_test)

            # Evaluate
            train_mae = mean_absolute_error(y_train, y_pred_train)
            test_mae = mean_absolute_error(y_test, y_pred_test)
            train_r2 = r2_score(y_train, y_pred_train)
            test_r2 = r2_score(y_test, y_pred_test)

            performances[name] = {
                'train_mae': train_mae,
                'test_mae': test_mae,
                'train_r2': train_r2,
                'test_r2': test_r2
            }

            print(f"    Train MAE: ${train_mae:,.0f} | R²: {train_r2:.3f}")
            print(f"    Test MAE:  ${test_mae:,.0f} | R²: {test_r2:.3f}")

            self.models[name] = model

        # Show best model
        best_model = min(performances.items(), key=lambda x: x[1]['test_mae'])
        print(f"\n  [BEST] Model: {best_model[0]} (Test MAE: ${best_model[1]['test_mae']:,.0f})")

        self.is_trained = True
        return performances

    def predict_with_confidence(self, property_data: Dict) -> Dict:
        """
        Predict ARV with confidence intervals using ensemble.

        Args:
            property_data: Dictionary with property features

        Returns:
            Dictionary with ARV prediction, confidence interval, and metadata
        """
        if not self.is_trained:
            raise ValueError("Model not trained. Call train_ensemble first.")

        # Convert to DataFrame
        df = pd.DataFrame([property_data])

        # Engineer features
        df = self._engineer_features(df)

        # Extract features
        X = df[self.feature_columns].fillna(0)
        X_scaled = self.scalers['standard'].transform(X)

        # Get predictions from all models
        predictions = []
        for name, model in self.models.items():
            if name == 'ridge':
                pred = model.predict(X_scaled)[0]
            else:
                pred = model.predict(X)[0]
            predictions.append(pred)

        # Calculate ensemble statistics
        arv_median = np.median(predictions)
        arv_mean = np.mean(predictions)
        arv_std = np.std(predictions)
        arv_min = np.min(predictions)
        arv_max = np.max(predictions)

        # Confidence based on model agreement
        cv = arv_std / arv_mean if arv_mean > 0 else 1.0  # Coefficient of variation

        if cv < 0.05:  # Models agree within 5%
            confidence = "HIGH"
        elif cv < 0.10:  # Models agree within 10%
            confidence = "MEDIUM"
        else:
            confidence = "LOW"

        # Calculate confidence interval (use ±1 std dev, ~68% confidence)
        arv_lower = max(0, arv_mean - arv_std)
        arv_upper = arv_mean + arv_std

        return {
            'arv_prediction': int(arv_mean),
            'arv_median': int(arv_median),
            'arv_lower': int(arv_lower),
            'arv_upper': int(arv_upper),
            'confidence': confidence,
            'cv': cv,
            'model_predictions': {
                name: int(pred) for name, pred in zip(self.models.keys(), predictions)
            }
        }

    def save_model(self, path: str):
        """Save trained model to disk"""
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, 'wb') as f:
            pickle.dump({
                'models': self.models,
                'scalers': self.scalers,
                'encoders': self.encoders,
                'feature_columns': self.feature_columns,
                'is_trained': self.is_trained
            }, f)
        print(f"\n  [OK] Model saved to: {path}")

    def load_model(self, path: str):
        """Load trained model from disk"""
        with open(path, 'rb') as f:
            data = pickle.load(f)
            self.models = data['models']
            self.scalers = data['scalers']
            self.encoders = data['encoders']
            self.feature_columns = data['feature_columns']
            self.is_trained = data['is_trained']
        print(f"  [OK] Model loaded from: {path}")


def compare_with_zillow(ml_arv: int, zillow_zestimate: Optional[float]) -> Dict:
    """
    Compare ML ARV with Zillow Zestimate and determine agreement.

    Args:
        ml_arv: ML-predicted ARV
        zillow_zestimate: Zillow Zestimate value (or None)

    Returns:
        Dictionary with comparison results and confidence
    """
    if zillow_zestimate is None or zillow_zestimate == 0:
        return {
            'agreement': 'NO_ZILLOW',
            'primary_arv': ml_arv,
            'confidence': 'ML_ONLY',
            'difference_pct': None,
            'flag_review': False
        }

    # Calculate difference
    diff = abs(ml_arv - zillow_zestimate)
    diff_pct = (diff / zillow_zestimate) * 100

    # Determine agreement
    if diff_pct < 5:
        agreement = 'STRONG_AGREE'
        confidence = 'VERY_HIGH'
        flag_review = False
    elif diff_pct < 10:
        agreement = 'AGREE'
        confidence = 'HIGH'
        flag_review = False
    elif diff_pct < 20:
        agreement = 'MODERATE'
        confidence = 'MEDIUM'
        flag_review = True
    else:
        agreement = 'DISAGREE'
        confidence = 'LOW'
        flag_review = True

    # Use average when they agree, ML when they disagree
    if agreement in ['STRONG_AGREE', 'AGREE']:
        primary_arv = int((ml_arv + zillow_zestimate) / 2)
    else:
        primary_arv = ml_arv  # Trust ML over Zillow when they disagree

    return {
        'agreement': agreement,
        'primary_arv': primary_arv,
        'confidence': confidence,
        'difference_pct': diff_pct,
        'flag_review': flag_review,
        'zillow_value': int(zillow_zestimate),
        'ml_value': ml_arv
    }
