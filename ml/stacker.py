"""
Stacker Ensemble Model
Combines predictions from tabular, image, and text models using a meta-learner
"""

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import cross_val_score
import joblib
import os
from typing import Dict, List, Optional, Tuple

try:
    # Try relative imports first (when running from ml/ directory)
    from tabular_model import TabularModel
except ImportError:
    # Fall back to absolute imports (when running from project root)
    from ml.tabular_model import TabularModel

# Optional imports for image and text models (require PyTorch)
try:
    from image_model import ImageModelWrapper
except ImportError:
    try:
        from ml.image_model import ImageModelWrapper
    except ImportError:
        ImageModelWrapper = None  # PyTorch not available

try:
    from text_model import TextModelWrapper
except ImportError:
    try:
        from ml.text_model import TextModelWrapper
    except ImportError:
        TextModelWrapper = None  # PyTorch not available


class StackerModel:
    """
    Stacking ensemble that combines predictions from multiple models.
    Uses a meta-learner (Ridge regression) to weight and combine base model predictions.
    """

    def __init__(self,
                 tabular_model_path: Optional[str] = None,
                 image_model_path: Optional[str] = None,
                 text_model_path: Optional[str] = None,
                 meta_model_path: Optional[str] = None):
        """
        Initialize stacker with optional pre-trained models

        Args:
            tabular_model_path: Path to trained tabular model
            image_model_path: Path to trained image model
            text_model_path: Path to trained text model
            meta_model_path: Path to trained meta-learner
        """
        # Initialize base models
        self.tabular_model = TabularModel(tabular_model_path) if tabular_model_path else None
        self.image_model = ImageModelWrapper(image_model_path) if image_model_path else None
        self.text_model = TextModelWrapper(text_model_path) if text_model_path else None

        # Initialize meta-learner
        self.meta_model = Ridge(alpha=1.0)

        if meta_model_path and os.path.exists(meta_model_path):
            self.load_meta_model(meta_model_path)

    def _get_base_predictions(self,
                             tabular_data: Optional[pd.DataFrame] = None,
                             image_paths: Optional[List[str]] = None,
                             texts: Optional[List[str]] = None) -> np.ndarray:
        """
        Get predictions from all available base models

        Returns:
            Array of shape (n_samples, n_models) with base model predictions
        """
        predictions = []

        if tabular_data is not None and self.tabular_model is not None:
            tabular_preds = self.tabular_model.predict(tabular_data)
            predictions.append(tabular_preds)

        if image_paths is not None and self.image_model is not None:
            image_preds = self.image_model.predict(image_paths)
            predictions.append(image_preds)

        if texts is not None and self.text_model is not None:
            text_preds = self.text_model.predict(texts)
            predictions.append(text_preds)

        if not predictions:
            raise ValueError("At least one model must be available for predictions")

        # Stack predictions
        return np.column_stack(predictions)

    def train_meta_model(self,
                        tabular_data: pd.DataFrame,
                        image_paths: List[str],
                        texts: List[str],
                        labels: np.ndarray,
                        cv_folds: int = 5) -> Dict:
        """
        Train the meta-learner on base model predictions

        Args:
            tabular_data: DataFrame with tabular features
            image_paths: List of image paths
            texts: List of property descriptions
            labels: True property values
            cv_folds: Number of cross-validation folds

        Returns:
            Dictionary with training metrics
        """
        # Get base model predictions
        print("Generating base model predictions...")
        base_predictions = self._get_base_predictions(tabular_data, image_paths, texts)

        # Train meta-model
        print("Training meta-learner...")
        self.meta_model.fit(base_predictions, labels)

        # Evaluate with cross-validation
        cv_scores = cross_val_score(
            self.meta_model, base_predictions, labels,
            cv=cv_folds, scoring='r2'
        )

        # Calculate metrics
        final_predictions = self.meta_model.predict(base_predictions)
        mae = np.mean(np.abs(labels - final_predictions))
        mape = np.mean(np.abs((labels - final_predictions) / labels)) * 100
        r2 = self.meta_model.score(base_predictions, labels)

        # Get model weights (coefficients)
        model_names = []
        if self.tabular_model is not None:
            model_names.append('tabular')
        if self.image_model is not None:
            model_names.append('image')
        if self.text_model is not None:
            model_names.append('text')

        weights = dict(zip(model_names, self.meta_model.coef_))

        metrics = {
            'r2': r2,
            'mae': mae,
            'mape': mape,
            'cv_scores': cv_scores,
            'cv_mean': cv_scores.mean(),
            'cv_std': cv_scores.std(),
            'model_weights': weights,
            'intercept': self.meta_model.intercept_
        }

        return metrics

    def predict(self,
               tabular_data: Optional[pd.DataFrame] = None,
               image_paths: Optional[List[str]] = None,
               texts: Optional[List[str]] = None) -> np.ndarray:
        """
        Make predictions using the stacked ensemble

        Args:
            tabular_data: DataFrame with tabular features
            image_paths: List of image paths
            texts: List of property descriptions

        Returns:
            Array of predicted property values
        """
        # Get base model predictions
        base_predictions = self._get_base_predictions(tabular_data, image_paths, texts)

        # Meta-model prediction
        return self.meta_model.predict(base_predictions)

    def predict_with_breakdown(self,
                              tabular_data: Optional[pd.DataFrame] = None,
                              image_paths: Optional[List[str]] = None,
                              texts: Optional[List[str]] = None) -> Dict:
        """
        Make predictions and return breakdown by model

        Returns:
            Dictionary with overall prediction and individual model contributions
        """
        base_predictions = self._get_base_predictions(tabular_data, image_paths, texts)
        final_prediction = self.meta_model.predict(base_predictions)

        # Calculate contributions
        model_names = []
        if self.tabular_model is not None:
            model_names.append('tabular')
        if self.image_model is not None:
            model_names.append('image')
        if self.text_model is not None:
            model_names.append('text')

        breakdown = {
            'final_prediction': final_prediction,
            'base_predictions': dict(zip(model_names, base_predictions[0])),
            'model_weights': dict(zip(model_names, self.meta_model.coef_)),
            'intercept': self.meta_model.intercept_
        }

        return breakdown

    def save_meta_model(self, path: str):
        """Save meta-learner to disk"""
        os.makedirs(os.path.dirname(path) if os.path.dirname(path) else '.', exist_ok=True)
        joblib.dump(self.meta_model, path)

    def load_meta_model(self, path: str):
        """Load meta-learner from disk"""
        self.meta_model = joblib.load(path)

    def save_all_models(self, output_dir: str):
        """
        Save all models to directory

        Args:
            output_dir: Directory to save models
        """
        os.makedirs(output_dir, exist_ok=True)

        if self.tabular_model is not None:
            self.tabular_model.save(os.path.join(output_dir, 'tabular_model.joblib'))

        if self.image_model is not None:
            self.image_model.save(os.path.join(output_dir, 'image_model.pth'))

        if self.text_model is not None:
            self.text_model.save(os.path.join(output_dir, 'text_model.pth'))

        self.save_meta_model(os.path.join(output_dir, 'meta_model.joblib'))

        print(f"All models saved to {output_dir}")


class SimpleAverageEnsemble:
    """
    Simple averaging ensemble as baseline
    Averages predictions from all available models
    """

    def __init__(self,
                 tabular_model: Optional[TabularModel] = None,
                 image_model: Optional[ImageModelWrapper] = None,
                 text_model: Optional[TextModelWrapper] = None):
        self.tabular_model = tabular_model
        self.image_model = image_model
        self.text_model = text_model

    def predict(self,
               tabular_data: Optional[pd.DataFrame] = None,
               image_paths: Optional[List[str]] = None,
               texts: Optional[List[str]] = None) -> np.ndarray:
        """Average predictions from available models"""
        predictions = []

        if tabular_data is not None and self.tabular_model is not None:
            predictions.append(self.tabular_model.predict(tabular_data))

        if image_paths is not None and self.image_model is not None:
            predictions.append(self.image_model.predict(image_paths))

        if texts is not None and self.text_model is not None:
            predictions.append(self.text_model.predict(texts))

        if not predictions:
            raise ValueError("At least one model must be available")

        return np.mean(predictions, axis=0)
