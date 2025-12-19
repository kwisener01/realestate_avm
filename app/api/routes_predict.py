"""
Prediction API routes
"""

from fastapi import APIRouter, HTTPException, status
from typing import Optional
import pandas as pd
import numpy as np
from datetime import datetime

from app.models.property_models import (
    PredictionRequest,
    PredictionResponse,
    BatchPredictionRequest,
    BatchPredictionResponse,
    ModelBreakdown,
    ErrorResponse
)

router = APIRouter(prefix="/predict", tags=["predictions"])

# Global variable for loaded models (initialized in main.py)
stacker_model = None


def set_model(model):
    """Set the global model instance"""
    global stacker_model
    stacker_model = model


@router.post("/", response_model=PredictionResponse, status_code=status.HTTP_200_OK)
async def predict_property_value(request: PredictionRequest):
    """
    Predict property value based on provided features.

    Returns the predicted price along with model breakdown if using ensemble.
    """
    if stacker_model is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model not loaded. Please ensure models are trained and available."
        )

    try:
        # Prepare tabular data
        features_dict = request.features.dict()
        tabular_df = pd.DataFrame([features_dict])

        # Prepare text data
        texts = [request.description] if request.description else None

        # Prepare image data (in production, download from URL)
        image_paths = None  # Would implement image download in production

        # Check if meta-model is trained (required for ensemble)
        meta_model_trained = False
        try:
            # Check if meta_model has been fitted
            if hasattr(stacker_model.meta_model, 'coef_'):
                meta_model_trained = True
        except:
            meta_model_trained = False

        # Make prediction
        if request.use_ensemble and meta_model_trained:
            # Get detailed breakdown
            breakdown_dict = stacker_model.predict_with_breakdown(
                tabular_data=tabular_df,
                image_paths=image_paths,
                texts=texts
            )

            predicted_price = float(breakdown_dict['final_prediction'][0])

            # Create model breakdown
            base_preds = breakdown_dict.get('base_predictions', {})
            model_breakdown = ModelBreakdown(
                tabular=base_preds.get('tabular'),
                image=base_preds.get('image'),
                text=base_preds.get('text')
            )

            # Calculate confidence (simplified - based on model agreement)
            if len(base_preds) > 1:
                predictions_list = list(base_preds.values())
                std_dev = np.std(predictions_list)
                mean_pred = np.mean(predictions_list)
                # Confidence inversely related to coefficient of variation
                confidence = max(0.0, min(1.0, 1.0 - (std_dev / mean_pred)))
            else:
                confidence = 0.85  # Default confidence for single model

        else:
            # Use only tabular model (meta-model not trained or ensemble not requested)
            predicted_price = float(stacker_model.tabular_model.predict(tabular_df)[0])
            model_breakdown = ModelBreakdown(tabular=predicted_price)
            confidence = 0.80

        response = PredictionResponse(
            property_id=request.property_id,
            predicted_price=predicted_price,
            confidence_score=confidence,
            model_breakdown=model_breakdown,
            timestamp=datetime.now()
        )

        return response

    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Prediction failed: {str(e)}"
        )


@router.post("/batch", response_model=BatchPredictionResponse, status_code=status.HTTP_200_OK)
async def predict_batch(request: BatchPredictionRequest):
    """
    Predict values for multiple properties in a single request.

    Maximum batch size: 100 properties
    """
    if stacker_model is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model not loaded"
        )

    predictions = []
    successful = 0
    failed = 0

    for prop_request in request.predictions:
        try:
            prediction = await predict_property_value(prop_request)
            predictions.append(prediction)
            successful += 1
        except Exception as e:
            # Log error but continue processing
            failed += 1
            # Add placeholder response
            predictions.append(
                PredictionResponse(
                    property_id=prop_request.property_id,
                    predicted_price=0.0,
                    confidence_score=0.0,
                    timestamp=datetime.now()
                )
            )

    return BatchPredictionResponse(
        predictions=predictions,
        total_count=len(request.predictions),
        successful_count=successful,
        failed_count=failed
    )


@router.get("/health")
async def prediction_health():
    """Check if prediction service is healthy"""
    if stacker_model is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Models not loaded"
        )

    models_available = {
        "tabular": stacker_model.tabular_model is not None,
        "image": stacker_model.image_model is not None,
        "text": stacker_model.text_model is not None,
        "meta": stacker_model.meta_model is not None
    }

    return {
        "status": "healthy",
        "timestamp": datetime.now(),
        "models_loaded": models_available
    }
