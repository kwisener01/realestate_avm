"""
Pydantic models for API requests and responses
"""

from pydantic import BaseModel, Field, validator
from typing import Optional, List
from datetime import datetime


class PropertyFeatures(BaseModel):
    """Features for property valuation"""
    bedrooms: int = Field(..., ge=1, description="Number of bedrooms")
    bathrooms: float = Field(..., ge=1.0, description="Number of bathrooms")
    sqft_living: int = Field(..., ge=100, description="Living area in square feet")
    sqft_lot: int = Field(..., ge=0, description="Lot size in square feet")
    floors: float = Field(..., ge=1.0, le=5.0, description="Number of floors")
    year_built: int = Field(..., ge=1800, le=2025, description="Year property was built")
    year_renovated: int = Field(default=0, ge=0, le=2025, description="Year of last renovation (0 if never)")
    latitude: float = Field(..., ge=-90.0, le=90.0, description="Latitude coordinate")
    longitude: float = Field(..., ge=-180.0, le=180.0, description="Longitude coordinate")
    property_type: str = Field(..., description="Type of property")
    neighborhood: str = Field(..., description="Neighborhood name")
    condition: str = Field(..., description="Property condition")
    view_quality: str = Field(..., description="Quality of view")

    @validator('property_type')
    def validate_property_type(cls, v):
        allowed = ['Single Family', 'Townhouse', 'Condo', 'Multi-Family']
        if v not in allowed:
            raise ValueError(f'property_type must be one of: {allowed}')
        return v

    @validator('condition')
    def validate_condition(cls, v):
        allowed = ['Poor', 'Fair', 'Average', 'Good', 'Excellent']
        if v not in allowed:
            raise ValueError(f'condition must be one of: {allowed}')
        return v

    @validator('view_quality')
    def validate_view_quality(cls, v):
        allowed = ['None', 'Fair', 'Good', 'Excellent']
        if v not in allowed:
            raise ValueError(f'view_quality must be one of: {allowed}')
        return v

    class Config:
        schema_extra = {
            "example": {
                "bedrooms": 3,
                "bathrooms": 2.5,
                "sqft_living": 2000,
                "sqft_lot": 5000,
                "floors": 2.0,
                "year_built": 2005,
                "year_renovated": 0,
                "latitude": 47.5112,
                "longitude": -122.257,
                "property_type": "Single Family",
                "neighborhood": "Downtown",
                "condition": "Good",
                "view_quality": "Fair"
            }
        }


class PredictionRequest(BaseModel):
    """Request for property valuation prediction"""
    property_id: Optional[str] = Field(None, description="Optional property identifier")
    features: PropertyFeatures
    description: Optional[str] = Field(None, max_length=1000, description="Property description")
    image_url: Optional[str] = Field(None, description="URL to property image")
    use_ensemble: bool = Field(default=True, description="Use ensemble model (vs single model)")

    class Config:
        schema_extra = {
            "example": {
                "property_id": "PROP_001",
                "features": {
                    "bedrooms": 3,
                    "bathrooms": 2.5,
                    "sqft_living": 2000,
                    "sqft_lot": 5000,
                    "floors": 2.0,
                    "year_built": 2005,
                    "year_renovated": 0,
                    "latitude": 47.5112,
                    "longitude": -122.257,
                    "property_type": "Single Family",
                    "neighborhood": "Downtown",
                    "condition": "Good",
                    "view_quality": "Fair"
                },
                "description": "Beautiful modern home with open floor plan and granite countertops",
                "image_url": "https://example.com/property.jpg",
                "use_ensemble": True
            }
        }


class ModelBreakdown(BaseModel):
    """Individual model predictions"""
    tabular: Optional[float] = Field(None, description="Tabular model prediction")
    image: Optional[float] = Field(None, description="Image model prediction")
    text: Optional[float] = Field(None, description="Text model prediction")


class PredictionResponse(BaseModel):
    """Response with property valuation prediction"""
    property_id: Optional[str] = Field(None, description="Property identifier")
    predicted_price: float = Field(..., description="Predicted property value")
    confidence_score: Optional[float] = Field(None, ge=0.0, le=1.0, description="Prediction confidence")
    model_breakdown: Optional[ModelBreakdown] = Field(None, description="Individual model contributions")
    timestamp: datetime = Field(default_factory=datetime.now, description="Prediction timestamp")

    class Config:
        schema_extra = {
            "example": {
                "property_id": "PROP_001",
                "predicted_price": 525000.50,
                "confidence_score": 0.89,
                "model_breakdown": {
                    "tabular": 520000.00,
                    "image": 535000.00,
                    "text": 530000.00
                },
                "timestamp": "2025-01-15T10:30:00"
            }
        }


class BatchPredictionRequest(BaseModel):
    """Request for batch predictions"""
    predictions: List[PredictionRequest] = Field(..., max_items=100)

    @validator('predictions')
    def validate_batch_size(cls, v):
        if len(v) > 100:
            raise ValueError('Batch size cannot exceed 100 properties')
        return v


class BatchPredictionResponse(BaseModel):
    """Response with batch predictions"""
    predictions: List[PredictionResponse]
    total_count: int
    successful_count: int
    failed_count: int


class PropertyCreate(BaseModel):
    """Model for creating a new property record"""
    features: PropertyFeatures
    description: Optional[str] = None
    image_path: Optional[str] = None
    actual_price: Optional[float] = Field(None, gt=0, description="Actual sale price")


class PropertyResponse(BaseModel):
    """Response with property details"""
    id: str
    features: PropertyFeatures
    description: Optional[str] = None
    actual_price: Optional[float] = None
    predicted_price: Optional[float] = None
    created_at: datetime
    updated_at: datetime

    class Config:
        orm_mode = True


class HealthCheck(BaseModel):
    """Health check response"""
    status: str
    timestamp: datetime
    models_loaded: dict


class ErrorResponse(BaseModel):
    """Error response"""
    error: str
    detail: Optional[str] = None
    timestamp: datetime = Field(default_factory=datetime.now)


class GoogleSheetsRequest(BaseModel):
    """Request for processing properties from Google Sheets"""
    sheet_url: str = Field(..., description="Google Sheets URL or Sheet ID")
    sheet_name: Optional[str] = Field("Sheet1", description="Name of the sheet/tab to read")
    start_row: int = Field(2, ge=1, description="Row to start reading data (1-indexed, default skips header)")
    write_back: bool = Field(True, description="Write predictions back to the sheet")
    use_ensemble: bool = Field(True, description="Use ensemble model for predictions")
    credentials_path: Optional[str] = Field(None, description="Path to Google service account credentials JSON")

    class Config:
        schema_extra = {
            "example": {
                "sheet_url": "https://docs.google.com/spreadsheets/d/1ABC...XYZ/edit",
                "sheet_name": "Sheet1",
                "start_row": 2,
                "write_back": True,
                "use_ensemble": True,
                "credentials_path": None
            }
        }


class GoogleSheetsResponse(BaseModel):
    """Response from Google Sheets processing"""
    sheet_id: str
    total_properties: int
    successful_predictions: int
    failed_predictions: int
    predictions: List[PredictionResponse]
    written_back: bool = False
    timestamp: datetime = Field(default_factory=datetime.now)

    class Config:
        schema_extra = {
            "example": {
                "sheet_id": "1ABC...XYZ",
                "total_properties": 10,
                "successful_predictions": 10,
                "failed_predictions": 0,
                "predictions": [],
                "written_back": True,
                "timestamp": "2025-01-15T10:30:00"
            }
        }
