"""
Property management API routes
"""

from fastapi import APIRouter, HTTPException, status, Query
from typing import List, Optional
from datetime import datetime
import uuid

from app.models.property_models import (
    PropertyCreate,
    PropertyResponse,
    PropertyFeatures
)

router = APIRouter(prefix="/properties", tags=["properties"])

# In-memory storage (replace with database in production)
properties_db = {}


@router.post("/", response_model=PropertyResponse, status_code=status.HTTP_201_CREATED)
async def create_property(property_data: PropertyCreate):
    """
    Create a new property record.

    This endpoint stores property information for future reference and comparison.
    """
    property_id = str(uuid.uuid4())
    now = datetime.now()

    property_record = {
        "id": property_id,
        "features": property_data.features,
        "description": property_data.description,
        "actual_price": property_data.actual_price,
        "predicted_price": None,  # Set when prediction is made
        "created_at": now,
        "updated_at": now
    }

    properties_db[property_id] = property_record

    return PropertyResponse(**property_record)


@router.get("/{property_id}", response_model=PropertyResponse)
async def get_property(property_id: str):
    """
    Retrieve a specific property by ID.
    """
    if property_id not in properties_db:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Property {property_id} not found"
        )

    return PropertyResponse(**properties_db[property_id])


@router.get("/", response_model=List[PropertyResponse])
async def list_properties(
    skip: int = Query(0, ge=0, description="Number of records to skip"),
    limit: int = Query(100, ge=1, le=1000, description="Maximum number of records to return"),
    neighborhood: Optional[str] = Query(None, description="Filter by neighborhood"),
    min_price: Optional[float] = Query(None, ge=0, description="Minimum price filter"),
    max_price: Optional[float] = Query(None, ge=0, description="Maximum price filter")
):
    """
    List all properties with optional filtering.

    Supports pagination and filtering by neighborhood and price range.
    """
    properties = list(properties_db.values())

    # Apply filters
    if neighborhood:
        properties = [
            p for p in properties
            if p['features'].neighborhood == neighborhood
        ]

    if min_price is not None:
        properties = [
            p for p in properties
            if p['actual_price'] and p['actual_price'] >= min_price
        ]

    if max_price is not None:
        properties = [
            p for p in properties
            if p['actual_price'] and p['actual_price'] <= max_price
        ]

    # Apply pagination
    total = len(properties)
    properties = properties[skip:skip + limit]

    return [PropertyResponse(**p) for p in properties]


@router.put("/{property_id}", response_model=PropertyResponse)
async def update_property(property_id: str, property_data: PropertyCreate):
    """
    Update an existing property record.
    """
    if property_id not in properties_db:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Property {property_id} not found"
        )

    existing = properties_db[property_id]
    existing.update({
        "features": property_data.features,
        "description": property_data.description,
        "actual_price": property_data.actual_price,
        "updated_at": datetime.now()
    })

    return PropertyResponse(**existing)


@router.delete("/{property_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_property(property_id: str):
    """
    Delete a property record.
    """
    if property_id not in properties_db:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Property {property_id} not found"
        )

    del properties_db[property_id]
    return None


@router.get("/stats/summary")
async def get_properties_summary():
    """
    Get summary statistics for all properties.
    """
    if not properties_db:
        return {
            "total_properties": 0,
            "avg_price": 0,
            "min_price": 0,
            "max_price": 0
        }

    prices = [p['actual_price'] for p in properties_db.values() if p['actual_price']]

    if not prices:
        return {
            "total_properties": len(properties_db),
            "avg_price": 0,
            "min_price": 0,
            "max_price": 0
        }

    return {
        "total_properties": len(properties_db),
        "avg_price": sum(prices) / len(prices),
        "min_price": min(prices),
        "max_price": max(prices),
        "neighborhoods": list(set(p['features'].neighborhood for p in properties_db.values()))
    }
