"""
Flip Calculator API Routes
"""
from fastapi import APIRouter, HTTPException
from typing import List
import sys
import os

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from app.models.flip_calculator_models import (
    FlipCalculatorInput,
    FlipCalculatorResult
)
from app.services.flip_calculator import calculate_flip_deal


router = APIRouter(
    prefix="/api/flip",
    tags=["flip-calculator"],
    responses={404: {"description": "Not found"}},
)


@router.post("/calculate", response_model=FlipCalculatorResult)
async def calculate_flip(input_data: FlipCalculatorInput):
    """
    Calculate flip deal profitability

    Takes property details and returns complete flip analysis including:
    - Acquisition costs
    - Renovation costs ($45/sqft standard)
    - Holding costs (interest, taxes, insurance, utilities)
    - Selling costs (commissions, closing costs)
    - Profit analysis and ROI
    - Deal quality indicators

    ## Example Request:
    ```json
    {
        "property_address": "123 Main St",
        "city": "Atlanta",
        "zip_code": "30303",
        "sqft_living": 2000,
        "purchase_price": 300000,
        "arv": 450000,
        "hold_time_months": 5
    }
    ```

    ## Returns:
    Complete flip analysis with all cost breakdowns and profit metrics
    """
    try:
        result = calculate_flip_deal(input_data)
        return result
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error calculating flip deal: {str(e)}")


@router.post("/calculate/batch", response_model=List[FlipCalculatorResult])
async def calculate_flip_batch(properties: List[FlipCalculatorInput]):
    """
    Calculate flip deals for multiple properties at once

    Useful for analyzing multiple opportunities simultaneously.
    Maximum 100 properties per request.

    ## Example Request:
    ```json
    [
        {
            "property_address": "123 Main St",
            "city": "Atlanta",
            "zip_code": "30303",
            "sqft_living": 2000,
            "purchase_price": 300000,
            "arv": 450000,
            "hold_time_months": 5
        },
        {
            "property_address": "456 Oak Ave",
            "city": "Atlanta",
            "zip_code": "30305",
            "sqft_living": 1800,
            "purchase_price": 250000,
            "arv": 400000,
            "hold_time_months": 6
        }
    ]
    ```
    """
    if len(properties) > 100:
        raise HTTPException(status_code=400, detail="Maximum 100 properties per batch request")

    try:
        results = []
        for prop_input in properties:
            result = calculate_flip_deal(prop_input)
            results.append(result)
        return results
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error calculating batch: {str(e)}")


@router.get("/defaults")
async def get_defaults():
    """
    Get default values for flip calculator parameters

    Returns the standard assumptions used in calculations:
    - Repair cost per sqft
    - Interest rates
    - Closing costs
    - Commission rates
    - Hold time
    - etc.
    """
    return {
        "repair_cost_per_sqft": 45,
        "hold_time_months": 5,
        "loan_to_cost_percentage": 1.0,
        "interest_rate_annual": 0.10,
        "loan_points": 0.01,
        "monthly_hoa_maintenance": 150,
        "annual_property_tax": 503,
        "monthly_insurance": 130,
        "monthly_utilities": 80,
        "staging_marketing": 100,
        "closing_cost_rate_buying": 0.01,
        "closing_cost_rate_selling": 0.01,
        "seller_credit_rate": 0.02,
        "listing_commission_rate": 0.01,
        "buyer_commission_rate": 0.025,
        "minimum_roi_threshold": 20
    }


@router.post("/quick-analysis")
async def quick_analysis(
    purchase_price: float,
    sqft_living: int,
    arv: float
):
    """
    Quick flip analysis with minimal inputs

    Uses default values for all other parameters.
    Good for rapid deal screening.

    ## Parameters:
    - purchase_price: Offer/purchase price
    - sqft_living: Square footage
    - arv: After Repair Value

    ## Example:
    ```
    POST /api/flip/quick-analysis?purchase_price=300000&sqft_living=2000&arv=450000
    ```
    """
    try:
        input_data = FlipCalculatorInput(
            property_address="Quick Analysis",
            city="",
            zip_code="00000",
            sqft_living=sqft_living,
            purchase_price=purchase_price,
            arv=arv
        )
        result = calculate_flip_deal(input_data)

        # Return simplified response
        return {
            "purchase_price": purchase_price,
            "sqft": sqft_living,
            "arv": arv,
            "repair_cost": result.renovation.repair_cost,
            "total_costs": result.profit_analysis.total_all_costs,
            "profit": result.profit_analysis.gross_profit,
            "roi_percent": result.profit_analysis.roi_percent,
            "is_profitable": result.profit_analysis.is_profitable,
            "meets_20pct_roi": result.profit_analysis.meets_minimum_roi,
            "meets_70pct_rule": result.profit_analysis.meets_70_percent_rule,
            "max_offer_70_rule": result.max_offer_70_rule,
            "deal_quality": "GOOD" if result.profit_analysis.meets_minimum_roi else
                           "MAYBE" if result.profit_analysis.is_profitable else "PASS"
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error in quick analysis: {str(e)}")
