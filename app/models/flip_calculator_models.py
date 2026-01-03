"""
Flip Calculator Data Models
Based on Bird Dog Flip Calculator analysis
"""
from pydantic import BaseModel, Field, computed_field
from typing import Optional


class FlipCalculatorInput(BaseModel):
    """
    Input fields for flip deal analysis
    All the data the user needs to provide
    """

    # ==========================================
    # PROPERTY INFORMATION
    # ==========================================
    property_address: str = Field(..., description="Full property address")
    city: str = Field(..., description="City")
    zip_code: str = Field(..., description="ZIP code")
    sqft_living: int = Field(..., ge=100, description="Living area square footage")

    # ==========================================
    # DEAL PARAMETERS
    # ==========================================
    purchase_price: float = Field(..., gt=0, description="Purchase/offer price")
    arv: float = Field(..., gt=0, description="After Repair Value (estimated sale price)")
    hold_time_months: int = Field(default=5, ge=1, le=24, description="Expected hold time in months")

    # ==========================================
    # ACQUISITION COSTS (User Inputs)
    # ==========================================
    reinstatement_misc: float = Field(default=0, ge=0, description="Reinstatement or misc acquisition costs")
    closing_cost_rate_buying: float = Field(default=0.01, ge=0, le=0.1, description="Closing costs as % of purchase (default 1%)")

    # ==========================================
    # RENOVATION COSTS (User Inputs)
    # ==========================================
    # Note: repair_cost calculated automatically at $45/sqft
    repair_cost_override: Optional[float] = Field(default=None, ge=0, description="Override auto repair calc (optional)")
    monthly_hoa_maintenance: float = Field(default=150, ge=0, description="Monthly HOA/maintenance/pool/landscape")

    # ==========================================
    # HOLDING COSTS (User Inputs)
    # ==========================================
    loan_to_cost_percentage: float = Field(default=1.0, ge=0, le=1.0, description="Loan-to-cost ratio (1.0 = 100%)")
    interest_rate_annual: float = Field(default=0.10, ge=0, le=0.3, description="Annual interest rate (default 10% for hard money)")
    loan_points: float = Field(default=0.01, ge=0, le=0.1, description="Loan origination points (default 1%)")
    annual_property_tax: float = Field(default=503, ge=0, description="Annual property tax")
    monthly_insurance: float = Field(default=130, ge=0, description="Monthly insurance cost")
    monthly_utilities: float = Field(default=80, ge=0, description="Monthly utilities cost")

    # ==========================================
    # SELLING COSTS (User Inputs)
    # ==========================================
    staging_marketing: float = Field(default=100, ge=0, description="Staging, marketing, photos cost")
    closing_cost_rate_selling: float = Field(default=0.01, ge=0, le=0.1, description="Selling closing costs as % of ARV")
    seller_credit_rate: float = Field(default=0.02, ge=0, le=0.1, description="Seller credit/concessions as % of ARV")
    listing_commission_rate: float = Field(default=0.01, ge=0, le=0.1, description="Listing agent commission %")
    buyer_commission_rate: float = Field(default=0.025, ge=0, le=0.1, description="Buyer agent commission %")

    class Config:
        json_schema_extra = {
            "example": {
                "property_address": "7256 E Mckellips",
                "city": "Scottsdale",
                "zip_code": "85257",
                "sqft_living": 1800,
                "purchase_price": 450000,
                "arv": 570000,
                "hold_time_months": 5,
                "reinstatement_misc": 0,
                "closing_cost_rate_buying": 0.01,
                "repair_cost_override": None,
                "monthly_hoa_maintenance": 150,
                "loan_to_cost_percentage": 1.0,
                "interest_rate_annual": 0.10,
                "loan_points": 0.01,
                "annual_property_tax": 503,
                "monthly_insurance": 130,
                "monthly_utilities": 80,
                "staging_marketing": 100,
                "closing_cost_rate_selling": 0.01,
                "seller_credit_rate": 0.02,
                "listing_commission_rate": 0.01,
                "buyer_commission_rate": 0.025
            }
        }


class AcquisitionCosts(BaseModel):
    """Calculated acquisition/buying costs"""
    purchase_price: float
    reinstatement_misc: float
    closing_costs: float  # = purchase_price * rate
    total_acquisition: float  # = sum of above

    class Config:
        json_schema_extra = {
            "example": {
                "purchase_price": 450000,
                "reinstatement_misc": 0,
                "closing_costs": 4500,
                "total_acquisition": 454500
            }
        }


class RenovationCosts(BaseModel):
    """Calculated renovation costs"""
    repair_cost: float  # = sqft * $45 (or override)
    monthly_maintenance_total: float  # = monthly_rate * hold_months
    total_renovation: float  # = sum of above

    class Config:
        json_schema_extra = {
            "example": {
                "repair_cost": 81000,  # 1800 sqft * $45
                "monthly_maintenance_total": 750,  # $150 * 5 months
                "total_renovation": 81750
            }
        }


class HoldingCosts(BaseModel):
    """Calculated holding costs"""
    loan_amount: float  # = total_acquisition + total_renovation (or LTC%)
    interest_payment: float  # = (loan_amount * rate / 12) * months
    loan_origination_points: float  # = purchase_price * points
    property_tax_prorated: float  # = annual_tax / 12 * months
    insurance_total: float  # = monthly_insurance * months
    utilities_total: float  # = monthly_utilities * months
    total_holding: float  # = sum of above

    class Config:
        json_schema_extra = {
            "example": {
                "loan_amount": 536250,
                "interest_payment": 22343.75,
                "loan_origination_points": 4500,
                "property_tax_prorated": 209.58,
                "insurance_total": 650,
                "utilities_total": 400,
                "total_holding": 28103.33
            }
        }


class SellingCosts(BaseModel):
    """Calculated selling costs"""
    staging_marketing: float
    closing_costs: float  # = arv * rate
    seller_credit: float  # = arv * rate
    listing_commission: float  # = arv * rate
    buyer_commission: float  # = arv * rate
    total_selling: float  # = sum of above

    class Config:
        json_schema_extra = {
            "example": {
                "staging_marketing": 100,
                "closing_costs": 5700,
                "seller_credit": 11400,
                "listing_commission": 5700,
                "buyer_commission": 14250,
                "total_selling": 37150
            }
        }


class ProfitAnalysis(BaseModel):
    """Final profit and ROI analysis"""
    total_all_costs: float  # = acquisition + renovation + holding + selling
    gross_profit: float  # = arv - total_all_costs
    profit_margin_percent: float  # = (gross_profit / arv) * 100
    roi_percent: float  # = (gross_profit / total_all_costs) * 100
    cash_needed: float  # = total_all_costs - loan_amount (if financed)

    # Deal quality indicators
    is_profitable: bool  # = gross_profit > 0
    meets_minimum_roi: bool  # = roi_percent >= 20
    meets_70_percent_rule: bool  # = purchase_price <= (arv * 0.7 - repair_cost)

    class Config:
        json_schema_extra = {
            "example": {
                "total_all_costs": 601503.33,
                "gross_profit": -31503.33,
                "profit_margin_percent": -5.53,
                "roi_percent": -5.24,
                "cash_needed": 65253.33,
                "is_profitable": False,
                "meets_minimum_roi": False,
                "meets_70_percent_rule": False
            }
        }


class FlipCalculatorResult(BaseModel):
    """
    Complete flip calculator output
    All calculated costs and profit analysis
    """

    # Input echo (for reference)
    property_address: str
    city: str
    zip_code: str
    sqft_living: int
    purchase_price: float
    arv: float
    hold_time_months: int

    # Calculated cost breakdowns
    acquisition: AcquisitionCosts
    renovation: RenovationCosts
    holding: HoldingCosts
    selling: SellingCosts

    # Final analysis
    profit_analysis: ProfitAnalysis

    # Recommendations
    max_offer_70_rule: float  # = (arv * 0.7) - repair_cost
    recommended_arv_for_profit: float  # = total_costs / 0.85 (assume 15% profit margin)

    class Config:
        json_schema_extra = {
            "example": {
                "property_address": "7256 E Mckellips",
                "city": "Scottsdale",
                "zip_code": "85257",
                "sqft_living": 1800,
                "purchase_price": 450000,
                "arv": 570000,
                "hold_time_months": 5,
                "acquisition": {
                    "purchase_price": 450000,
                    "reinstatement_misc": 0,
                    "closing_costs": 4500,
                    "total_acquisition": 454500
                },
                "renovation": {
                    "repair_cost": 81000,
                    "monthly_maintenance_total": 750,
                    "total_renovation": 81750
                },
                "holding": {
                    "loan_amount": 536250,
                    "interest_payment": 22343.75,
                    "loan_origination_points": 4500,
                    "property_tax_prorated": 209.58,
                    "insurance_total": 650,
                    "utilities_total": 400,
                    "total_holding": 28103.33
                },
                "selling": {
                    "staging_marketing": 100,
                    "closing_costs": 5700,
                    "seller_credit": 11400,
                    "listing_commission": 5700,
                    "buyer_commission": 14250,
                    "total_selling": 37150
                },
                "profit_analysis": {
                    "total_all_costs": 601503.33,
                    "gross_profit": -31503.33,
                    "profit_margin_percent": -5.53,
                    "roi_percent": -5.24,
                    "cash_needed": 65253.33,
                    "is_profitable": False,
                    "meets_minimum_roi": False,
                    "meets_70_percent_rule": False
                },
                "max_offer_70_rule": 318000,
                "recommended_arv_for_profit": 707650.98
            }
        }


# Constants
REPAIR_COST_PER_SQFT = 45  # Standard repair cost assumption
DEFAULT_HOLD_MONTHS = 5
DEFAULT_HARD_MONEY_RATE = 0.10  # 10% annual
DEFAULT_LOAN_POINTS = 0.01  # 1%
MINIMUM_ROI_THRESHOLD = 20  # 20% ROI for "good deal"
