"""
Google Sheets integration API routes - Updated for MLS Data
Uses area-specific ARV multiples from flip analysis
"""

from fastapi import APIRouter, HTTPException, status
from typing import Optional, List
import gspread
from google.oauth2.service_account import Credentials
from google.auth.exceptions import GoogleAuthError
import pandas as pd
import re
import os
from datetime import datetime

from app.models.property_models import (
    GoogleSheetsRequest,
    GoogleSheetsResponse,
)
from app.models.flip_calculator_models import FlipCalculatorInput
from app.services.flip_calculator import calculate_flip_deal
from app.services.rentcast_api import RentcastAPIService

router = APIRouter(prefix="/sheets", tags=["google-sheets"])

# Global variable for loaded models (set from main.py)
stacker_model = None

# Load area-specific ARV multiples
arv_multiples_zip = None
arv_multiples_city = None


def load_arv_multiples():
    """Load area-specific ARV multiples from CSV files"""
    global arv_multiples_zip, arv_multiples_city

    try:
        zip_file = 'data/arv_multiples_by_area.csv'
        city_file = 'data/arv_multiples_by_city.csv'

        if os.path.exists(zip_file):
            arv_multiples_zip = pd.read_csv(zip_file)
            print(f"Loaded ARV multiples for {len(arv_multiples_zip)} zip codes")

        if os.path.exists(city_file):
            arv_multiples_city = pd.read_csv(city_file)
            print(f"Loaded ARV multiples for {len(arv_multiples_city)} cities")

        if arv_multiples_zip is None and arv_multiples_city is None:
            print("WARNING: No ARV multiples loaded. Using defaults.")
    except Exception as e:
        print(f"Error loading ARV multiples: {e}")


def set_model(model):
    """Set the global model instance"""
    global stacker_model
    stacker_model = model
    # Load ARV multiples when model is set
    load_arv_multiples()


def get_arv_multiple(zipcode, city, scenario='moderate'):
    """
    Get ARV multiple for a specific zip/city

    Args:
        zipcode: 5-digit zip code
        city: City name
        scenario: 'conservative', 'moderate', or 'aggressive'

    Returns:
        ARV multiple (float)
    """
    global arv_multiples_zip, arv_multiples_city

    # Try zip code first (more specific)
    if arv_multiples_zip is not None and zipcode:
        zip_match = arv_multiples_zip[arv_multiples_zip['Zip'] == str(zipcode)]
        if len(zip_match) > 0:
            col_name = f'{scenario.capitalize()}_Multiple'
            return float(zip_match.iloc[0][col_name])

    # Fall back to city level
    if arv_multiples_city is not None and city:
        city_match = arv_multiples_city[arv_multiples_city['City'].str.lower() == str(city).lower()]
        if len(city_match) > 0:
            if scenario == 'conservative':
                return float(city_match.iloc[0]['Conservative'])
            elif scenario == 'moderate':
                return float(city_match.iloc[0]['Median'])
            elif scenario == 'aggressive':
                return float(city_match.iloc[0]['Aggressive'])

    # Default to overall averages if no match
    defaults = {
        'conservative': 2.59,
        'moderate': 2.82,
        'aggressive': 3.22
    }
    return defaults.get(scenario, 2.82)


def extract_sheet_id(sheet_url: str) -> str:
    """Extract Google Sheets ID from URL or return as-is if already an ID"""
    pattern = r'/spreadsheets/d/([a-zA-Z0-9-_]+)'
    match = re.search(pattern, sheet_url)

    if match:
        return match.group(1)

    return sheet_url


def get_google_sheets_client(credentials_path: Optional[str] = None):
    """
    Get authenticated Google Sheets client.

    Args:
        credentials_path: Path to service account credentials JSON file or JSON string

    Returns:
        Authenticated gspread client
    """
    import json

    scopes = [
        'https://www.googleapis.com/auth/spreadsheets',
        'https://www.googleapis.com/auth/drive'
    ]

    creds_data = credentials_path or os.getenv('GOOGLE_SHEETS_CREDENTIALS')

    if not creds_data:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Google Sheets credentials not provided. Set GOOGLE_SHEETS_CREDENTIALS environment variable."
        )

    try:
        # Try to parse as JSON first (for environment variable)
        try:
            creds_dict = json.loads(creds_data)
            credentials = Credentials.from_service_account_info(creds_dict, scopes=scopes)
        except (json.JSONDecodeError, ValueError):
            # If not JSON, treat as file path
            if os.path.exists(creds_data):
                credentials = Credentials.from_service_account_file(creds_data, scopes=scopes)
            else:
                raise HTTPException(
                    status_code=status.HTTP_404_NOT_FOUND,
                    detail=f"Credentials file not found: {creds_data}"
                )

        client = gspread.authorize(credentials)
        return client
    except GoogleAuthError as e:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=f"Google authentication failed: {str(e)}"
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to initialize Google Sheets client: {str(e)}"
        )


def clean_price(val):
    """Convert price string to numeric"""
    try:
        if not val or str(val).strip() == '':
            return None
        cleaned = re.sub(r'[,$]', '', str(val))
        return float(cleaned)
    except:
        return None


def safe_int(val, default=0):
    """Safely convert to int, handling commas"""
    try:
        if not val or not str(val).strip():
            return default
        # Remove commas before converting
        clean_val = str(val).strip().replace(',', '')
        return int(float(clean_val))
    except:
        return default


def clean_zip(val):
    """Extract 5-digit zip code"""
    try:
        if not val:
            return None
        # Extract first 5 digits
        zip_str = str(val).strip()
        match = re.search(r'(\d{5})', zip_str)
        if match:
            return match.group(1)
        return None
    except:
        return None


@router.post("/predict", response_model=GoogleSheetsResponse, status_code=status.HTTP_200_OK)
async def predict_from_sheets(request: GoogleSheetsRequest):
    """
    Process MLS listings from Google Sheets with Area-Specific ARV Analysis.

    ## Expected Sheet Format (MLS Data)

    Required columns (names are flexible):
    - **City** - Property city
    - **Zip** - 5-digit zip code
    - **List Price** (or Price, MLS Amount) - Listing price
    - **Days On Market** (or DOM) - Days listed

    Optional columns (will be included in analysis but not required):
    - Street Number, Street Name, Address
    - Parcel Number, County Code, MLS #
    - Owner info, Agent info, etc.

    ## Output Columns

    Writes 35 columns to the sheet (columns X through BK):

    **ARV Analysis (X-AB):**
    - **X: Deal Status** - GOOD DEAL, MAYBE, or NO DEAL
    - **Y: ARV (Conservative)** - Conservative ARV estimate (25th percentile)
    - **Z: ARV (Moderate)** - Recommended ARV estimate (median)
    - **AA: ARV (Aggressive)** - Optimistic ARV estimate (75th percentile)
    - **AB: Confidence** - HIGH, MEDIUM, or LOW based on data source

    **Flip Calculator Analysis (AC-BK) - Uses $45/sqft for repairs:**
    - **AC-AE**: Quick indicators (Profitable, Meets 20% ROI, Meets 70% Rule)
    - **AF-AH**: Key metrics (ROI %, Gross Profit, Profit Margin %)
    - **AI-AM**: Core numbers (Purchase, ARV, Total Costs, Cash Needed, Max Offer)
    - **AN-AQ**: Cost summaries (Acquisition, Renovation, Holding, Selling)
    - **AR-BY**: Detailed breakdowns (Repairs, Loan costs, Commissions, etc.)
    - **BK**: Recommended ARV for profitability

    ## ARV Calculation

    Uses area-specific multiples derived from 598 actual flip transactions:
    - Zip code level: Most accurate (if available)
    - City level: Fallback if zip not found
    - Default: Overall market average (2.82x)

    Formula: ARV = Assessed Value × Area Multiple

    If assessed value not available, estimates as 35% of list price.

    ## Deal Criteria

    - **GOOD DEAL**: List price ≤ 50% of moderate ARV
    - **MAYBE**: List price ≤ 60% of moderate ARV
    - **NO DEAL**: List price > 60% of moderate ARV

    ## Authentication

    Requires Google service account credentials with Editor access to the sheet.
    Set via GOOGLE_SHEETS_CREDENTIALS environment variable.
    """

    # Get Google Sheets client
    client = get_google_sheets_client(request.credentials_path)

    # Extract sheet ID and open spreadsheet
    sheet_id = extract_sheet_id(request.sheet_url)

    try:
        spreadsheet = client.open_by_key(sheet_id)

        # Get first worksheet (since user said only one tab)
        worksheet = spreadsheet.get_worksheet(0)

    except gspread.SpreadsheetNotFound:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Spreadsheet not found: {sheet_id}. Ensure the service account has Editor access."
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to access spreadsheet: {str(e)}"
        )

    # Read all data from sheet
    try:
        all_values = worksheet.get_all_values()

        if len(all_values) < request.start_row:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Sheet has fewer rows than start_row ({request.start_row})"
            )

        # Get header row and find column positions
        header_row = all_values[0] if all_values else []

        # Find column indices by name (case-insensitive, flexible)
        def find_column(header_row, possible_names):
            """Find column index by trying multiple possible header names"""
            header_lower = [h.lower().strip() for h in header_row]
            for name in possible_names:
                name_lower = name.lower()
                if name_lower in header_lower:
                    return header_lower.index(name_lower)
            return None

        col_city = find_column(header_row, ['city'])
        col_zip = find_column(header_row, ['zip', 'zipcode', 'zip code'])
        col_list_price = find_column(header_row, ['list price', 'price', 'mls amount', 'sale price'])
        col_dom = find_column(header_row, ['days on market', 'dom', 'days on mkt'])
        col_assessed = find_column(header_row, ['total assessed value', 'assessed value', 'tax assessed value'])
        col_sqft = find_column(header_row, ['building sqft', 'sqft', 'square feet', 'sq ft', 'sqft living', 'square footage'])
        col_address = find_column(header_row, ['address', 'street address', 'property address'])

        # Verify we have required columns
        missing_cols = []
        if col_city is None:
            missing_cols.append('City')
        if col_zip is None:
            missing_cols.append('Zip')
        if col_list_price is None:
            missing_cols.append('List Price (or Price/MLS Amount)')

        if missing_cols:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Missing required columns: {', '.join(missing_cols)}. Found: {', '.join(header_row[:10])}"
            )

        # Get data rows (skip header)
        data_rows = all_values[request.start_row - 1:]

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to read sheet data: {str(e)}"
        )

    # Initialize Rentcast API service
    rentcast = RentcastAPIService()

    # Process each row and calculate area-specific ARV
    successful = 0
    failed = 0
    results_to_write = []

    for idx, row in enumerate(data_rows, start=request.start_row):
        try:
            # Extract data from the correct columns
            city = str(row[col_city]).strip() if col_city < len(row) and row[col_city] else None
            zipcode = clean_zip(row[col_zip]) if col_zip < len(row) else None
            list_price = clean_price(row[col_list_price]) if col_list_price < len(row) else None
            days_on_market = safe_int(row[col_dom]) if col_dom is not None and col_dom < len(row) else 0
            assessed_value = clean_price(row[col_assessed]) if col_assessed is not None and col_assessed < len(row) else None

            # Skip if no list price
            if not list_price or list_price <= 0:
                failed += 1
                results_to_write.append(["ERROR - No Price", "", "", "", ""])
                continue

            # Skip if no city or zip
            if not city and not zipcode:
                failed += 1
                results_to_write.append(["ERROR - No Location", "", "", "", ""])
                continue

            # Estimate assessed value if not provided (typically 35% of market value)
            if not assessed_value or assessed_value <= 0:
                assessed_value = list_price * 0.35

            successful += 1

            # Calculate flip deal if sqft available
            flip_results = []
            sqft_source = "Sheet"

            # Get sqft from sheet
            sqft = 0
            if col_sqft is not None and col_sqft < len(row):
                sqft = safe_int(row[col_sqft], 0)

            # Now process flip calculator if we have sqft
            if sqft > 0:
                try:
                    # Get address if available
                    address = str(row[col_address]).strip() if col_address is not None and col_address < len(row) else "Property"

                    # Create flip calculator input
                    # Use list price as both purchase price and ARV estimate
                    flip_input = FlipCalculatorInput(
                        property_address=address,
                        city=city or "",
                        zip_code=zipcode or "00000",
                        sqft_living=sqft,
                        purchase_price=list_price,
                        arv=list_price,  # Use list price as ARV estimate
                        hold_time_months=5
                    )

                    # Calculate flip deal
                    flip_result = calculate_flip_deal(flip_input)

                    # Format flip results (32 columns: 2 sqft + 30 flip)
                    flip_results = [
                        # Sqft info (2 columns)
                        str(sqft),
                        sqft_source,
                        # Quick indicators
                        "YES" if flip_result.profit_analysis.is_profitable else "NO",
                        "YES" if flip_result.profit_analysis.meets_minimum_roi else "NO",
                        "YES" if flip_result.profit_analysis.meets_70_percent_rule else "NO",
                        f"{flip_result.profit_analysis.roi_percent:.1f}%",
                        f"${flip_result.profit_analysis.gross_profit:,.0f}",
                        f"{flip_result.profit_analysis.profit_margin_percent:.1f}%",
                        # Key numbers
                        f"${flip_result.acquisition.purchase_price:,.0f}",
                        f"${flip_result.arv:,.0f}",
                        f"${flip_result.profit_analysis.total_all_costs:,.0f}",
                        f"${flip_result.profit_analysis.cash_needed:,.0f}",
                        f"${flip_result.max_offer_70_rule:,.0f}",
                        # Cost breakdowns
                        f"${flip_result.acquisition.total_acquisition:,.0f}",
                        f"${flip_result.renovation.total_renovation:,.0f}",
                        f"${flip_result.holding.total_holding:,.0f}",
                        f"${flip_result.selling.total_selling:,.0f}",
                        # Detailed costs
                        f"${flip_result.renovation.repair_cost:,.0f}",
                        f"${flip_result.acquisition.closing_costs:,.0f}",
                        f"${flip_result.renovation.monthly_maintenance_total:,.0f}",
                        f"${flip_result.holding.loan_amount:,.0f}",
                        f"${flip_result.holding.interest_payment:,.0f}",
                        f"${flip_result.holding.loan_origination_points:,.0f}",
                        f"${flip_result.holding.property_tax_prorated:,.0f}",
                        f"${flip_result.holding.insurance_total:,.0f}",
                        f"${flip_result.holding.utilities_total:,.0f}",
                        f"${flip_result.selling.staging_marketing:,.0f}",
                        f"${flip_result.selling.closing_costs:,.0f}",
                        f"${flip_result.selling.seller_credit:,.0f}",
                        f"${flip_result.selling.listing_commission:,.0f}",
                        f"${flip_result.selling.buyer_commission:,.0f}",
                        f"${flip_result.recommended_arv_for_profit:,.0f}",
                    ]

                    # Fetch Rentcast value estimate and create comparison columns
                    # Always show ARV_Needed even if Rentcast fails
                    arv_needed = flip_result.recommended_arv_for_profit

                    try:
                        address = str(row[col_address]).strip() if col_address is not None and col_address < len(row) else None
                        print(f"  Fetching Rentcast value for {address}, {city}")
                        market_value = rentcast.get_value_estimate(address, city, "GA", zipcode) if address else None

                        if market_value:
                            value_vs_arv = market_value - arv_needed
                            value_supports = "YES" if market_value >= arv_needed else "NO"

                            # Determine deal status
                            if market_value >= arv_needed * 1.05:  # 5% cushion
                                deal_status = "GOOD DEAL"
                            elif market_value >= arv_needed:
                                deal_status = "MAYBE"
                            else:
                                deal_status = "NO DEAL"

                            value_cols = [
                                deal_status,
                                f"${market_value:,.0f}",
                                f"${arv_needed:,.0f}",
                                f"${value_vs_arv:,.0f}",
                                value_supports
                            ]
                        else:
                            print(f"  Rentcast value not available")
                            value_cols = ["UNKNOWN", "Not Available", f"${arv_needed:,.0f}", "N/A", "N/A"]
                    except Exception as e:
                        print(f"  Error fetching Rentcast value: {e}")
                        value_cols = ["ERROR", "API Error", f"${arv_needed:,.0f}", "N/A", "N/A"]

                except Exception as e:
                    print(f"Error calculating flip for row {idx}: {e}")
                    # Add empty columns if error (5 value + 2 sqft + 30 flip = 37)
                    value_cols = ["ERROR", "", "", "", ""]
                    flip_results = ["", "", "ERROR"] + [""] * 29
            else:
                # No sqft data - add empty columns (5 value + 2 sqft + 30 flip = 37)
                value_cols = ["NO SQFT", "N/A", "N/A", "N/A", "N/A"]
                flip_results = ["0", "Missing", "N/A - No Sqft"] + [""] * 29

            # Format output (5 market value + 2 sqft + 30 flip = 37 total)
            results_to_write.append(value_cols + flip_results)

        except Exception as e:
            failed += 1
            # Add empty columns for failed rows (5 zest + 2 sqft + 30 flip = 37)
            results_to_write.append(["ERROR", str(e)[:30], "", "", ""] + [""] * 32)
            print(f"Error processing row {idx}: {e}")

    # Write results back to sheet if requested
    written_back = False
    if request.write_back and results_to_write:
        try:
            # Write to columns X through BM (37 columns: 5 value + 2 sqft + 30 flip)
            # First, add/update header row
            header_row_num = request.start_row - 1
            if header_row_num >= 1:
                headers = [
                    # Market value comparison columns (X-AB)
                    'Deal_Status', 'Market_Value', 'ARV_Needed', 'Market_Value_vs_ARV', 'Market_Supports_Deal',
                    # Sqft info columns (AC-AD)
                    'Sqft_Used', 'Sqft_Source',
                    # Flip calculator columns (AE-BM)
                    'Flip_Is_Profitable', 'Flip_Meets_20pct_ROI', 'Flip_Meets_70pct_Rule',
                    'Flip_ROI_Pct', 'Flip_Gross_Profit', 'Flip_Profit_Margin_Pct',
                    'Flip_Purchase_Price', 'Flip_ARV', 'Flip_Total_All_Costs', 'Flip_Cash_Needed', 'Flip_Max_Offer_70_Rule',
                    'Flip_Total_Acquisition', 'Flip_Total_Renovation', 'Flip_Total_Holding', 'Flip_Total_Selling',
                    'Flip_Repair_Cost', 'Flip_Closing_Costs_Buy', 'Flip_Monthly_Maintenance',
                    'Flip_Loan_Amount', 'Flip_Interest_Payment', 'Flip_Loan_Points',
                    'Flip_Property_Tax', 'Flip_Insurance', 'Flip_Utilities',
                    'Flip_Staging', 'Flip_Closing_Costs_Sell', 'Flip_Seller_Credit',
                    'Flip_Listing_Commission', 'Flip_Buyer_Commission', 'Flip_Recommended_ARV'
                ]
                worksheet.update(f'X{header_row_num}:BM{header_row_num}', [headers])

            # Write prediction results
            start_cell = f'X{request.start_row}'
            end_row = request.start_row + len(results_to_write) - 1
            end_cell = f'BM{end_row}'

            worksheet.update(f'{start_cell}:{end_cell}', results_to_write)
            written_back = True

        except Exception as e:
            print(f"Failed to write results back to sheet: {e}")
            # Don't fail the request if write-back fails

    return GoogleSheetsResponse(
        sheet_id=sheet_id,
        total_properties=len(data_rows),
        successful_predictions=successful,
        failed_predictions=failed,
        predictions=[],
        written_back=written_back,
        timestamp=datetime.now()
    )


@router.get("/health")
async def sheets_health():
    """Check if Google Sheets integration is healthy"""
    creds_path = os.getenv('GOOGLE_SHEETS_CREDENTIALS')

    # Check if ARV multiples are loaded
    multiples_loaded = arv_multiples_zip is not None or arv_multiples_city is not None

    return {
        "status": "ready",
        "credentials_configured": creds_path is not None,
        "arv_multiples_loaded": multiples_loaded,
        "zip_codes_available": len(arv_multiples_zip) if arv_multiples_zip is not None else 0,
        "cities_available": len(arv_multiples_city) if arv_multiples_city is not None else 0,
        "timestamp": datetime.now()
    }
