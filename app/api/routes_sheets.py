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
from app.services.zillow_api import ZillowAPIService

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


def check_sheet_permissions(client, sheet_id: str) -> dict:
    """
    Check if we have edit permissions on the Google Sheet.

    Args:
        client: Authenticated gspread client
        sheet_id: Google Sheets ID

    Returns:
        dict with 'can_edit' (bool) and 'message' (str)
    """
    try:
        # Try to get the spreadsheet
        spreadsheet = client.open_by_key(sheet_id)

        # Attempt to read sheet properties (requires at least view access)
        properties = spreadsheet.fetch_sheet_metadata()

        # Try a test write operation to verify edit permissions
        # We'll update the spreadsheet properties with the same title (no actual change)
        try:
            # This requires edit permission but doesn't actually change anything
            test_worksheet = spreadsheet.get_worksheet(0)
            # Attempting to get cell will work with view-only
            # But attempting to update (even with same value) requires edit
            test_cell = test_worksheet.acell('A1')

            # Try to update with same value - this will fail if view-only
            test_worksheet.update('A1', [[test_cell.value]])

            return {
                'can_edit': True,
                'message': 'Sheet has edit permissions'
            }
        except gspread.exceptions.APIError as e:
            # Check if it's a permission error
            if 'insufficient permissions' in str(e).lower() or 'permission denied' in str(e).lower():
                return {
                    'can_edit': False,
                    'message': 'Sheet is view-only. Please share the sheet with the service account with Editor permissions.'
                }
            # Re-raise if it's a different error
            raise

    except gspread.SpreadsheetNotFound:
        return {
            'can_edit': False,
            'message': f'Spreadsheet not found or not shared with service account'
        }
    except Exception as e:
        return {
            'can_edit': False,
            'message': f'Error checking permissions: {str(e)}'
        }


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
    - **Zillow URL** (or Zillow Link, URL, Property URL) - Direct Zillow property URL with ZPID (recommended for accurate Zestimate)
    - Street Number, Street Name, Address
    - Parcel Number, County Code, MLS #
    - Owner info, Agent info, etc.

    ## Output Columns

    Writes 8 columns to the sheet (columns X through AE):

    **Zestimate & ARV Analysis:**
    - **X: Zestimate** - Zillow property valuation (from API if URL provided, otherwise defaults to list price)
    - **Y: ARV (80%)** - After Repair Value = Zestimate × 0.80
    - **Z: ARV Needed** - ARV needed for 20% ROI from flip calculator
    - **AA: Deal Status** - GOOD DEAL, MAYBE, or NO DEAL (based on ARV 80% vs ARV Needed)
    - **AB: Market Supports Deal** - YES/NO if ARV (80%) supports the deal
    - **AC: Rehab Cost** - Total renovation costs from flip calculator
    - **AD: Total Cost** - Total all-in costs (purchase + renovation + holding + selling)
    - **AE: Maximum Allowable Offer** - 50% of ARV (80%) [MAO = ARV × 0.50]

    **Note on Flip Calculator:**
    Flip calculator parameters (repair costs, hold time, financing terms, etc.) are customizable
    via the web UI. The flip calculator runs internally for each property to calculate ARV Needed,
    but detailed flip metrics are not written to the sheet to keep output focused on market value
    analysis and deal quality indicators.

    ## Deal Quality Determination

    Deal quality is determined by comparing ARV (80% of Zestimate) against the calculated ARV Needed:

    - **GOOD DEAL**: ARV (80%) ≥ ARV Needed × 1.05 (5% cushion for profit)
    - **MAYBE**: ARV (80%) ≥ ARV Needed (breakeven or minimal profit)
    - **NO DEAL**: ARV (80%) < ARV Needed (insufficient value to support flip)

    ## Maximum Allowable Offer (MAO)

    MAO is calculated as 50% of the ARV (80% of Zestimate), providing a conservative entry point:
    - Formula: MAO = min(ARV (80%) × 0.50, List Price)
    - Base calculation: Zestimate × 0.80 × 0.50 = Zestimate × 0.40
    - **Important**: MAO is capped at the list price - never offer more than asking
    - This ensures sufficient margin for repairs, holding costs, and profit
    - Adjust offer based on property condition and market dynamics

    ## Square Footage Handling

    - **Sheet Data**: Uses Building Sqft column when available
    - **Zillow Fallback**: Automatically fetches sqft from Zillow API if missing
    - **Source Tracking**: Sqft_Source column shows data origin (Sheet or Zillow)

    ## Authentication

    Requires Google service account credentials with Editor access to the sheet.
    Set via GOOGLE_SHEETS_CREDENTIALS environment variable.
    """

    # Get Google Sheets client
    client = get_google_sheets_client(request.credentials_path)

    # Extract sheet ID
    sheet_id = extract_sheet_id(request.sheet_url)

    # Check permissions before processing
    permission_check = check_sheet_permissions(client, sheet_id)
    if not permission_check['can_edit']:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail=permission_check['message']
        )

    # Open spreadsheet (we already checked permissions, but keep error handling)
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
        col_bedrooms = find_column(header_row, ['bedrooms', 'beds', 'bed', 'br'])
        col_bathrooms = find_column(header_row, ['bathrooms', 'baths', 'bath', 'ba'])
        col_zillow_url = find_column(header_row, ['zillow url', 'zillow link', 'url', 'property url', 'zillow'])

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

    # Initialize Zillow API service
    zillow = ZillowAPIService()

    # Extract custom parameters or use defaults
    params = request.parameters
    print(f"DEBUG: Parameters received: {params}")

    if params:
        repair_cost_per_sqft = params.repair_cost_per_sqft if params.repair_cost_per_sqft is not None else 45
        hold_time_months = params.hold_time_months if params.hold_time_months is not None else 5
        interest_rate = params.interest_rate_annual if params.interest_rate_annual is not None else 0.10
        loan_points = params.loan_points if params.loan_points is not None else 0.01
        loan_to_cost = params.loan_to_cost_ratio if params.loan_to_cost_ratio is not None else 0.90
        monthly_hoa = params.monthly_hoa_maintenance if params.monthly_hoa_maintenance is not None else 150
        monthly_insurance = params.monthly_insurance if params.monthly_insurance is not None else 100
        monthly_utilities = params.monthly_utilities if params.monthly_utilities is not None else 150
        property_tax_rate = params.property_tax_rate_annual if params.property_tax_rate_annual is not None else 0.012
        closing_buy_pct = params.closing_costs_buy_percent if params.closing_costs_buy_percent is not None else 0.01
        closing_sell_pct = params.closing_costs_sell_percent if params.closing_costs_sell_percent is not None else 0.01
        seller_credit_pct = params.seller_credit_percent if params.seller_credit_percent is not None else 0.03
        staging_cost = params.staging_marketing if params.staging_marketing is not None else 2000
        listing_commission = params.listing_commission_rate if params.listing_commission_rate is not None else 0.025
        buyer_commission = params.buyer_commission_rate if params.buyer_commission_rate is not None else 0.025
    else:
        # Use all defaults
        repair_cost_per_sqft = 45
        hold_time_months = 5
        interest_rate = 0.10
        loan_points = 0.01
        loan_to_cost = 0.90
        monthly_hoa = 150
        monthly_insurance = 100
        monthly_utilities = 150
        property_tax_rate = 0.012
        closing_buy_pct = 0.01
        closing_sell_pct = 0.01
        seller_credit_pct = 0.03
        staging_cost = 2000
        listing_commission = 0.025
        buyer_commission = 0.025

    print(f"DEBUG: Using parameters - repair_cost: {repair_cost_per_sqft}, hold_time: {hold_time_months}")

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
            bedrooms = safe_int(row[col_bedrooms], None) if col_bedrooms is not None and col_bedrooms < len(row) else None
            bathrooms = None
            if col_bathrooms is not None and col_bathrooms < len(row):
                try:
                    bath_val = str(row[col_bathrooms]).strip()
                    if bath_val:
                        bathrooms = float(bath_val.replace(',', ''))
                except:
                    bathrooms = None

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

            # Get sqft from sheet first
            sqft = 0
            sqft_source = "Sheet"
            if col_sqft is not None and col_sqft < len(row):
                sqft = safe_int(row[col_sqft], 0)

            # Get address and Zillow URL for API calls
            address = str(row[col_address]).strip() if col_address is not None and col_address < len(row) else None
            zillow_url_from_sheet = str(row[col_zillow_url]).strip() if col_zillow_url is not None and col_zillow_url < len(row) and row[col_zillow_url] else None

            # Fetch Zestimate from Zillow (use URL with ZPID if available, otherwise construct)
            zestimate = None
            if address or zillow_url_from_sheet:
                try:
                    if zillow_url_from_sheet:
                        print(f"  Fetching Zestimate using Zillow URL: {zillow_url_from_sheet[:50]}...")
                    else:
                        print(f"  Fetching Zestimate from Zillow for {address}, {city}")

                    zestimate = zillow.get_value_estimate(
                        address or "Unknown", city, "GA", zipcode,
                        square_footage=sqft if sqft > 0 else None,
                        bedrooms=bedrooms,
                        bathrooms=bathrooms,
                        zillow_url=zillow_url_from_sheet
                    )
                    if zestimate:
                        print(f"  Retrieved Zestimate: ${zestimate:,.0f}")
                    else:
                        print(f"  Zestimate not available, defaulting to list price: ${list_price:,.0f}")
                        zestimate = list_price  # Default to list price if Zestimate unavailable

                except Exception as e:
                    print(f"  Error fetching Zestimate from Zillow: {e}, defaulting to list price")
                    zestimate = list_price  # Default to list price on error

            # Calculate flip deal if we have sqft
            flip_results = []
            if sqft > 0:
                try:
                    # Get address if available
                    address = str(row[col_address]).strip() if col_address is not None and col_address < len(row) else "Property"

                    # Create flip calculator input with custom parameters
                    flip_input = FlipCalculatorInput(
                        property_address=address,
                        city=city or "",
                        zip_code=zipcode or "00000",
                        sqft_living=sqft,
                        purchase_price=list_price,
                        arv=list_price,
                        repair_cost_per_sqft=repair_cost_per_sqft,
                        hold_time_months=hold_time_months,
                        interest_rate_annual=interest_rate,
                        loan_points=loan_points,
                        loan_to_cost_ratio=loan_to_cost,
                        monthly_hoa_maintenance=monthly_hoa,
                        monthly_insurance=monthly_insurance,
                        monthly_utilities=monthly_utilities,
                        property_tax_rate_annual=property_tax_rate,
                        closing_costs_buy_percent=closing_buy_pct,
                        closing_costs_sell_percent=closing_sell_pct,
                        seller_credit_percent=seller_credit_pct,
                        staging_marketing=staging_cost,
                        listing_commission_rate=listing_commission,
                        buyer_commission_rate=buyer_commission
                    )

                    # Calculate flip deal
                    flip_result = calculate_flip_deal(flip_input)
                    print(f"DEBUG: Calculated flip for row {idx}, ROI: {flip_result.profit_analysis.roi_percent:.1f}%")

                    # Calculate flip results (for internal use, not written to sheet)
                    flip_results = [
                        # Sqft info (2 columns)
                        str(sqft),
                        sqft_source,
                        # Quick indicators (3 columns)
                        "YES" if flip_result.profit_analysis.is_profitable else "NO",
                        "YES" if flip_result.profit_analysis.meets_minimum_roi else "NO",
                        "YES" if flip_result.profit_analysis.meets_70_percent_rule else "NO",
                        # Key metrics (3 columns)
                        f"{flip_result.profit_analysis.roi_percent:.1f}%",
                        f"${flip_result.profit_analysis.gross_profit:,.0f}",
                        f"{flip_result.profit_analysis.profit_margin_percent:.1f}%",
                        # Core numbers (4 columns)
                        f"${flip_result.acquisition.purchase_price:,.0f}",
                        f"${flip_result.profit_analysis.total_all_costs:,.0f}",
                        f"${flip_result.profit_analysis.cash_needed:,.0f}",
                        f"${flip_result.max_offer_70_rule:,.0f}",
                        # Cost summaries (4 columns)
                        f"${flip_result.acquisition.total_acquisition:,.0f}",
                        f"${flip_result.renovation.total_renovation:,.0f}",
                        f"${flip_result.holding.total_holding:,.0f}",
                        f"${flip_result.selling.total_selling:,.0f}",
                    ]

                    # Calculate ARV as 80% of Zestimate
                    arv_needed = flip_result.recommended_arv_for_profit

                    try:
                        if zestimate:
                            # Calculate ARV as 80% of Zestimate
                            arv_80_percent = zestimate * 0.80

                            # Determine deal status by comparing ARV (80%) vs ARV Needed
                            if arv_80_percent >= arv_needed * 1.05:  # 5% cushion
                                deal_status = "GOOD DEAL"
                            elif arv_80_percent >= arv_needed:
                                deal_status = "MAYBE"
                            else:
                                deal_status = "NO DEAL"

                            value_supports = "YES" if arv_80_percent >= arv_needed else "NO"

                            # Calculate Maximum Allowable Offer (MAO) as 50% of ARV (80%)
                            # But never higher than the list price
                            mao = arv_80_percent * 0.50  # = Zestimate * 0.40
                            mao = min(mao, list_price)  # Cap at list price

                            # 8 columns: Zestimate, ARV (80%), ARV Needed, Deal Status, Market Supports, Rehab, Total Cost, MAO
                            value_cols = [
                                f"${zestimate:,.0f}",  # Zestimate
                                f"${arv_80_percent:,.0f}",  # ARV (80%)
                                f"${arv_needed:,.0f}",  # ARV Needed
                                deal_status,  # Deal Status
                                value_supports,  # Market Supports Deal
                                f"${flip_result.renovation.total_renovation:,.0f}",  # Rehab Cost
                                f"${flip_result.profit_analysis.total_all_costs:,.0f}",  # Total Cost
                                f"${mao:,.0f}"  # Maximum Allowable Offer
                            ]
                        else:
                            print(f"  Zestimate not available")
                            mao = arv_needed * 0.50
                            mao = min(mao, list_price)  # Cap at list price
                            value_cols = ["Not Available", "N/A", f"${arv_needed:,.0f}", "UNKNOWN", "N/A", f"${flip_result.renovation.total_renovation:,.0f}", f"${flip_result.profit_analysis.total_all_costs:,.0f}", f"${mao:,.0f}"]
                    except Exception as e:
                        print(f"  Error calculating deal analysis: {e}")
                        mao = arv_needed * 0.50 if arv_needed else 0
                        mao = min(mao, list_price) if mao and list_price else mao  # Cap at list price
                        value_cols = ["ERROR", "ERROR", f"${arv_needed:,.0f}" if arv_needed else "N/A", "ERROR", "N/A", "N/A", "N/A", f"${mao:,.0f}"]

                except Exception as e:
                    print(f"Error calculating flip for row {idx}: {e}")
                    value_cols = ["ERROR", "", "", "", "", "", "", ""]
                    flip_results = ["", "", "ERROR"] + [""] * 13
            else:
                # Still no sqft after API call - cannot calculate flip
                print(f"  No sqft available for row {idx} - skipping flip calculation")
                zestimate_val = f"${zestimate:,.0f}" if zestimate else "N/A"
                arv_80_val = f"${zestimate * 0.80:,.0f}" if zestimate else "N/A"
                value_cols = [zestimate_val, arv_80_val, "N/A", "NO SQFT", "N/A", "N/A", "N/A", "N/A"]
                flip_results = ["0", "Missing", "N/A - No Sqft"] + [""] * 13

            # Format output (8 columns: Zestimate, ARV 80%, ARV Needed, Deal Status, Market Supports, Rehab, Total Cost, MAO)
            results_to_write.append(value_cols)

        except Exception as e:
            failed += 1
            # Add empty columns for failed rows (8 columns)
            results_to_write.append(["ERROR", str(e)[:30], "", "", "", "", "", ""])
            print(f"Error processing row {idx}: {e}")

    # Write results back to sheet if requested
    written_back = False
    print(f"DEBUG: write_back={request.write_back}, results_to_write length={len(results_to_write)}")
    print(f"DEBUG: First result row (if any): {results_to_write[0] if results_to_write else 'None'}")

    if request.write_back and results_to_write:
        try:
            print(f"DEBUG: Starting write operation to sheet...")
            # Write to columns X through AE (8 columns)
            # First, add/update header row
            header_row_num = request.start_row - 1
            if header_row_num >= 1:
                headers = [
                    # Zestimate & ARV Analysis (X-AE)
                    'Zestimate', 'ARV_80_Percent', 'ARV_Needed', 'Deal_Status', 'Market_Supports_Deal',
                    'Rehab_Cost', 'Total_Cost', 'Maximum_Allowable_Offer'
                ]
                print(f"DEBUG: Writing headers to row {header_row_num}")
                worksheet.update(f'X{header_row_num}:AE{header_row_num}', [headers])

            # Write prediction results
            start_cell = f'X{request.start_row}'
            end_row = request.start_row + len(results_to_write) - 1
            end_cell = f'AE{end_row}'

            print(f"DEBUG: Writing {len(results_to_write)} rows from {start_cell} to {end_cell}")
            worksheet.update(f'{start_cell}:{end_cell}', results_to_write)
            written_back = True
            print(f"DEBUG: Write operation completed successfully!")

        except Exception as e:
            print(f"Failed to write results back to sheet: {e}")
            import traceback
            traceback.print_exc()
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
