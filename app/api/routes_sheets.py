"""
Google Sheets integration API routes - Updated for MLS Data
Uses area-specific ARV multiples from flip analysis
"""

from fastapi import APIRouter, HTTPException, status
from typing import Optional, List
import requests
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
from app.services.ml_arv_service import MLARVPredictor, compare_with_zillow

router = APIRouter(prefix="/sheets", tags=["google-sheets"])

# Global variable for loaded models (set from main.py)
stacker_model = None
ml_arv_model = None

# Load area-specific ARV multiples
arv_multiples_zip = None
arv_multiples_city = None


def load_ml_arv_model():
    """Load the trained ML ARV model"""
    global ml_arv_model

    try:
        model_path = 'models/ml_arv_hybrid.pkl'
        if os.path.exists(model_path):
            ml_arv_model = MLARVPredictor()
            ml_arv_model.load_model(model_path)
            print(f"Loaded ML ARV hybrid model from {model_path}")
        else:
            print(f"ML ARV model not found at {model_path}. Will use Zillow-only predictions.")
    except Exception as e:
        print(f"Error loading ML ARV model: {e}")
        ml_arv_model = None


def get_hybrid_arv(property_data: dict, zestimate: Optional[float]) -> dict:
    """
    Get hybrid ARV using ML model (primary) and Zillow (validation).

    Returns dict with:
        - arv_primary: Primary ARV to use
        - arv_ml: ML predicted ARV (or None)
        - arv_zillow: Zillow-based ARV (zestimate * 0.80)
        - confidence: Confidence level
        - agreement: Agreement status
        - flag_review: Whether to flag for manual review
    """
    global ml_arv_model

    # Get Zillow-based ARV (fallback)
    arv_zillow = zestimate * 0.80 if zestimate else None

    # Try to get ML ARV if model is loaded
    arv_ml = None
    ml_result = None

    if ml_arv_model and ml_arv_model.is_trained:
        try:
            ml_result = ml_arv_model.predict_with_confidence(property_data)
            arv_ml = ml_result['arv_prediction']
        except Exception as e:
            print(f"  Warning: ML ARV prediction failed: {e}")
            arv_ml = None

    # Determine primary ARV using hybrid logic
    if arv_ml:
        # Have ML prediction - compare with Zillow
        comparison = compare_with_zillow(arv_ml, zestimate)
        return {
            'arv_primary': comparison['primary_arv'],
            'arv_ml': arv_ml,
            'arv_ml_lower': ml_result['arv_lower'] if ml_result else None,
            'arv_ml_upper': ml_result['arv_upper'] if ml_result else None,
            'arv_zillow': int(arv_zillow) if arv_zillow else None,
            'confidence': comparison['confidence'],
            'agreement': comparison['agreement'],
            'flag_review': comparison['flag_review'],
            'ml_confidence': ml_result['confidence'] if ml_result else None
        }
    elif arv_zillow:
        # Only have Zillow - use it
        return {
            'arv_primary': int(arv_zillow),
            'arv_ml': None,
            'arv_ml_lower': None,
            'arv_ml_upper': None,
            'arv_zillow': int(arv_zillow),
            'confidence': 'ZILLOW_ONLY',
            'agreement': 'NO_ML',
            'flag_review': False,
            'ml_confidence': None
        }
    else:
        # No prediction available
        return {
            'arv_primary': None,
            'arv_ml': None,
            'arv_ml_lower': None,
            'arv_ml_upper': None,
            'arv_zillow': None,
            'confidence': 'NO_DATA',
            'agreement': 'NO_DATA',
            'flag_review': True,
            'ml_confidence': None
        }


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
    # Load ARV multiples and ML model when model is set
    load_arv_multiples()
    load_ml_arv_model()


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
        try:
            properties = spreadsheet.fetch_sheet_metadata()
        except gspread.exceptions.APIError as e:
            # Check if this is an Excel file that needs conversion
            if 'This operation is not supported for this document' in str(e):
                return {
                    'can_edit': False,
                    'message': 'This appears to be an Excel file. Please convert it to Google Sheets format: Open the file in Google Drive, then File > Save as Google Sheets.'
                }
            # Re-raise if it's a different error
            raise

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
        # Check if this is the Excel file error
        if 'This operation is not supported for this document' in str(e):
            return {
                'can_edit': False,
                'message': 'This appears to be an Excel file. Please convert it to Google Sheets format: Open the file in Google Drive, then File > Save as Google Sheets.'
            }
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
    Process property listings from Google Sheets with ARV Analysis.

    ## Deal Types

    Supports two deal types controlled by the `deal_type` parameter:

    ### On-Market (MLS Data) - `deal_type: "on_market"`
    Standard MLS format with City, Zip, List Price columns.
    - If no Zestimate → defaults to List Price
    - If square footage provided → calculates Rehab Cost
    - Calculates Maximum Allowable Offer (MAO)
    Outputs to columns X-Z: Zestimate, Rehab_Cost, Offer_Amount

    ### Off-Market - `deal_type: "off_market"`
    Simplified format with full address in column A and square footage in column G.
    - If no Zestimate → writes "No Zestimate" and skips property
    - If Zestimate exists:
      - If square footage provided → uses it for Rehab Cost
      - If no square footage → fetches from Zillow API for Rehab Cost
    - Calculates Maximum Allowable Offer (MAO)
    Outputs to columns N-P (skipping M): Zestimate, Rehab_Cost, MAO

    ## Off-Market Sheet Format

    Expected columns:
    - **A: Address** - Full address (e.g., "123 Main St, Atlanta, GA, USA")
    - **B: State** - State abbreviation
    - **C: MSA** - Metro area
    - **D: Year Built**
    - **E: Bed** - Bedrooms
    - **F: Bath** - Bathrooms
    - **G: SF** - Square footage
    - **H: Occ or Vacant**
    - **I-K: Lease info**
    - **L: Price** - Asking price

    ## On-Market Sheet Format (MLS Data)

    Required columns (names are flexible):
    - **City** - Property city
    - **Zip** - 5-digit zip code
    - **List Price** (or Price, MLS Amount) - Listing price

    ## Output Columns

    **On-Market**: Writes 3 columns (X-Z): Zestimate, Rehab_Cost, Offer_Amount
    **Off-Market**: Writes 3 columns (N-P): Zestimate, Rehab_Cost, MAO

    ## MAO Calculation

    MAO (Maximum Allowable Offer) uses the 70% rule:
    - Formula: MAO = (0.70 × Zestimate) - Rehab_Cost
    - This ensures sufficient margin for repairs, holding costs, and profit

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

        # Check deal type and route to appropriate processing
        deal_type = request.deal_type.lower() if request.deal_type else "on_market"
        print(f"DEBUG: Processing deal_type = {deal_type}")

        # Get data rows (skip header)
        data_rows = all_values[request.start_row - 1:]

        # ============ OFF-MARKET PROCESSING ============
        if deal_type == "off_market":
            print(f"DEBUG: Using OFF-MARKET processing for {len(data_rows)} rows")
            # Off-market format: Address in A, SF in G, Price in L
            # Output to N, O, P (skip M): Zestimate, Rehab_Cost, MAO

            # Initialize Zillow API service
            zillow = ZillowAPIService()

            # Extract repair cost parameter
            params = request.parameters
            repair_cost_per_sqft = params.repair_cost_per_sqft if params and params.repair_cost_per_sqft is not None else 45

            successful = 0
            failed = 0
            results_to_write = []

            for idx, row in enumerate(data_rows, start=request.start_row):
                try:
                    # Off-market columns (0-indexed):
                    # A=0: Address, B=1: State, C=2: MSA, D=3: Year Built,
                    # E=4: Bed, F=5: Bath, G=6: SF, H=7: Occ/Vacant,
                    # I=8: Lease Start, J=9: Lease End, K=10: Leased Rent, L=11: Price
                    full_address = str(row[0]).strip() if len(row) > 0 and row[0] else None
                    state = str(row[1]).strip() if len(row) > 1 and row[1] else "GA"
                    bedrooms = safe_int(row[4], None) if len(row) > 4 else None
                    bathrooms = None
                    if len(row) > 5 and row[5]:
                        try:
                            bathrooms = float(str(row[5]).strip().replace(',', ''))
                        except:
                            bathrooms = None
                    sqft = safe_int(row[6], 0) if len(row) > 6 else 0
                    list_price = clean_price(row[11]) if len(row) > 11 else None

                    # Skip if no address
                    if not full_address:
                        failed += 1
                        results_to_write.append(["ERROR - No Address", "", ""])
                        continue

                    # Parse city and zip from full address (format: "123 St, City, ST, USA")
                    address_parts = full_address.split(',')
                    city = address_parts[1].strip() if len(address_parts) > 1 else None
                    zipcode = None
                    # Try to extract zip from address
                    zip_match = re.search(r'(\d{5})', full_address)
                    if zip_match:
                        zipcode = zip_match.group(1)

                    print(f"  Row {idx}: Processing {full_address[:50]}...")

                    # Fetch Zestimate from Zillow using full address
                    zestimate = None
                    try:
                        api_result = zillow.get_value_estimate(
                            full_address, city, state, zipcode,
                            square_footage=sqft if sqft > 0 else None,
                            bedrooms=bedrooms,
                            bathrooms=bathrooms,
                            return_details=True
                        )

                        if api_result and isinstance(api_result, dict):
                            zestimate = api_result.get('zestimate')
                            # For OFF-MARKET: If no sqft provided, get it from API
                            if sqft == 0 and api_result.get('sqft'):
                                sqft = api_result['sqft']
                                print(f"    Got sqft from API: {sqft:,}")

                        if zestimate and zestimate > 0:
                            print(f"    SUCCESS: Zestimate = ${zestimate:,.0f}")
                        else:
                            # OFF-MARKET: No Zestimate means skip this property
                            print(f"    No Zestimate found - SKIPPING property")
                            results_to_write.append(["No Zestimate", "", ""])
                            continue

                    except Exception as e:
                        # OFF-MARKET: API error means no Zestimate, skip property
                        print(f"    API error: {e} - SKIPPING property")
                        results_to_write.append(["No Zestimate", "", ""])
                        continue

                    # OFF-MARKET: We have a Zestimate, now calculate rehab cost and MAO
                    # If sqft still not available after API call, we cannot proceed
                    if sqft == 0:
                        print(f"    No square footage available - SKIPPING property")
                        results_to_write.append([f"${zestimate:,.0f}", "No Square Footage", ""])
                        continue

                    # Calculate rehab cost and MAO
                    rehab_cost = sqft * repair_cost_per_sqft
                    mao = (0.7 * zestimate) - rehab_cost
                    mao = max(0, mao)

                    value_cols = [
                        f"${zestimate:,.0f}",
                        f"${rehab_cost:,.0f}",
                        f"${mao:,.0f}"
                    ]
                    print(f"    Zestimate=${zestimate:,.0f}, Rehab=${rehab_cost:,.0f}, MAO=${mao:,.0f}")

                    results_to_write.append(value_cols)
                    successful += 1

                except Exception as e:
                    failed += 1
                    results_to_write.append(["ERROR", str(e)[:30], ""])
                    print(f"Error processing row {idx}: {e}")

            # Write results to columns N-P (skipping M)
            written_back = False
            if request.write_back and results_to_write:
                try:
                    # Write headers
                    header_row_num = request.start_row - 1
                    if header_row_num >= 1:
                        headers = ['Zestimate', 'Rehab_Cost', 'MAO']
                        worksheet.update(f'N{header_row_num}:P{header_row_num}', [headers])

                    # Write data to N-P
                    start_cell = f'N{request.start_row}'
                    end_row = request.start_row + len(results_to_write) - 1
                    end_cell = f'P{end_row}'

                    print(f"DEBUG: Writing {len(results_to_write)} rows from {start_cell} to {end_cell}")
                    worksheet.update(f'{start_cell}:{end_cell}', results_to_write)
                    written_back = True
                    print(f"DEBUG: Write operation completed successfully!")

                except Exception as e:
                    print(f"Failed to write results back to sheet: {e}")
                    import traceback
                    traceback.print_exc()

            return GoogleSheetsResponse(
                sheet_id=sheet_id,
                total_properties=len(data_rows),
                successful_predictions=successful,
                failed_predictions=failed,
                predictions=[],
                written_back=written_back,
                timestamp=datetime.now()
            )

        # ============ ON-MARKET PROCESSING (existing logic) ============
        print(f"DEBUG: Using ON-MARKET processing for {len(data_rows)} rows")

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

        # Address component columns (for combining into full address)
        col_street_number = find_column(header_row, ['street number', 'street #', 'number', 'house number'])
        col_street_name = find_column(header_row, ['street name', 'street'])
        col_street_suffix = find_column(header_row, ['street suffix', 'street type', 'suffix', 'type'])

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
            address = str(row[col_address]).strip() if col_address is not None and col_address < len(row) and row[col_address] else None

            # If no complete address column, try to build from components (columns D, E, H)
            if not address and col_street_number is not None and col_street_name is not None:
                street_number = str(row[col_street_number]).strip() if col_street_number < len(row) and row[col_street_number] else ""
                street_name = str(row[col_street_name]).strip() if col_street_name < len(row) and row[col_street_name] else ""
                street_suffix = str(row[col_street_suffix]).strip() if col_street_suffix is not None and col_street_suffix < len(row) and row[col_street_suffix] else ""

                # Combine components to create full address
                address_parts = [street_number, street_name, street_suffix]
                address = " ".join([part for part in address_parts if part])

                if address:
                    print(f"  Combined address from components: {address}")

            zillow_url_from_sheet = str(row[col_zillow_url]).strip() if col_zillow_url is not None and col_zillow_url < len(row) and row[col_zillow_url] else None

            # ON-MARKET: Fetch Zestimate and property details from Zillow
            # If no Zestimate available, default to list price
            zestimate = None
            zestimate_source = "API"  # Track where Zestimate came from

            if address or zillow_url_from_sheet:
                try:
                    if zillow_url_from_sheet:
                        print(f"  Fetching Zestimate using Zillow URL: {zillow_url_from_sheet[:50]}...")
                    else:
                        print(f"  Fetching Zestimate from Zillow for {address}, {city}")

                    # Get full property details including sqft
                    api_result = zillow.get_value_estimate(
                        address or "Unknown", city, "GA", zipcode,
                        square_footage=sqft if sqft > 0 else None,
                        bedrooms=bedrooms,
                        bathrooms=bathrooms,
                        zillow_url=zillow_url_from_sheet,
                        return_details=True  # Get sqft along with zestimate
                    )

                    if api_result and isinstance(api_result, dict):
                        zestimate = api_result.get('zestimate')

                        # ON-MARKET: Use API sqft if sheet doesn't have it
                        if sqft == 0 and api_result.get('sqft'):
                            sqft = api_result['sqft']
                            sqft_source = "Zillow API"
                            print(f"  SUCCESS: Got sqft from API: {sqft:,}")

                    if zestimate and zestimate > 0:
                        print(f"  SUCCESS: Retrieved Zestimate from API: ${zestimate:,.0f}")
                        zestimate_source = "Zillow API"
                    else:
                        # ON-MARKET: No Zestimate means use list price as default
                        print(f"  API returned no Zestimate, defaulting to list price: ${list_price:,.0f}")
                        zestimate = list_price
                        zestimate_source = "List Price (fallback)"

                except requests.exceptions.Timeout:
                    # ON-MARKET: Timeout means use list price
                    print(f"  API timeout, defaulting to list price: ${list_price:,.0f}")
                    zestimate = list_price
                    zestimate_source = "List Price (timeout)"
                except Exception as e:
                    # ON-MARKET: Error means use list price
                    print(f"  API error ({type(e).__name__}), defaulting to list price: ${list_price:,.0f}")
                    zestimate = list_price
                    zestimate_source = "List Price (error)"
            else:
                # ON-MARKET: No address means use list price
                print(f"  No address or URL available, using list price: ${list_price:,.0f}")
                zestimate = list_price
                zestimate_source = "List Price (no address)"

            # Check if we have a real Zestimate (not just list price fallback)
            has_real_zestimate = zestimate_source == "Zillow API"
            zestimate_display = f"${zestimate:,.0f}" if has_real_zestimate else f"List Price ${zestimate:,.0f}"

            # ON-MARKET: Calculate 3-column output: Zestimate, Rehab Cost, Offer Amount
            # Only calculate rehab cost if square footage is provided in sheet
            if sqft > 0:
                # Calculate rehab cost
                rehab_cost = sqft * repair_cost_per_sqft

                # Calculate offer amount (Maximum Allowable Offer) using 70% rule
                if zestimate and zestimate > 0:
                    offer_amount = (0.7 * zestimate) - rehab_cost
                    offer_amount = max(0, offer_amount)  # Don't go negative

                    value_cols = [
                        f"${zestimate:,.0f}",  # Zestimate (or List Price if no Zestimate)
                        f"${rehab_cost:,.0f}",  # Rehab Cost
                        f"${offer_amount:,.0f}"  # Maximum Allowable Offer (70% rule)
                    ]
                    print(f"  Row {idx}: Zestimate=${zestimate:,.0f}, Rehab=${rehab_cost:,.0f}, Offer=${offer_amount:,.0f}")
                else:
                    value_cols = ["N/A", f"${rehab_cost:,.0f}", "N/A"]
            else:
                # ON-MARKET: No sqft provided - cannot calculate rehab cost
                value_cols = [f"${zestimate:,.0f}", "NO SQFT", "N/A"]
                print(f"  No sqft available for row {idx}")

            # Format output (3 columns: Zestimate, Rehab_Cost, Offer_Amount)
            results_to_write.append(value_cols)

        except Exception as e:
            failed += 1
            # Add empty columns for failed rows (3 columns)
            results_to_write.append(["ERROR", str(e)[:30], ""])
            print(f"Error processing row {idx}: {e}")

    # Write results back to sheet if requested
    written_back = False
    print(f"DEBUG: write_back={request.write_back}, results_to_write length={len(results_to_write)}")
    print(f"DEBUG: First result row (if any): {results_to_write[0] if results_to_write else 'None'}")

    if request.write_back and results_to_write:
        try:
            print(f"DEBUG: Starting write operation to sheet...")
            # Write to columns X through Z (3 columns)
            # First, add/update header row
            header_row_num = request.start_row - 1
            if header_row_num >= 1:
                headers = [
                    'Zestimate', 'Rehab_Cost', 'Offer_Amount'
                ]
                print(f"DEBUG: Writing headers to row {header_row_num}")
                worksheet.update(f'X{header_row_num}:Z{header_row_num}', [headers])

            # Write prediction results
            start_cell = f'X{request.start_row}'
            end_row = request.start_row + len(results_to_write) - 1
            end_cell = f'Z{end_row}'

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
