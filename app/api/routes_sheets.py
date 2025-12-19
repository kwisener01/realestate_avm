"""
Google Sheets integration API routes
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
    PredictionRequest,
    PredictionResponse,
    PropertyFeatures
)

router = APIRouter(prefix="/sheets", tags=["google-sheets"])

# Global variable for loaded models (set from main.py)
stacker_model = None


def set_model(model):
    """Set the global model instance"""
    global stacker_model
    stacker_model = model


def estimate_arv_multiplier(distance: float, days_on_market: int, city: str) -> float:
    """
    Estimate ARV multiplier based on location and market factors

    Logic:
    - Closer to Atlanta (lower distance) = higher ARV potential
    - Longer days on market = more distressed = higher ARV upside
    - Certain areas have better appreciation potential
    """
    # Base multiplier for fix-and-flip
    base_multiplier = 1.75

    # Distance factor: Closer to Atlanta = higher multiplier
    if distance <= 10:
        distance_bonus = 0.25
    elif distance <= 20:
        distance_bonus = 0.15
    elif distance <= 30:
        distance_bonus = 0.10
    elif distance <= 40:
        distance_bonus = 0.05
    else:
        distance_bonus = 0.0

    # Days on market factor: More days = more distressed = higher potential
    if days_on_market >= 180:
        dom_bonus = 0.15
    elif days_on_market >= 90:
        dom_bonus = 0.10
    elif days_on_market >= 30:
        dom_bonus = 0.05
    else:
        dom_bonus = 0.0

    # High-value areas
    high_value_cities = ['Atlanta', 'Decatur', 'Brookhaven', 'Sandy Springs',
                        'Alpharetta', 'Roswell', 'Marietta']
    moderate_value_cities = ['Lithonia', 'Riverdale', 'College Park', 'East Point',
                            'Forest Park', 'Smyrna', 'Dunwoody']

    city_bonus = 0.0
    if city in high_value_cities:
        city_bonus = 0.10
    elif city in moderate_value_cities:
        city_bonus = 0.05

    # Calculate final multiplier
    final_multiplier = base_multiplier + distance_bonus + dom_bonus + city_bonus

    # Cap between 1.5 and 2.3
    final_multiplier = max(1.5, min(2.3, final_multiplier))

    return final_multiplier


def extract_sheet_id(sheet_url: str) -> str:
    """Extract Google Sheets ID from URL or return as-is if already an ID"""
    # Pattern for Google Sheets URL
    pattern = r'/spreadsheets/d/([a-zA-Z0-9-_]+)'
    match = re.search(pattern, sheet_url)

    if match:
        return match.group(1)

    # Assume it's already an ID if no pattern match
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

    # Define required scopes
    scopes = [
        'https://www.googleapis.com/auth/spreadsheets',
        'https://www.googleapis.com/auth/drive'
    ]

    # Use provided credentials or environment variable
    creds_data = credentials_path or os.getenv('GOOGLE_SHEETS_CREDENTIALS')

    if not creds_data:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Google Sheets credentials not provided. Set GOOGLE_SHEETS_CREDENTIALS environment variable or provide credentials_path in request."
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


def parse_sheet_row(row: List[str]) -> tuple[Optional[PropertyFeatures], Optional[str], Optional[dict]]:
    """
    Parse a row from Google Sheets into PropertyFeatures plus hybrid ARV data.

    Expected columns (in order):
    1-13: bedrooms, bathrooms, sqft_living, sqft_lot, floors, year_built, year_renovated,
          latitude, longitude, property_type, neighborhood, condition, view_quality
    14: description (optional)
    15: list_price (for location ARV)
    16: distance (from Atlanta, in miles)
    17: days_on_market
    18: city (for location ARV)

    Args:
        row: List of cell values from a sheet row

    Returns:
        Tuple of (PropertyFeatures object or None, error message or None, hybrid_data dict or None)
    """
    try:
        # Ensure we have enough columns (at least 13 for required features)
        if len(row) < 13:
            return None, f"Need 13+ cols (has {len(row)})", None

        # Helper to safely convert values with validation-compliant defaults
        def safe_int(val, default, min_val=None, max_val=None):
            try:
                result = int(float(val)) if val and str(val).strip() else default
                if min_val is not None and result < min_val:
                    result = default
                if max_val is not None and result > max_val:
                    result = default
                return result
            except:
                return default

        def safe_float(val, default, min_val=None, max_val=None):
            try:
                result = float(val) if val and str(val).strip() else default
                if min_val is not None and result < min_val:
                    result = default
                if max_val is not None and result > max_val:
                    result = default
                return result
            except:
                return default

        def safe_str(val, default, allowed=None):
            try:
                result = str(val).strip() if val else default
                if allowed and result not in allowed:
                    return default
                return result
            except:
                return default

        def extract_distance(val):
            """Extract distance from string like '50.8 mi' or just '50.8'"""
            try:
                if not val or str(val).strip() == '':
                    return 30.0  # Default to moderate distance
                val_str = str(val).strip()
                match = re.search(r'([\d.]+)', val_str)
                if match:
                    return float(match.group(1))
                return 30.0
            except:
                return 30.0

        def clean_price(val):
            """Convert price string to numeric"""
            try:
                if not val or str(val).strip() == '':
                    return None
                cleaned = re.sub(r'[,$]', '', str(val))
                return float(cleaned)
            except:
                return None

        # Parse with validation-compliant defaults
        features = PropertyFeatures(
            bedrooms=safe_int(row[0], 3, min_val=1),
            bathrooms=safe_float(row[1], 2.0, min_val=1.0),
            sqft_living=safe_int(row[2], 2000, min_val=100),
            sqft_lot=safe_int(row[3], 5000, min_val=0),
            floors=safe_float(row[4], 1.0, min_val=1.0, max_val=5.0),
            year_built=safe_int(row[5], 2000, min_val=1800, max_val=2025),
            year_renovated=safe_int(row[6], 0, min_val=0, max_val=2025),
            latitude=safe_float(row[7], 33.75, min_val=-90.0, max_val=90.0),
            longitude=safe_float(row[8], -84.28, min_val=-180.0, max_val=180.0),
            property_type=safe_str(row[9], "Single Family",
                                  allowed=['Single Family', 'Townhouse', 'Condo', 'Multi-Family']),
            neighborhood=safe_str(row[10], "Unknown"),
            condition=safe_str(row[11], "Average",
                             allowed=['Poor', 'Fair', 'Average', 'Good', 'Excellent']),
            view_quality=safe_str(row[12], "None",
                                allowed=['None', 'Fair', 'Good', 'Excellent'])
        )

        # Parse hybrid ARV data (columns 15-18) if available
        hybrid_data = None
        if len(row) >= 18:
            list_price = clean_price(row[14])  # Column 15
            distance = extract_distance(row[15])  # Column 16
            days_on_market = safe_int(row[16], 0, min_val=0)  # Column 17
            city = safe_str(row[17], "Atlanta")  # Column 18

            hybrid_data = {
                'list_price': list_price,
                'distance': distance,
                'days_on_market': days_on_market,
                'city': city
            }

        return features, None, hybrid_data
    except Exception as e:
        # Extract just the key part of validation error
        error_msg = str(e)
        if 'validation error' in error_msg.lower():
            # Simplify pydantic error messages
            lines = error_msg.split('\n')
            return None, f"Invalid: {lines[0][:40]}", None
        return None, f"Error: {error_msg[:40]}", None


@router.post("/predict", response_model=GoogleSheetsResponse, status_code=status.HTTP_200_OK)
async def predict_from_sheets(request: GoogleSheetsRequest):
    """
    Process properties from Google Sheets with Location-Based ARV Model.

    ## Expected Sheet Format

    The sheet should have a header row with these column names (order doesn't matter):
    - **List Price** - Property listing price
    - **Distance** - Distance from Atlanta (e.g., "50.8 mi" or just "50.8")
    - **Days On Market** - How long property has been listed
    - **City** - City name (for location bonus)

    Optional columns:
    - County, Address, MLS #, etc. (will be ignored)

    ## Output Columns

    Writes 3 columns to the sheet:
    - **X: Deal Status** - "Deal" if List Price ≤ 50% of ARV, else "No Deal"
    - **Y: ARV (Location)** - Location-based ARV estimate
    - **Z: Confidence** - Confidence level based on factors

    ## ARV Calculation

    Uses distance from Atlanta, days on market, and city to calculate ARV:
    - Base multiplier: 1.75x
    - Distance bonus: 0-10 mi (+0.25), 10-20 mi (+0.15), etc.
    - DOM bonus: 180+ days (+0.15), 90-180 days (+0.10), etc.
    - City premium: High-value cities (+0.10), moderate (+0.05)
    - Final multiplier: 1.5x - 2.3x

    ## Authentication

    Requires Google service account credentials with access to the sheet.
    Provide credentials via:
    - Request parameter: `credentials_path`
    - Environment variable: `GOOGLE_SHEETS_CREDENTIALS`
    """

    # Get Google Sheets client
    client = get_google_sheets_client(request.credentials_path)

    # Extract sheet ID and open spreadsheet
    sheet_id = extract_sheet_id(request.sheet_url)

    try:
        spreadsheet = client.open_by_key(sheet_id)
        worksheet = spreadsheet.worksheet(request.sheet_name)
    except gspread.SpreadsheetNotFound:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Spreadsheet not found: {sheet_id}. Ensure the service account has access."
        )
    except gspread.WorksheetNotFound:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Worksheet '{request.sheet_name}' not found in spreadsheet"
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

        # Find column indices by name (case-insensitive)
        def find_column(header_row, possible_names):
            """Find column index by trying multiple possible header names"""
            header_lower = [h.lower().strip() for h in header_row]
            for name in possible_names:
                name_lower = name.lower()
                if name_lower in header_lower:
                    return header_lower.index(name_lower)
            return None

        col_list_price = find_column(header_row, ['list price', 'price', 'listing price'])
        col_distance = find_column(header_row, ['distance'])
        col_dom = find_column(header_row, ['days on market', 'dom', 'days on mkt'])
        col_city = find_column(header_row, ['city'])

        # Verify we have required columns
        missing_cols = []
        if col_list_price is None:
            missing_cols.append('List Price')
        if col_distance is None:
            missing_cols.append('Distance')
        if col_dom is None:
            missing_cols.append('Days On Market')
        if col_city is None:
            missing_cols.append('City')

        if missing_cols:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Missing required columns: {', '.join(missing_cols)}. Found columns: {', '.join(header_row[:10])}"
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

    # Helper functions for parsing
    def extract_distance(val):
        """Extract distance from string like '50.8 mi' or just '50.8'"""
        try:
            if not val or str(val).strip() == '':
                return 30.0  # Default to moderate distance
            val_str = str(val).strip()
            match = re.search(r'([\d.]+)', val_str)
            if match:
                return float(match.group(1))
            return 30.0
        except:
            return 30.0

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
        """Safely convert to int"""
        try:
            return int(float(val)) if val and str(val).strip() else default
        except:
            return default

    # Process each row and calculate location-based ARV
    successful = 0
    failed = 0
    results_to_write = []

    for idx, row in enumerate(data_rows, start=request.start_row):
        try:
            # Extract data from the correct columns
            list_price = clean_price(row[col_list_price]) if col_list_price < len(row) else None
            distance = extract_distance(row[col_distance]) if col_distance < len(row) else 30.0
            days_on_market = safe_int(row[col_dom]) if col_dom < len(row) else 0
            city = str(row[col_city]).strip() if col_city < len(row) and row[col_city] else "Atlanta"

            # Skip if no list price
            if not list_price or list_price <= 0:
                failed += 1
                results_to_write.append(["ERROR - No Price", "", ""])
                continue

            # Calculate location-based ARV
            multiplier = estimate_arv_multiplier(distance, days_on_market, city)
            arv_location = list_price * multiplier

            # Determine deal status (list price <= 50% of ARV = deal)
            is_deal = list_price <= (arv_location * 0.5)
            deal_status = "Deal" if is_deal else "No Deal"

            # Calculate confidence based on factors
            confidence = "MEDIUM"
            if distance <= 10 and days_on_market >= 90:
                confidence = "HIGH"
            elif distance > 40 or days_on_market < 30:
                confidence = "LOW"

            successful += 1

            # Format output (3 columns: Deal Status, ARV, Confidence)
            results_to_write.append([
                deal_status,
                f"${arv_location:,.0f}",
                confidence
            ])

        except Exception as e:
            failed += 1
            results_to_write.append(["ERROR", str(e)[:50], ""])
            print(f"Error processing row {idx}: {e}")

    # Write results back to sheet if requested
    written_back = False
    if request.write_back and results_to_write:
        try:
            # Write to columns X, Y, Z (24, 25, 26)
            # First, add/update header row
            header_row = request.start_row - 1
            if header_row >= 1:
                worksheet.update(f'X{header_row}:Z{header_row}',
                               [['Deal Status', 'ARV (Location)', 'Confidence']])

            # Write prediction results
            start_cell = f'X{request.start_row}'
            end_row = request.start_row + len(results_to_write) - 1
            end_cell = f'Z{end_row}'

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
        predictions=[],  # Empty list since we're not using ML predictions
        written_back=written_back,
        timestamp=datetime.now()
    )


@router.get("/health")
async def sheets_health():
    """Check if Google Sheets integration is healthy"""
    creds_path = os.getenv('GOOGLE_SHEETS_CREDENTIALS')

    return {
        "status": "ready",
        "credentials_configured": creds_path is not None and os.path.exists(creds_path) if creds_path else False,
        "model_loaded": stacker_model is not None,
        "timestamp": datetime.now()
    }
