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


def parse_sheet_row(row: List[str]) -> tuple[Optional[PropertyFeatures], Optional[str]]:
    """
    Parse a row from Google Sheets into PropertyFeatures.

    Expected columns (in order):
    bedrooms, bathrooms, sqft_living, sqft_lot, floors, year_built, year_renovated,
    latitude, longitude, property_type, neighborhood, condition, view_quality, description (optional)

    Args:
        row: List of cell values from a sheet row

    Returns:
        Tuple of (PropertyFeatures object or None, error message or None)
    """
    try:
        # Ensure we have enough columns (at least 13 for required features)
        if len(row) < 13:
            return None, f"Need 13+ cols (has {len(row)})"

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
        return features, None
    except Exception as e:
        # Extract just the key part of validation error
        error_msg = str(e)
        if 'validation error' in error_msg.lower():
            # Simplify pydantic error messages
            lines = error_msg.split('\n')
            return None, f"Invalid: {lines[0][:40]}"
        return None, f"Error: {error_msg[:40]}"


@router.post("/predict", response_model=GoogleSheetsResponse, status_code=status.HTTP_200_OK)
async def predict_from_sheets(request: GoogleSheetsRequest):
    """
    Process properties from Google Sheets and return predictions.

    ## Expected Sheet Format

    The sheet should have columns in this order (with header row):
    1. bedrooms (int)
    2. bathrooms (float)
    3. sqft_living (int)
    4. sqft_lot (int)
    5. floors (float)
    6. year_built (int)
    7. year_renovated (int)
    8. latitude (float)
    9. longitude (float)
    10. property_type (string: Single Family, Townhouse, Condo, Multi-Family)
    11. neighborhood (string)
    12. condition (string: Poor, Fair, Average, Good, Excellent)
    13. view_quality (string: None, Fair, Good, Excellent)
    14. description (string, optional)

    ## Authentication

    Requires Google service account credentials with access to the sheet.
    Provide credentials via:
    - Request parameter: `credentials_path`
    - Environment variable: `GOOGLE_SHEETS_CREDENTIALS`

    ## Features
    - Reads property data from specified sheet
    - Generates predictions for all properties
    - Optionally writes predictions back to columns N-P (predicted_price, confidence, timestamp)
    """
    if stacker_model is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model not loaded. Please ensure models are trained and available."
        )

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

        # Get data rows (skip header)
        data_rows = all_values[request.start_row - 1:]

    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to read sheet data: {str(e)}"
        )

    # Process each row and make predictions
    predictions: List[PredictionResponse] = []
    successful = 0
    failed = 0
    results_to_write = []

    for idx, row in enumerate(data_rows, start=request.start_row):
        try:
            # Parse features
            features, error_msg = parse_sheet_row(row)

            if features is None:
                failed += 1
                results_to_write.append(["ERROR", error_msg or "Parse failed", ""])
                continue

            # Get description if available (column 14)
            description = row[13] if len(row) > 13 and row[13] else None

            # Create prediction request
            pred_request = PredictionRequest(
                property_id=f"row_{idx}",
                features=features,
                description=description,
                use_ensemble=request.use_ensemble
            )

            # Make prediction (import the prediction function)
            from app.api.routes_predict import predict_property_value

            prediction = await predict_property_value(pred_request)
            predictions.append(prediction)
            successful += 1

            # Prepare data for writing back
            results_to_write.append([
                f"{prediction.predicted_price:.2f}",
                f"{prediction.confidence_score:.2f}" if prediction.confidence_score else "",
                prediction.timestamp.strftime("%Y-%m-%d %H:%M:%S")
            ])

        except Exception as e:
            failed += 1
            results_to_write.append(["ERROR", str(e)[:100], ""])
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
                               [['predicted_price', 'confidence', 'timestamp']])

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
        predictions=predictions,
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
