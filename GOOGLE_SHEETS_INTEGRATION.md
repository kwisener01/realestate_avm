# Google Sheets Integration Guide

This guide explains how to use the Google Sheets integration to analyze property data directly from your spreadsheets.

## Overview

The `/sheets/predict` endpoint allows you to:
- 📊 Read property data from Google Sheets
- 🤖 Generate predictions for all properties
- ✍️ Automatically write predictions back to the sheet
- 📈 Process up to 100+ properties in a single request

---

## Setup

### Step 1: Create Google Service Account

1. Go to [Google Cloud Console](https://console.cloud.google.com/)
2. Create a new project or select existing
3. Enable **Google Sheets API** and **Google Drive API**
4. Create a **Service Account**:
   - Go to "IAM & Admin" → "Service Accounts"
   - Click "Create Service Account"
   - Name it (e.g., "realestate-avm-api")
   - Grant it "Editor" role
   - Click "Create Key" → Choose JSON format
   - Save the downloaded JSON file securely

### Step 2: Share Your Sheet

1. Open your Google Sheet
2. Click "Share" button
3. Add the service account email (from the JSON file, looks like `xxx@xxx.iam.gserviceaccount.com`)
4. Grant "Editor" access

### Step 3: Configure Credentials

**Option A: Environment Variable (Recommended)**
```bash
export GOOGLE_SHEETS_CREDENTIALS="/path/to/credentials.json"
```

**Option B: Pass in Request**
```json
{
  "credentials_path": "/path/to/credentials.json",
  ...
}
```

---

## Google Sheet Format

Your sheet must have these columns **in this exact order** (with header row):

| Column | Name | Type | Example | Required |
|--------|------|------|---------|----------|
| A | bedrooms | int | 3 | ✅ |
| B | bathrooms | float | 2.5 | ✅ |
| C | sqft_living | int | 2000 | ✅ |
| D | sqft_lot | int | 5000 | ✅ |
| E | floors | float | 2.0 | ✅ |
| F | year_built | int | 2005 | ✅ |
| G | year_renovated | int | 0 | ✅ |
| H | latitude | float | 47.5112 | ✅ |
| I | longitude | float | -122.257 | ✅ |
| J | property_type | string | Single Family | ✅ |
| K | neighborhood | string | Downtown | ✅ |
| L | condition | string | Good | ✅ |
| M | view_quality | string | Fair | ✅ |
| N | description | string | Beautiful home... | ❌ |

### Valid Values

**property_type**: `Single Family`, `Townhouse`, `Condo`, `Multi-Family`

**condition**: `Poor`, `Fair`, `Average`, `Good`, `Excellent`

**view_quality**: `None`, `Fair`, `Good`, `Excellent`

### Example Sheet

```
| bedrooms | bathrooms | sqft_living | sqft_lot | floors | year_built | year_renovated | latitude | longitude | property_type | neighborhood | condition | view_quality | description |
|----------|-----------|-------------|----------|--------|------------|----------------|----------|-----------|---------------|--------------|-----------|--------------|-------------|
| 3 | 2.5 | 2000 | 5000 | 2.0 | 2005 | 0 | 47.5112 | -122.257 | Single Family | Downtown | Good | Fair | Modern home |
| 4 | 3.0 | 2500 | 6000 | 2.0 | 2010 | 2020 | 47.6062 | -122.332 | Townhouse | Capitol Hill | Excellent | Good | Renovated |
```

---

## API Usage

### Endpoint

```
POST /sheets/predict
```

### Request Body

```json
{
  "sheet_url": "https://docs.google.com/spreadsheets/d/YOUR_SHEET_ID/edit",
  "sheet_name": "Sheet1",
  "start_row": 2,
  "write_back": true,
  "use_ensemble": true,
  "credentials_path": null
}
```

**Parameters:**
- `sheet_url` (required): Full Google Sheets URL or just the Sheet ID
- `sheet_name` (optional): Tab name to read from (default: "Sheet1")
- `start_row` (optional): Row to start reading (default: 2, skips header)
- `write_back` (optional): Write predictions back to sheet (default: true)
- `use_ensemble` (optional): Use ensemble model (default: true)
- `credentials_path` (optional): Path to credentials JSON (default: env var)

### Response

```json
{
  "sheet_id": "1ABC...XYZ",
  "total_properties": 10,
  "successful_predictions": 10,
  "failed_predictions": 0,
  "predictions": [
    {
      "property_id": "row_2",
      "predicted_price": 525000.50,
      "confidence_score": 0.89,
      "model_breakdown": {
        "tabular": 520000.00,
        "image": 535000.00,
        "text": 530000.00
      },
      "timestamp": "2025-01-15T10:30:00"
    }
  ],
  "written_back": true,
  "timestamp": "2025-01-15T10:30:00"
}
```

### Results Written to Sheet

When `write_back: true`, predictions are written to columns **N, O, P**:

| Column | Name | Example |
|--------|------|---------|
| N | predicted_price | 525000.50 |
| O | confidence | 0.89 |
| P | timestamp | 2025-01-15 10:30:00 |

---

## Examples

### Python Example

```python
import requests
import json

# API endpoint
url = "http://localhost:8000/sheets/predict"

# Request body
payload = {
    "sheet_url": "https://docs.google.com/spreadsheets/d/1ABC...XYZ/edit",
    "sheet_name": "Properties",
    "start_row": 2,
    "write_back": True,
    "use_ensemble": True
}

# Make request
response = requests.post(url, json=payload)

# Check response
if response.status_code == 200:
    result = response.json()
    print(f"✅ Processed {result['total_properties']} properties")
    print(f"✅ Successful: {result['successful_predictions']}")
    print(f"❌ Failed: {result['failed_predictions']}")
else:
    print(f"Error: {response.status_code}")
    print(response.json())
```

### cURL Example

```bash
curl -X POST "http://localhost:8000/sheets/predict" \
  -H "Content-Type: application/json" \
  -d '{
    "sheet_url": "https://docs.google.com/spreadsheets/d/1ABC...XYZ/edit",
    "sheet_name": "Sheet1",
    "start_row": 2,
    "write_back": true,
    "use_ensemble": true
  }'
```

### JavaScript Example

```javascript
const response = await fetch('http://localhost:8000/sheets/predict', {
  method: 'POST',
  headers: {
    'Content-Type': 'application/json',
  },
  body: JSON.stringify({
    sheet_url: 'https://docs.google.com/spreadsheets/d/1ABC...XYZ/edit',
    sheet_name: 'Sheet1',
    start_row: 2,
    write_back: true,
    use_ensemble: true
  })
});

const data = await response.json();
console.log(data);
```

---

## Health Check

Check if Google Sheets integration is properly configured:

```bash
GET /sheets/health
```

Response:
```json
{
  "status": "ready",
  "credentials_configured": true,
  "model_loaded": true,
  "timestamp": "2025-01-15T10:30:00"
}
```

---

## Troubleshooting

### Error: "Credentials not provided"
**Solution:** Set `GOOGLE_SHEETS_CREDENTIALS` environment variable or pass `credentials_path` in request.

### Error: "Spreadsheet not found"
**Solutions:**
- Ensure service account has access to the sheet (check "Share" settings)
- Verify the sheet URL/ID is correct
- Check that the sheet isn't deleted

### Error: "Worksheet not found"
**Solution:** Check that `sheet_name` matches the tab name exactly (case-sensitive).

### Error: "Parse failed"
**Solutions:**
- Verify column order matches the required format
- Check for empty/invalid values in required columns
- Ensure data types are correct (e.g., numbers not text)

### Predictions not written back
**Solutions:**
- Set `write_back: true` in request
- Ensure service account has "Editor" access to the sheet
- Check API logs for write errors (non-fatal)

---

## Best Practices

1. **Use consistent data formats** - Keep property_type, condition, view_quality values standardized
2. **Include descriptions** - Text descriptions improve prediction accuracy
3. **Batch processing** - Process multiple properties in one request for efficiency
4. **Monitor confidence scores** - Low confidence (<0.7) may indicate unusual properties
5. **Keep credentials secure** - Never commit credentials to version control
6. **Use environment variables** - Store credentials path in env vars, not in code

---

## Next Steps

- Explore the interactive API docs at http://localhost:8000/docs
- Try the batch prediction endpoint for non-Sheets workflows
- Set up automated scheduled predictions using cron/Task Scheduler
- Integrate with your existing data pipelines

---

## Support

For issues or questions:
- Check API documentation: http://localhost:8000/docs
- View logs for detailed error messages
- Ensure all dependencies are installed: `pip install -r requirements.txt`
