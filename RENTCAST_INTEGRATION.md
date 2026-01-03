# Rentcast API Integration

## Overview

This application uses the [Rentcast API](https://developers.rentcast.io/) to obtain accurate property valuations and comparable sales data for real estate flip analysis.

### Why Rentcast?

- **Market Value Estimates**: Professional-grade AVM (Automated Valuation Model) for property valuations
- **Comparable Sales**: Access to recent comparable property sales data
- **Property Details**: Fetch missing property attributes (square footage, beds, baths)
- **High Accuracy**: Optimized with best-practice parameters for precise valuations

---

## API Configuration

### API Key

The Rentcast API key is configured via environment variable:

```bash
RENTCAST_API_KEY=09941f5d40cc4eb896e8322c691ed644
```

### Service Location

- **File**: `app/services/rentcast_api.py`
- **Class**: `RentcastAPIService`

---

## API Methods

### 1. `get_value_estimate()`

Fetches property market value with optimized accuracy parameters.

**Parameters:**
- `address` (required): Street address
- `city`: City name
- `state`: State code (default: "GA")
- `zipcode`: ZIP code
- `property_type`: Property type (default: "Single Family")
- `bedrooms`: Number of bedrooms (optional, improves accuracy)
- `bathrooms`: Number of bathrooms (optional, improves accuracy)
- `square_footage`: Square footage (optional, improves accuracy)
- `max_radius`: Maximum distance for comps in miles (default: 3.0)
- `days_old`: Maximum age of comps in days (default: 180)
- `comp_count`: Number of comps to use (default: 20)

**Returns:** Property value as float, or None if unavailable

**Example:**
```python
rentcast = RentcastAPIService()
value = rentcast.get_value_estimate(
    address="123 Main St",
    city="Atlanta",
    zipcode="30301",
    square_footage=2000,
    bedrooms=3,
    bathrooms=2.5
)
```

### 2. `get_property_data()`

Fetches complete property data including value, comparables, and attributes.

**Parameters:** Same as `get_value_estimate()`

**Returns:** Complete property data dictionary with:
- `price` or `value`: Property valuation
- `comparables`: Array of comparable properties
- `squareFootage`: Property square footage
- Additional property details

**Example:**
```python
data = rentcast.get_property_data(
    address="123 Main St",
    city="Atlanta",
    zipcode="30301",
    bedrooms=3,
    bathrooms=2.5
)

if data and 'comparables' in data:
    comps = data['comparables'][:3]  # Top 3 comps
```

### 3. `get_comparables()`

Convenience method to fetch only comparable properties.

**Parameters:** Same as `get_property_data()`

**Returns:** List of comparable properties sorted by correlation

---

## Accuracy Optimization

### Property Attributes

Following [Rentcast's accuracy guidelines](https://developers.rentcast.io/reference/property-valuation#increasing-avm-accuracy), we pass property details to improve valuation accuracy:

1. **Property Type**: Always set to "Single Family" for residential flips
2. **Bedrooms**: Extracted from sheet columns when available
3. **Bathrooms**: Extracted from sheet columns when available
4. **Square Footage**: From sheet data or fetched via Rentcast

### Comparable Selection Parameters

Optimized for metro Atlanta market:

| Parameter | Value | Purpose |
|-----------|-------|---------|
| `maxRadius` | 3.0 miles | Hyper-local comps for dense urban areas |
| `daysOld` | 180 days | Recent sales (6 months) for current market |
| `compCount` | 20 | Larger pool for better accuracy |

**Rationale:**
- **3 miles**: Metro Atlanta has dense housing stock; smaller radius ensures comps are truly comparable
- **180 days**: Recent enough to reflect current market while maintaining sufficient comp pool
- **20 comps**: Larger sample size reduces outlier impact and improves statistical accuracy

---

## Google Sheets Integration

### Column Extraction

The system automatically extracts property details from Google Sheets:

**Bedrooms** - Looks for columns:
- Bedrooms, Beds, Bed, BR

**Bathrooms** - Looks for columns:
- Bathrooms, Baths, Bath, BA

**Square Footage** - Looks for columns:
- Building Sqft, Sqft, Square Feet, Sq Ft, Sqft Living, Square Footage

### API Call Flow

1. **Extract property data** from sheet (address, sqft, beds, baths)
2. **Calculate flip analysis** using available data
3. **Fetch market value** from Rentcast with all property attributes
4. **Fetch comparables** for GOOD DEAL properties only (saves API calls)
5. **Write results** back to sheet with deal quality indicators

### Sqft Fallback Logic

If square footage is missing from sheet:
1. Call Rentcast `get_property_data()` with beds/baths
2. Extract `squareFootage` from response
3. Mark source as "Rentcast" in `Sqft_Source` column
4. Proceed with flip calculation using fetched sqft

---

## Output Columns

### Market Value Analysis (Columns X-AC)

| Column | Name | Description |
|--------|------|-------------|
| X | Deal_Status | GOOD DEAL / MAYBE / NO DEAL |
| Y | Market_Value | Rentcast property valuation |
| Z | ARV_Needed | ARV needed for 20% ROI |
| AA | Market_Value_vs_ARV | Difference (Market - ARV) |
| AB | Market_Supports_Deal | YES if value ≥ ARV |
| AC | Maximum_Allowable_Offer | 50% of ARV Needed |

### Comparable Properties (Columns AD-AF)

Only populated for **GOOD DEAL** properties:

**Format:** `Address | $Price | Date | Beds/Baths | Sqft`

**Example:** `123 Oak St, Atlanta GA | $425,000 | 2024-12-15 | 3bd/2ba | 2000sf`

---

## Deal Quality Logic

Deal quality is determined by comparing Rentcast Market Value to ARV Needed:

```python
if market_value >= arv_needed * 1.05:  # 5% cushion
    deal_status = "GOOD DEAL"
elif market_value >= arv_needed:
    deal_status = "MAYBE"
else:
    deal_status = "NO DEAL"
```

### Decision Tree

```
Market Value ≥ ARV Needed × 1.05 → GOOD DEAL
    ↓ (YES: Fetch comparables)
    └─ Shows 3 comparable properties

Market Value ≥ ARV Needed → MAYBE
    └─ (Marginal deal, minimal profit)

Market Value < ARV Needed → NO DEAL
    └─ (Insufficient value to support flip)
```

---

## API Response Examples

### Successful Valuation Response

```json
{
  "price": 425000,
  "confidence": "High",
  "squareFootage": 2000,
  "bedrooms": 3,
  "bathrooms": 2.5,
  "comparables": [
    {
      "formattedAddress": "123 Oak St, Atlanta GA 30301",
      "price": 420000,
      "lastSeenDate": "2024-12-15",
      "squareFootage": 1950,
      "bedrooms": 3,
      "bathrooms": 2.0
    }
  ]
}
```

### Error Handling

```python
# API call fails gracefully
market_value = rentcast.get_value_estimate(address, city, "GA", zipcode)

if not market_value:
    # Shows UNKNOWN status with ARV_Needed still calculated
    deal_status = "UNKNOWN"
    market_value_display = "Not Available"
```

---

## Best Practices

### 1. Always Pass Property Attributes

✅ **Good:**
```python
rentcast.get_value_estimate(
    address=address,
    city=city,
    zipcode=zipcode,
    square_footage=sqft,
    bedrooms=beds,
    bathrooms=baths
)
```

❌ **Avoid:**
```python
# Missing property attributes reduces accuracy
rentcast.get_value_estimate(address, city)
```

### 2. Optimize Comp Parameters for Market

For **dense urban areas** (Atlanta, NYC, etc.):
- Use smaller `maxRadius` (2-4 miles)
- Recent `daysOld` (90-180 days)
- Higher `compCount` (15-25)

For **suburban/rural areas**:
- Use larger `maxRadius` (5-10 miles)
- Longer `daysOld` (180-365 days)
- Standard `compCount` (10-15)

### 3. Cache Responses When Possible

Rentcast API has rate limits. Consider caching:
- Property valuations (valid for ~30 days)
- Comparable sales (valid for ~60 days)

---

## Error Scenarios

| Scenario | Behavior | Output |
|----------|----------|--------|
| **API Key Invalid** | Logs error, returns None | Deal_Status: ERROR |
| **Address Not Found** | Returns None gracefully | Deal_Status: UNKNOWN |
| **No Comparables Found** | Value returned, comps empty | Shows value, empty comps |
| **API Rate Limited** | Logs 429 error, returns None | Deal_Status: ERROR |
| **Network Timeout** | Timeout after 10 seconds | Deal_Status: ERROR |

---

## Testing

### Manual Test

```python
from app.services.rentcast_api import RentcastAPIService

rentcast = RentcastAPIService()

# Test valuation
value = rentcast.get_value_estimate(
    address="500 Gayle Dr",
    city="Acworth",
    state="GA",
    square_footage=1632,
    bedrooms=3,
    bathrooms=2.0
)

print(f"Value: ${value:,.0f}")

# Test with comps
data = rentcast.get_property_data(
    address="500 Gayle Dr",
    city="Acworth",
    state="GA"
)

if data and 'comparables' in data:
    print(f"Found {len(data['comparables'])} comparables")
```

### Expected Output

```
Value: $312,000
Found 20 comparables
```

---

## Deployment

### Environment Variables

Ensure `RENTCAST_API_KEY` is set in Railway:

```bash
railway variables set RENTCAST_API_KEY=09941f5d40cc4eb896e8322c691ed644
```

### Verification

After deployment, test the health endpoint:

```bash
curl https://realestatefearless-perfection-production.up.railway.app/health
```

---

## API Limits and Pricing

**Rentcast API Pricing:** ~$0.10-0.50 per lookup

**Rate Limits:** Check [Rentcast documentation](https://developers.rentcast.io/) for current limits

**Cost Optimization:**
- Only fetch comps for GOOD DEAL properties
- Cache results when appropriate
- Use batch processing for large datasets

---

## Related Documentation

- [Rentcast API Documentation](https://developers.rentcast.io/)
- [Increasing AVM Accuracy](https://developers.rentcast.io/reference/property-valuation#increasing-avm-accuracy)
- [Google Sheets Integration](GOOGLE_SHEETS_INTEGRATION.md)
- [Flip Calculator API](FLIP_CALCULATOR_API_DOCS.md)
- [Railway Deployment](RAILWAY_DEPLOYMENT_READY.md)

---

## Changelog

### 2026-01-03
- ✅ Added property attributes (sqft, beds, baths) for improved accuracy
- ✅ Optimized comparable selection parameters (maxRadius, daysOld, compCount)
- ✅ Implemented automatic bedrooms/bathrooms extraction from sheets
- ✅ Added sqft fallback via Rentcast when missing from sheets

### 2024-XX-XX
- ✅ Initial Rentcast integration
- ✅ Replaced Zillow API with Rentcast for reliable valuations
- ✅ Added comparable properties fetching for GOOD DEAL properties
