# Flip Calculator - Complete Calculations & Formulas

## Overview

This document details every calculation performed by the flip calculator, including all formulas, parameters, and data flows used to analyze real estate flip deals.

---

## 📊 Parameters & Defaults

### Customizable Parameters (via Web UI)

All parameters can be adjusted through the web interface at the root URL.

| Parameter | Default Value | Description | Range |
|-----------|--------------|-------------|-------|
| **Repair Cost per Sqft** | $45 | Cost to renovate per square foot | ≥ $0 |
| **Hold Time (Months)** | 5 | Expected time to flip property | ≥ 1 month |
| **Annual Interest Rate** | 10% (0.10) | Hard money loan interest rate | 0-100% |
| **Loan Points** | 1% (0.01) | Upfront loan origination fee | 0-100% |
| **Loan to Cost Ratio** | 90% (0.90) | Percentage of costs financed | 0-100% |
| **Monthly HOA/Maintenance** | $150 | Monthly HOA and maintenance | ≥ $0 |
| **Monthly Insurance** | $100 | Monthly property insurance | ≥ $0 |
| **Monthly Utilities** | $150 | Monthly utility costs | ≥ $0 |
| **Property Tax Rate (Annual)** | 1.2% (0.012) | Annual property tax rate | 0-100% |
| **Closing Costs - Buy** | 1% (0.01) | Purchase closing costs | 0-100% |
| **Closing Costs - Sell** | 1% (0.01) | Sale closing costs | 0-100% |
| **Seller Credit** | 3% (0.03) | Credit to buyer at closing | 0-100% |
| **Staging & Marketing** | $2,000 | Staging and marketing budget | ≥ $0 |
| **Listing Commission** | 2.5% (0.025) | Listing agent commission | 0-100% |
| **Buyer Commission** | 2.5% (0.025) | Buyer's agent commission | 0-100% |

---

## 🧮 Core Calculations

### 1. Acquisition Costs

```python
# Purchase Price (from MLS data)
purchase_price = list_price

# Closing Costs on Purchase
closing_costs_buy = purchase_price × closing_costs_buy_percent

# Total Acquisition Cost
total_acquisition = purchase_price + closing_costs_buy
```

**Example:**
- Purchase Price: $250,000
- Closing Costs (1%): $2,500
- **Total Acquisition: $252,500**

---

### 2. Renovation Costs

```python
# Repair Cost Calculation
repair_cost = sqft_living × repair_cost_per_sqft

# Total Renovation Cost
total_renovation = repair_cost
```

**Example:**
- Square Footage: 2,000 sqft
- Repair Cost per Sqft: $45
- **Total Renovation: $90,000**

---

### 3. Financing Costs

```python
# Loan Amount (financed portion of all costs)
loan_amount = (purchase_price + repair_cost) × loan_to_cost_ratio

# Upfront Loan Points
loan_points_cost = loan_amount × loan_points

# Monthly Interest Payment
monthly_interest_rate = interest_rate_annual / 12
monthly_interest_payment = loan_amount × monthly_interest_rate

# Total Interest Over Hold Period
total_interest = monthly_interest_payment × hold_time_months

# Total Financing Cost
total_financing = loan_points_cost + total_interest
```

**Example:**
- Loan Amount (90% of $340,000): $306,000
- Loan Points (1%): $3,060
- Monthly Interest (10% annual / 12): $2,550/month
- Hold Time: 5 months
- Total Interest: $12,750
- **Total Financing: $15,810**

---

### 4. Holding Costs

```python
# Monthly Property Tax
monthly_property_tax = (purchase_price × property_tax_rate_annual) / 12

# Monthly Holding Costs
monthly_holding = (
    monthly_property_tax +
    monthly_insurance +
    monthly_utilities +
    monthly_hoa_maintenance
)

# Total Holding Costs
total_holding_costs = monthly_holding × hold_time_months

# Total Holding (Financing + Holding Costs)
total_holding = total_financing + total_holding_costs
```

**Example:**
- Monthly Property Tax ($250,000 × 1.2% / 12): $250
- Monthly Insurance: $100
- Monthly Utilities: $150
- Monthly HOA/Maintenance: $150
- **Monthly Holding Costs: $650**
- Hold Time: 5 months
- Total Holding Costs: $3,250
- Total Financing: $15,810
- **Total Holding: $19,060**

---

### 5. Selling Costs

```python
# ARV (After Repair Value)
arv = purchase_price  # Can be adjusted, defaults to purchase price

# Listing Agent Commission
listing_commission = arv × listing_commission_rate

# Buyer's Agent Commission
buyer_commission = arv × buyer_commission_rate

# Seller Credit
seller_credit = arv × seller_credit_percent

# Closing Costs on Sale
closing_costs_sell = arv × closing_costs_sell_percent

# Staging and Marketing
staging_marketing = staging_marketing_cost

# Total Selling Costs
total_selling = (
    listing_commission +
    buyer_commission +
    seller_credit +
    closing_costs_sell +
    staging_marketing
)
```

**Example (ARV = $400,000):**
- Listing Commission (2.5%): $10,000
- Buyer Commission (2.5%): $10,000
- Seller Credit (3%): $12,000
- Closing Costs (1%): $4,000
- Staging & Marketing: $2,000
- **Total Selling: $38,000**

---

### 6. Total All-In Costs

```python
# Sum of All Cost Categories
total_all_costs = (
    total_acquisition +
    total_renovation +
    total_holding +
    total_selling
)

# Cash Needed (out-of-pocket)
cash_needed = total_all_costs - loan_amount
```

**Example:**
- Total Acquisition: $252,500
- Total Renovation: $90,000
- Total Holding: $19,060
- Total Selling: $38,000
- **Total All Costs: $399,560**
- Loan Amount: $306,000
- **Cash Needed: $93,560**

---

### 7. Profit Analysis

```python
# Gross Profit
gross_profit = arv - total_all_costs

# Net Profit (after loan repayment)
net_profit = arv - total_all_costs

# ROI (Return on Investment)
roi_percent = (gross_profit / cash_needed) × 100

# Profit Margin
profit_margin_percent = (gross_profit / arv) × 100

# Is Profitable?
is_profitable = gross_profit > 0

# Meets 20% ROI Minimum?
meets_minimum_roi = roi_percent >= 20.0

# Meets 70% Rule?
# (Purchase price + repairs should be ≤ 70% of ARV)
meets_70_percent_rule = (purchase_price + repair_cost) <= (arv × 0.70)

# Maximum Offer (70% Rule)
max_offer_70_rule = (arv × 0.70) - repair_cost
```

**Example (ARV = $400,000):**
- ARV: $400,000
- Total All Costs: $399,560
- **Gross Profit: $440**
- Cash Needed: $93,560
- **ROI: 0.5%** ❌
- **Profit Margin: 0.1%** ❌
- **Is Profitable: YES** ✅
- **Meets 20% ROI: NO** ❌
- Purchase + Repairs: $340,000
- 70% of ARV: $280,000
- **Meets 70% Rule: NO** ❌
- **Max Offer (70% Rule): $190,000**

---

## 🎯 ARV Calculation Logic

### ARV Needed for 20% ROI

The flip calculator works backwards from a target 20% ROI to determine the minimum ARV needed:

```python
# Target ROI
target_roi = 0.20  # 20%

# Calculate ARV Needed
# Formula: ARV = Total Costs × (1 + Target ROI) / (1 - Total Selling Rate)
# Where Total Selling Rate = sum of all selling percentages

total_selling_rate = (
    listing_commission_rate +
    buyer_commission_rate +
    seller_credit_percent +
    closing_costs_sell_percent
)

# ARV calculation (iterative approach)
# Start with estimated ARV, calculate costs, adjust until 20% ROI achieved

recommended_arv = calculate_arv_for_target_roi(
    purchase_price=purchase_price,
    repair_cost=repair_cost,
    total_acquisition=total_acquisition,
    total_renovation=total_renovation,
    total_holding=total_holding,
    target_roi=0.20,
    cash_needed_ratio=cash_needed / total_all_costs,
    total_selling_rate=total_selling_rate
)
```

**Example:**
- Starting with Purchase: $250,000, Repairs: $90,000
- Fixed Costs: $271,560
- Target ROI: 20%
- **Calculated ARV Needed: ~$450,000** (to achieve 20% ROI)

---

## 📈 Deal Quality Determination

### Rentcast Market Value vs ARV Needed

```python
# Get Market Value from Rentcast API
market_value = rentcast.get_value_estimate(
    address=address,
    city=city,
    zipcode=zipcode,
    square_footage=sqft,
    bedrooms=bedrooms,
    bathrooms=bathrooms
)

# Calculate Variance
market_value_vs_arv = market_value - arv_needed

# Determine Deal Quality
if market_value >= arv_needed × 1.05:  # 5% cushion
    deal_status = "GOOD DEAL"
elif market_value >= arv_needed:
    deal_status = "MAYBE"
else:
    deal_status = "NO DEAL"

# Market Supports Deal?
market_supports_deal = "YES" if market_value >= arv_needed else "NO"

# Maximum Allowable Offer (MAO)
mao = arv_needed × 0.50  # 50% of ARV Needed
```

**Decision Logic:**

| Market Value | Deal Status | Explanation |
|--------------|-------------|-------------|
| ≥ ARV × 1.05 | **GOOD DEAL** | 5%+ cushion for profit |
| ≥ ARV | **MAYBE** | Breakeven or minimal profit |
| < ARV | **NO DEAL** | Insufficient value |

**Example:**
- ARV Needed: $450,000
- Rentcast Market Value: $475,000
- Variance: +$25,000
- **Deal Status: GOOD DEAL** ✅
- **Market Supports: YES** ✅
- **MAO: $225,000** (50% of ARV)

---

## 📋 Output Columns (X-AH)

### Market Value Analysis (X-AE)

| Column | Name | Formula | Example |
|--------|------|---------|---------|
| **X** | Deal_Status | Based on market_value vs arv_needed | "GOOD DEAL" |
| **Y** | Market_Value | Rentcast API valuation | $475,000 |
| **Z** | ARV_Needed | Calculated ARV for 20% ROI | $450,000 |
| **AA** | Market_Value_vs_ARV | market_value - arv_needed | +$25,000 |
| **AB** | Market_Supports_Deal | "YES" if market_value ≥ arv_needed | "YES" |
| **AC** | Rehab_Cost | total_renovation | $90,000 |
| **AD** | Total_Cost | total_all_costs | $399,560 |
| **AE** | Maximum_Allowable_Offer | arv_needed × 0.50 | $225,000 |

### Comparable Properties (AF-AH)

| Column | Content | When Shown |
|--------|---------|------------|
| **AF** | Comp_1 | Only for "GOOD DEAL" |
| **AG** | Comp_2 | Only for "GOOD DEAL" |
| **AH** | Comp_3 | Only for "GOOD DEAL" |

**Comp Format:**
```
Address | $Price | Date | Beds/Baths | Sqft

Example:
123 Oak St, Atlanta GA 30301 | $425,000 | 2024-12-15 | 3bd/2ba | 2000sf
```

---

## 🔄 Data Flow

### 1. Input Data (from Google Sheets)

```
Property Row:
- Address
- City
- Zip Code
- List Price (Purchase Price)
- Square Footage (from sheet or Rentcast fallback)
- Bedrooms (optional, improves Rentcast accuracy)
- Bathrooms (optional, improves Rentcast accuracy)
```

### 2. Calculation Pipeline

```
Input Data
    ↓
Flip Calculator Parameters (Web UI or Defaults)
    ↓
Calculate Acquisition Costs
    ↓
Calculate Renovation Costs
    ↓
Calculate Financing Costs
    ↓
Calculate Holding Costs
    ↓
Calculate Selling Costs (based on estimated ARV)
    ↓
Calculate Total All Costs
    ↓
Determine ARV Needed for 20% ROI
    ↓
Fetch Rentcast Market Value
    ↓
Compare Market Value vs ARV Needed
    ↓
Determine Deal Quality
    ↓
Fetch Comparables (if GOOD DEAL)
    ↓
Write Results to Columns X-AH
```

### 3. Output Data (to Google Sheets)

```
11 Columns (X-AH):
- 6 Market Analysis columns
- 2 Cost columns (Rehab, Total)
- 1 MAO column
- 3 Comparable properties columns
```

---

## 🧪 Example Complete Calculation

### Property Details
- **Address:** 500 Gayle Dr, Acworth, GA
- **Purchase Price:** $250,000
- **Square Footage:** 2,000 sqft
- **Bedrooms:** 3
- **Bathrooms:** 2.0

### Step-by-Step Calculation

#### 1. Acquisition
```
Purchase Price:        $250,000
Closing Costs (1%):    $  2,500
────────────────────────────────
Total Acquisition:     $252,500
```

#### 2. Renovation
```
Sqft:                  2,000
Cost per Sqft:         $45
────────────────────────────────
Total Renovation:      $90,000
```

#### 3. Financing
```
Loan Base:             $340,000 (Purchase + Repairs)
Loan Amount (90%):     $306,000
Loan Points (1%):      $  3,060
Monthly Interest:      $  2,550 (10% annual / 12)
Hold Time:             5 months
Total Interest:        $ 12,750
────────────────────────────────
Total Financing:       $ 15,810
```

#### 4. Holding Costs
```
Property Tax:          $   250/month ($250k × 1.2% / 12)
Insurance:             $   100/month
Utilities:             $   150/month
HOA/Maintenance:       $   150/month
────────────────────────────────
Monthly Holding:       $   650/month
Hold Time:             5 months
Total Holding Costs:   $  3,250

Total Holding:         $ 19,060 (Financing + Holding Costs)
```

#### 5. Selling (Based on ARV = $450,000)
```
Listing Commission:    $ 11,250 (2.5%)
Buyer Commission:      $ 11,250 (2.5%)
Seller Credit:         $ 13,500 (3%)
Closing Costs:         $  4,500 (1%)
Staging/Marketing:     $  2,000
────────────────────────────────
Total Selling:         $ 42,500
```

#### 6. Total Costs & Profit
```
Total Acquisition:     $252,500
Total Renovation:      $ 90,000
Total Holding:         $ 19,060
Total Selling:         $ 42,500
────────────────────────────────
Total All Costs:       $404,060

Cash Needed:           $ 98,060 (Total - Loan)

ARV Needed (20% ROI):  $450,000
Gross Profit:          $ 45,940 (ARV - Total Costs)
ROI:                   46.8% (Profit / Cash Needed)
Profit Margin:         10.2% (Profit / ARV)
```

#### 7. Market Analysis
```
Rentcast Market Value: $475,000
ARV Needed:            $450,000
Variance:              +$25,000
────────────────────────────────
Deal Status:           GOOD DEAL ✅
Market Supports:       YES ✅
MAO (50% ARV):         $225,000
```

#### 8. Decision Metrics
```
✅ Is Profitable:       YES (Profit: $45,940)
✅ Meets 20% ROI:       YES (46.8% > 20%)
❌ Meets 70% Rule:      NO ($340k > $315k)
   Max Offer 70%:      $225,000
```

### Final Output to Sheet (X-AH)

| X | Y | Z | AA | AB | AC | AD | AE | AF | AG | AH |
|---|---|---|----|----|----|----|----|----|----|----|
| GOOD DEAL | $475,000 | $450,000 | +$25,000 | YES | $90,000 | $404,060 | $225,000 | [Comp 1] | [Comp 2] | [Comp 3] |

---

## 🎨 Custom Parameters Example

### Conservative Investor Profile

```javascript
// Higher safety margins
{
  repair_cost_per_sqft: 55,        // Higher repair estimate
  hold_time_months: 6,             // Longer hold time
  interest_rate_annual: 0.12,      // Higher interest rate
  monthly_hoa_maintenance: 200,    // Higher maintenance
  listing_commission_rate: 0.03,   // 3% listing
  buyer_commission_rate: 0.03      // 3% buyer
}
```

### Aggressive Investor Profile

```javascript
// Tighter margins, faster flips
{
  repair_cost_per_sqft: 35,        // Lower repair estimate
  hold_time_months: 4,             // Faster flip
  interest_rate_annual: 0.09,      // Lower rate (cash buyer)
  monthly_hoa_maintenance: 100,    // Lower maintenance
  listing_commission_rate: 0.02,   // 2% listing
  buyer_commission_rate: 0.02      // 2% buyer
}
```

---

## 📚 References

- **Flip Calculator Code:** `app/services/flip_calculator.py`
- **API Models:** `app/models/property_models.py`
- **Sheet Processing:** `app/api/routes_sheets.py`
- **Rentcast Integration:** `app/services/rentcast_api.py`
- **Web UI:** `app/static/index.html`

---

## 🔧 API Integration

### Rentcast API Parameters

For maximum accuracy, the following property details are passed to Rentcast:

```python
rentcast.get_value_estimate(
    address=address,
    city=city,
    state="GA",
    zipcode=zipcode,
    property_type="Single Family",
    bedrooms=bedrooms,           # Improves accuracy
    bathrooms=bathrooms,         # Improves accuracy
    square_footage=sqft,         # Improves accuracy
    max_radius=3.0,              # 3 miles (dense Atlanta market)
    days_old=180,                # 6 months of recent sales
    comp_count=20                # Larger sample for accuracy
)
```

### Comparable Properties

For "GOOD DEAL" properties, top 3 comparables are fetched:

```python
comparables = rentcast.get_property_data(...)['comparables'][:3]

# Each comp includes:
- formattedAddress
- price
- lastSeenDate
- squareFootage
- bedrooms
- bathrooms
```

---

## 📊 Summary Statistics

### Cost Distribution (Typical Deal)

```
Acquisition:    62% ($252,500)
Renovation:     22% ($ 90,000)
Holding:         5% ($ 19,060)
Selling:        11% ($ 42,500)
────────────────────────────────
Total:         100% ($404,060)
```

### Profitability Thresholds

| Metric | Threshold | Purpose |
|--------|-----------|---------|
| **ROI** | ≥ 20% | Minimum acceptable return |
| **70% Rule** | Purchase + Repairs ≤ 70% ARV | Traditional rule of thumb |
| **Market Support** | Market Value ≥ ARV Needed | Deal viability |
| **GOOD DEAL Margin** | Market Value ≥ ARV × 1.05 | 5% safety cushion |

---

## 🚀 Quick Start

1. **Open Web UI:** Navigate to Railway app root URL
2. **Adjust Parameters:** Customize flip calculator settings
3. **Enter Sheet URL:** Paste Google Sheets URL
4. **Run Analysis:** Click "Analyze Properties"
5. **Review Results:** Check columns X-AH for deal quality

---

*Last Updated: 2026-01-03*
*Version: 2.0*
