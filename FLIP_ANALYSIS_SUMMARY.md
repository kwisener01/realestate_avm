# Real Estate Flip Analysis Summary

**Analysis Date:** 2025-12-28
**Dataset:** 729 properties in Atlanta Metro / DeKalb County area

---

## Executive Summary

We analyzed 729 properties to identify flip transactions and calculate ARV (After Repair Value) estimation multiples. The analysis identified:

- **598 potential flip properties** (sold within 3 years at 20%+ above assessed value)
- **152 properties with confirmed profit** (MLS data showing buy and sell prices)
- **13 quick flips** (properties bought and sold within 1 year)

---

## Key Findings: ARV Estimation Multiples

### Method 1: Based on Assessed Value (598 properties)

| Scenario | Multiple | Description |
|----------|----------|-------------|
| **Conservative** | 2.59x | 25th percentile - Safe estimate for most deals |
| **Moderate** | 2.82x | Median - Most reliable middle-ground estimate |
| **Aggressive** | 3.22x | 75th percentile - Hot market/premium properties |
| **Average** | 3.04x | Mean - Overall average across all flips |

**Range:** 1.55x to 19.93x (outliers exist)

### Method 2: Based on Purchase Price (152 properties with MLS data)

| Scenario | Multiple | Description |
|----------|----------|-------------|
| **Conservative** | 1.25x | 25th percentile - Minimal profit scenario |
| **Moderate** | 1.57x | Median - Typical flip profit margin |
| **Aggressive** | 1.93x | 75th percentile - High-profit flips |

**Profit Statistics:**
- Average profit: $307,161
- Median profit: $223,950
- Average margin: 79.7%
- Median margin: 56.6%

### Method 3: Quick Flips Only (13 properties held <= 1 year)

| Scenario | Multiple | Description |
|----------|----------|-------------|
| **Conservative** | 1.06x | 25th percentile - Minimal quick profit |
| **Moderate** | 1.40x | Median - Typical quick flip return |
| **Aggressive** | 1.78x | 75th percentile - Excellent quick flip |
| **Average** | 1.52x | Mean - Average quick flip multiple |

**Quick Flip Statistics:**
- Average holding period: 132 days (4.4 months)
- Median holding period: 144 days (4.8 months)
- Average profit: $132,112
- Median profit: $160,000
- Average margin: 51.8%
- Median margin: 40.0%

---

## ARV Estimation by City

| City | Sample Size | Median Multiple | Median Profit | Notes |
|------|-------------|-----------------|---------------|-------|
| **Scottdale** | 7 | 2.52x | - | Highest average (4.37x) but volatile |
| **Atlanta** | 254 | 2.88x | $221,850 | Largest sample, reliable data |
| **Brookhaven** | 87 | 2.88x | $362,500 | High-value properties |
| **Stone Mountain** | 52 | 2.93x | $206,500 | Good appreciation |
| **Decatur** | 110 | 2.73x | $207,950 | Stable market |
| **Lithonia** | 50 | 2.63x | $159,482 | Lower price point |
| **Tucker** | 17 | 2.73x | $188,900 | Limited sample |

---

## Recommended ARV Estimation Strategy

### For Investment Analysis:

1. **Conservative Approach** (Reduce Risk)
   - Use **2.59x assessed value** OR **1.25x purchase price**
   - Best for: New investors, competitive markets, uncertain conditions

2. **Moderate Approach** (Balanced)
   - Use **2.82x assessed value** OR **1.57x purchase price**
   - Best for: Experienced investors, typical market conditions

3. **Aggressive Approach** (Maximum Profit)
   - Use **3.22x assessed value** OR **1.93x purchase price**
   - Best for: Hot markets, premium locations (Brookhaven, Atlanta core), excellent condition properties

### For Quick Flips (< 1 Year Hold):

- **Conservative:** 1.06x purchase price (break-even + costs)
- **Moderate:** 1.40x purchase price (40% profit margin)
- **Aggressive:** 1.78x purchase price (78% profit margin)

### Adjustment Factors:

Add/subtract 10-20% based on:
- Property condition (Interior, Exterior, Kitchen, Bath)
- Location premium (proximity to city center, schools, amenities)
- Market timing (current market conditions)
- Days on market (longer DOM may indicate lower ARV)

---

## Top Flip Examples

### Highest Profit Quick Flips (<= 1 Year):

1. **3047 Terramar Dr, Atlanta**
   - Held: 148 days
   - Bought: $475,000 → Sold: $800,000
   - Profit: $325,000 (68% margin)
   - Multiple: 1.68x

2. **6241 Noreen Way, Lithonia**
   - Held: 219 days
   - Bought: $99,242 → Sold: $318,000
   - Profit: $218,758 (220% margin!)
   - Multiple: 3.20x

3. **818 Mountain View Run, Stone Mountain**
   - Held: 144 days
   - Bought: $235,000 → Sold: $440,000
   - Profit: $205,000 (87% margin)
   - Multiple: 1.87x

### Highest Overall Profits:

1. **2939 Mabry Ln Ne, Brookhaven**
   - Bought: $475,000 → Sold: $2,650,000
   - Profit: $2,175,000 (458% margin)
   - Multiple: 5.58x

2. **1282 Kendrick Rd Ne, Brookhaven**
   - Bought: $547,994 → Sold: $1,715,000
   - Profit: $1,167,006 (213% margin)
   - Multiple: 3.13x

3. **3158 Saybrook Dr Ne, Brookhaven**
   - Bought: $363,000 → Sold: $1,525,000
   - Profit: $1,162,000 (320% margin)
   - Multiple: 4.20x

---

## Using These Multiples in Your Analysis

### Example Calculation:

**Property Details:**
- Address: 123 Main St, Atlanta
- Assessed Value: $100,000
- Purchase Price: $80,000
- Estimated Rehab: $30,000

**ARV Estimation:**

1. **Method 1 (Assessed Value):**
   - Conservative: $100,000 × 2.59 = $259,000
   - Moderate: $100,000 × 2.82 = $282,000
   - Aggressive: $100,000 × 3.22 = $322,000

2. **Method 2 (Purchase Price for 1-year flip):**
   - Conservative: $80,000 × 1.40 = $112,000 (not viable after rehab)
   - Moderate: $80,000 × 1.57 = $125,600 (marginal)
   - Aggressive: $80,000 × 1.93 = $154,400 (profitable)

3. **70% Rule Check:**
   - Target: ARV × 70% - Rehab = Max Purchase Price
   - Using moderate ARV ($282,000): $282,000 × 0.70 - $30,000 = $167,400
   - Current: $80,000 (Great deal! Well below max)

---

## Data Files Generated

1. `data/flip_analysis_results.csv` - 598 potential flips with all metrics
2. `data/mls_flip_analysis.csv` - 152 confirmed flips with profit data
3. `data/quick_flip_analysis.csv` - 13 quick flips (≤ 1 year)

---

## Recommendations

1. **Use the 2.82x multiple** as your baseline ARV estimate (based on assessed value)
2. **For quick flips**, expect 1.40x-1.57x returns on purchase price
3. **Brookhaven and Atlanta** show highest profit potential but require larger capital
4. **Stone Mountain and Lithonia** offer entry-level opportunities with solid returns
5. **Hold time averages 132 days** for quick flips - factor into financing costs
6. **Median profit margin of 57%** provides cushion for unexpected costs

---

**Note:** These multiples are based on historical data from the Atlanta metro area (2022-2025). Always conduct property-specific analysis and adjust for current market conditions, property condition, and location factors.
