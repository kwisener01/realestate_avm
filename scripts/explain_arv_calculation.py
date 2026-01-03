"""
Explain ARV calculation for a specific property
"""
import pandas as pd
import sys

# Constants
REPAIR_COST_PER_SQFT = 45  # Standard repair cost assumption

# Load the ARV multiples
zip_df = pd.read_csv('data/arv_multiples_by_area.csv')
city_df = pd.read_csv('data/arv_multiples_by_city.csv')

# Property details
zipcode = '30102'
city = 'Acworth'
list_price = 170000
sqft_living = 1800  # Living area square footage

print('='*80)
print('ARV CALCULATION FOR: 500 Gayle Drive SE, Acworth, GA 30102')
print('List Price: $170,000')
print('='*80)

# Step 1: Check zip code match
print('\nSTEP 1: Check for Zip Code 30102')
zip_match = zip_df[zip_df['Zip'] == zipcode]
if len(zip_match) > 0:
    print(f'  [FOUND] Zip {zipcode} in database')
    print(f'  Sample Size: {zip_match.iloc[0]["Sample_Size"]} properties')
    print(f'  Conservative: {zip_match.iloc[0]["Conservative_Multiple"]}x')
    print(f'  Moderate: {zip_match.iloc[0]["Moderate_Multiple"]}x')
    print(f'  Aggressive: {zip_match.iloc[0]["Aggressive_Multiple"]}x')
    mult_conservative = zip_match.iloc[0]['Conservative_Multiple']
    mult_moderate = zip_match.iloc[0]['Moderate_Multiple']
    mult_aggressive = zip_match.iloc[0]['Aggressive_Multiple']
    source = f'Zip {zipcode}'
    sample_size = zip_match.iloc[0]['Sample_Size']
else:
    print(f'  [NOT FOUND] Zip {zipcode} NOT in database')
    mult_conservative = None
    mult_moderate = None
    mult_aggressive = None
    sample_size = 0

# Step 2: Check city match
print(f'\nSTEP 2: Check for City "{city}"')
city_match = city_df[city_df['City'].str.lower() == city.lower()]
if len(city_match) > 0:
    print(f'  [FOUND] {city} in database')
    print(f'  Sample Size: {int(city_match.iloc[0]["Count"])} properties')
    print(f'  Conservative: {city_match.iloc[0]["Conservative"]}x')
    print(f'  Moderate: {city_match.iloc[0]["Median"]}x')
    print(f'  Aggressive: {city_match.iloc[0]["Aggressive"]}x')
    if mult_conservative is None:
        mult_conservative = city_match.iloc[0]['Conservative']
        mult_moderate = city_match.iloc[0]['Median']
        mult_aggressive = city_match.iloc[0]['Aggressive']
        source = f'City {city}'
        sample_size = int(city_match.iloc[0]['Count'])
else:
    print(f'  [NOT FOUND] City {city} NOT in database')

# Step 3: Use defaults if no match
if mult_conservative is None:
    print('\nSTEP 3: Using DEFAULT values (no area match)')
    mult_conservative = 2.59
    mult_moderate = 2.82
    mult_aggressive = 3.22
    source = 'Default (overall average)'
    sample_size = 598
    print(f'  Conservative: {mult_conservative}x')
    print(f'  Moderate: {mult_moderate}x')
    print(f'  Aggressive: {mult_aggressive}x')
    print(f'  Based on: 598 properties (Atlanta metro average)')

# Step 4: Estimate assessed value
print('\nSTEP 4: Estimate Assessed Value')
print('  (Since assessed value not provided in MLS data)')
assessed_value = list_price * 0.35
print(f'  Formula: Assessed Value = List Price x 0.35')
print(f'  Calculation: ${list_price:,} x 0.35 = ${assessed_value:,.0f}')
print(f'  Note: 0.35 is typical ratio of assessed to market value')

# Step 5: Calculate ARVs
print('\nSTEP 5: Calculate ARV Estimates')
print(f'  Using multiples from: {source}')
print(f'  Sample size: {sample_size} properties')
arv_conservative = assessed_value * mult_conservative
arv_moderate = assessed_value * mult_moderate
arv_aggressive = assessed_value * mult_aggressive

print(f'\n  Conservative ARV = ${assessed_value:,.0f} x {mult_conservative} = ${arv_conservative:,.0f}')
print(f'  Moderate ARV     = ${assessed_value:,.0f} x {mult_moderate} = ${arv_moderate:,.0f}')
print(f'  Aggressive ARV   = ${assessed_value:,.0f} x {mult_aggressive} = ${arv_aggressive:,.0f}')

# Step 6: Determine deal status
print('\nSTEP 6: Determine Deal Status')
arv_ratio = list_price / arv_moderate
print(f'  List Price / Moderate ARV = ${list_price:,} / ${arv_moderate:,.0f} = {arv_ratio:.1%}')

if arv_ratio <= 0.50:
    deal_status = 'GOOD DEAL'
    explanation = 'List price <= 50% of ARV (meets 50% rule)'
elif arv_ratio <= 0.60:
    deal_status = 'MAYBE'
    explanation = 'List price <= 60% of ARV (marginal deal)'
else:
    deal_status = 'NO DEAL'
    explanation = 'List price > 60% of ARV (too expensive)'

print(f'  Deal Status: {deal_status}')
print(f'  Reason: {explanation}')

# Calculate potential profit
print('\nSTEP 7: Potential Profit Analysis')
estimated_rehab = sqft_living * REPAIR_COST_PER_SQFT  # Calculate based on square footage
total_cost = list_price + estimated_rehab
potential_profit = arv_moderate - total_cost
roi = (potential_profit / total_cost) * 100

print(f'  Purchase Price:     ${list_price:,.0f}')
print(f'  Living Area:        {sqft_living:,} sqft')
print(f'  Estimated Rehab:    ${estimated_rehab:,.0f} (${REPAIR_COST_PER_SQFT}/sqft × {sqft_living:,} sqft)')
print(f'  Total Investment:   ${total_cost:,.0f}')
print(f'  ARV (Moderate):     ${arv_moderate:,.0f}')
print(f'  Potential Profit:   ${potential_profit:,.0f}')
print(f'  ROI:                {roi:.1f}%')

# Determine confidence
if sample_size >= 20:
    confidence = 'HIGH'
elif sample_size >= 10:
    confidence = 'MEDIUM'
else:
    confidence = 'LOW'

print('\n' + '='*80)
print('FINAL OUTPUT (What appears in your Google Sheet):')
print('='*80)
print(f'  Column X (Deal Status):      {deal_status}')
print(f'  Column Y (Conservative ARV): ${arv_conservative:,.0f}')
print(f'  Column Z (Moderate ARV):     ${arv_moderate:,.0f}  <- USE THIS')
print(f'  Column AA (Aggressive ARV):  ${arv_aggressive:,.0f}')
print(f'  Column AB (Confidence):      {confidence}')
print('='*80)

# Recommendation
print('\nRECOMMENDATION:')
if deal_status == 'GOOD DEAL':
    print(f'  This could be a good flip opportunity!')
    print(f'  Potential profit: ${potential_profit:,.0f}')
elif deal_status == 'MAYBE':
    print(f'  Marginal deal - analyze carefully')
    print(f'  Profit margin is tight: ${potential_profit:,.0f}')
else:
    print(f'  Not recommended - list price too high relative to ARV')
    print(f'  Would need price reduction to ${arv_moderate * 0.5:,.0f} to be a deal')
