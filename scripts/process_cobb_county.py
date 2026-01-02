"""
Process Cobb County data with filtering for rentals and invalid data

This script:
1. Loads Cobb County property data
2. Filters out rental properties (low prices like $2k-$3k)
3. Removes invalid/incomplete records
4. Runs flip analysis
5. Generates ARV multiples

Usage:
    python scripts/process_cobb_county.py <path_to_cobb_file.csv>
"""

import pandas as pd
import sys
import os
import re

def clean_price(val):
    """Convert price string to numeric"""
    try:
        if pd.isna(val) or val == '':
            return None
        cleaned = re.sub(r'[,$]', '', str(val))
        return float(cleaned)
    except:
        return None

def filter_cobb_data(input_file, output_file='data/cobb_county_properties.csv'):
    """
    Load and filter Cobb County data

    Filters out:
    - Rental properties (prices < $50,000)
    - Properties with missing critical data
    - Duplicates
    - Invalid price ranges
    """
    print("="*100)
    print("PROCESSING COBB COUNTY DATA")
    print("="*100)

    # Load data
    print(f"\nStep 1: Loading data from {input_file}")
    df = pd.read_csv(input_file)
    print(f"  Initial records: {len(df)}")
    print(f"  Columns: {len(df.columns)}")

    # Show sample of columns
    print(f"\n  Key columns found:")
    key_cols = ['City', 'Zip', 'Last Sale Amount', 'MLS Amount', 'Total Assessed Value',
                'List Price', 'Price', 'Last Sale Recording Date']
    for col in key_cols:
        if col in df.columns:
            print(f"    ✓ {col}")

    # Identify price column
    price_col = None
    for col in ['Last Sale Amount', 'MLS Amount', 'List Price', 'Price', 'Sale Price']:
        if col in df.columns:
            price_col = col
            print(f"\n  Using '{col}' as price column")
            break

    if not price_col:
        print("\n  ERROR: No price column found!")
        print(f"  Available columns: {', '.join(df.columns.tolist()[:20])}")
        return False

    # Clean price column
    print(f"\nStep 2: Cleaning price data")
    df['price_clean'] = df[price_col].apply(clean_price)

    # Show price distribution before filtering
    print(f"\n  Price distribution (before filtering):")
    print(f"    Records with price: {df['price_clean'].notna().sum()}")
    print(f"    Min: ${df['price_clean'].min():,.0f}" if df['price_clean'].min() else "    Min: N/A")
    print(f"    Max: ${df['price_clean'].max():,.0f}" if df['price_clean'].max() else "    Max: N/A")
    print(f"    Mean: ${df['price_clean'].mean():,.0f}" if df['price_clean'].mean() else "    Mean: N/A")
    print(f"    Median: ${df['price_clean'].median():,.0f}" if df['price_clean'].median() else "    Median: N/A")

    # Count potential rentals (< $50k)
    rentals = df[df['price_clean'] < 50000]
    print(f"\n  Potential rentals (price < $50k): {len(rentals)}")
    if len(rentals) > 0 and len(rentals) <= 10:
        print(f"    Sample prices: {rentals['price_clean'].dropna().head(10).tolist()}")

    # Step 3: Apply filters
    print(f"\nStep 3: Applying filters")

    original_count = len(df)

    # Filter 1: Remove rentals (< $50,000)
    df = df[df['price_clean'] >= 50000].copy()
    print(f"  ✓ Removed rentals (< $50k): {original_count - len(df)} records")
    original_count = len(df)

    # Filter 2: Remove extremely high prices (likely errors or commercial)
    df = df[df['price_clean'] <= 5000000].copy()
    print(f"  ✓ Removed extreme prices (> $5M): {original_count - len(df)} records")
    original_count = len(df)

    # Filter 3: Remove missing prices
    df = df[df['price_clean'].notna()].copy()
    print(f"  ✓ Removed missing prices: {original_count - len(df)} records")
    original_count = len(df)

    # Filter 4: Ensure we have city or zip
    has_location = df['City'].notna() | df['Zip'].notna()
    df = df[has_location].copy()
    print(f"  ✓ Removed missing location: {original_count - len(df)} records")

    # Filter 5: Remove duplicates (by address)
    if 'Address' in df.columns:
        original_count = len(df)
        df = df.drop_duplicates(subset=['Address'], keep='first')
        print(f"  ✓ Removed duplicates: {original_count - len(df)} records")

    # Step 4: Summary
    print(f"\nStep 4: Summary")
    print(f"  Final record count: {len(df)}")
    print(f"  Price range: ${df['price_clean'].min():,.0f} - ${df['price_clean'].max():,.0f}")
    print(f"  Average price: ${df['price_clean'].mean():,.0f}")
    print(f"  Median price: ${df['price_clean'].median():,.0f}")

    # Show city breakdown
    if 'City' in df.columns:
        print(f"\n  Properties by city:")
        city_counts = df['City'].value_counts().head(10)
        for city, count in city_counts.items():
            print(f"    {city}: {count} properties")

    # Step 5: Save cleaned data
    print(f"\nStep 5: Saving cleaned data")
    os.makedirs(os.path.dirname(output_file) if os.path.dirname(output_file) else '.', exist_ok=True)
    df.to_csv(output_file, index=False)
    print(f"  ✓ Saved to: {output_file}")

    print("\n" + "="*100)
    print("DATA CLEANING COMPLETE")
    print("="*100)
    print(f"\nNext step: Run flip analysis")
    print(f"  python scripts/analyze_flip_properties.py {output_file}")

    return True

def main():
    if len(sys.argv) < 2:
        print(__doc__)
        print("\nPlease provide the path to your Cobb County CSV file:")
        print("  python scripts/process_cobb_county.py <path_to_file.csv>")
        print("\nExample:")
        print("  python scripts/process_cobb_county.py Downloads/cobb_properties.csv")
        return

    input_file = sys.argv[1]

    if not os.path.exists(input_file):
        print(f"ERROR: File not found: {input_file}")
        return

    success = filter_cobb_data(input_file)

    if success:
        print("\n✓ Ready for analysis!")
        print("\nRun these commands next:")
        print("  1. python scripts/analyze_flip_properties.py data/cobb_county_properties.csv")
        print("  2. python scripts/analyze_by_area.py")
        print("  3. python scripts/merge_county_multiples.py")

if __name__ == "__main__":
    main()
