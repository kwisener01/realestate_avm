"""
Process MLS listings and calculate ARV estimates with area-specific multiples

Input format expected:
- Street Number, Street Name, City, Zip
- List Price
- Days On Market
- MLS #, Parcel Number, etc.
"""

import pandas as pd
import sys

# Import the ARV calculator
from apply_arv_to_listings import ListingARVCalculator

def process_mls_file(input_file, output_file=None):
    """
    Process MLS listings file and add ARV estimates

    Args:
        input_file: Path to CSV or Excel file with MLS listings
        output_file: Path to save results (optional, defaults to input_file_with_arv.csv)
    """

    print("="*100)
    print("MLS LISTING ARV CALCULATOR")
    print("="*100)

    # Load the data
    print(f"\nLoading data from {input_file}...")

    if input_file.endswith('.xlsx') or input_file.endswith('.xls'):
        df = pd.read_excel(input_file)
    else:
        df = pd.read_csv(input_file)

    print(f"Loaded {len(df)} listings")
    print(f"\nColumns found: {', '.join(df.columns.tolist())}")

    # Initialize calculator
    calculator = ListingARVCalculator()

    # Check for assessed value column
    assessed_col = None
    for col in ['Total Assessed Value', 'Assessed Value', 'Tax Assessed Value', 'Assessment']:
        if col in df.columns:
            assessed_col = col
            break

    if assessed_col:
        print(f"Using '{assessed_col}' for ARV calculation")
    else:
        print("No assessed value column found - will estimate from list price")

    # Calculate ARV estimates
    df_results = calculator.calculate_arv_estimates(df, assessed_value_col=assessed_col or 'Total Assessed Value')

    # Create output filename if not specified
    if output_file is None:
        if input_file.endswith('.csv'):
            output_file = input_file.replace('.csv', '_with_arv.csv')
        else:
            output_file = 'data/mls_listings_with_arv.csv'

    # Print summary
    calculator.print_summary(df_results)

    # Save results
    calculator.save_results(df_results, output_file)

    # Print column guide
    print("\n" + "="*100)
    print("OUTPUT COLUMNS GUIDE")
    print("="*100)
    print("""
ARV Estimates (3 scenarios):
  - ARV_Conservative: Lower-end estimate (safer)
  - ARV_Moderate: Middle estimate (recommended)
  - ARV_Aggressive: Higher-end estimate (optimistic)

Multiples Used:
  - Multiple_Conservative: Multiple applied for conservative estimate
  - Multiple_Moderate: Multiple applied for moderate estimate
  - Multiple_Aggressive: Multiple applied for aggressive estimate
  - ARV_Multiple_Source: Where the multiple came from (Zip code or City)
  - ARV_Sample_Size: Number of properties used to calculate the multiple

Profit Analysis:
  - Est_Rehab_Cost: Estimated rehab ($45/sqft, or 15% of price if sqft unavailable)
  - Potential_Profit_[Conservative/Moderate/Aggressive]: ARV - Purchase - Rehab
  - ROI_[Conservative/Moderate/Aggressive]: Return on investment %
  - Is_Good_Deal: True if moderate ROI > 20%

How to use:
  1. Sort by ROI_Moderate (highest first) to find best deals
  2. Filter Is_Good_Deal = True to see only profitable properties
  3. Use ARV_Conservative for your worst-case scenario analysis
  4. Check ARV_Sample_Size - higher is more reliable
    """)

    return df_results

def main():
    if len(sys.argv) < 2:
        print("Usage: python process_mls_listings.py <input_file> [output_file]")
        print("\nExample:")
        print("  python process_mls_listings.py data/listings.csv")
        print("  python process_mls_listings.py 'Property Export Decatur+List.xlsx' data/analyzed_deals.csv")

        # Try to find a file automatically
        print("\nLooking for listing files...")
        import os
        common_files = [
            'Property Export Decatur+List.xlsx',
            'data/listings.csv',
            'data/mls_listings.csv'
        ]

        for file in common_files:
            if os.path.exists(file):
                print(f"\nFound: {file}")
                print("Processing automatically...")
                process_mls_file(file)
                return

        print("\nNo listing files found. Please specify the input file.")
        return

    input_file = sys.argv[1]
    output_file = sys.argv[2] if len(sys.argv) > 2 else None

    process_mls_file(input_file, output_file)

if __name__ == "__main__":
    main()
