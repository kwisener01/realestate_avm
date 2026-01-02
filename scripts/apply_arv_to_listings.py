import pandas as pd
import numpy as np

class ListingARVCalculator:
    """Apply area-specific ARV multiples to MLS listings"""

    def __init__(self):
        # Load area-specific multiples
        try:
            self.zip_multiples = pd.read_csv('data/arv_multiples_by_area.csv')
            self.city_multiples = pd.read_csv('data/arv_multiples_by_city.csv')
            print("Loaded area-specific ARV multiples")
        except FileNotFoundError:
            print("ERROR: Run 'python scripts/analyze_by_area.py' first to generate multiples")
            self.zip_multiples = None
            self.city_multiples = None

    def clean_price(self, price_str):
        """Convert price string to numeric"""
        if pd.isna(price_str):
            return None
        # Remove $, commas
        cleaned = str(price_str).replace('$', '').replace(',', '')
        try:
            return float(cleaned)
        except:
            return None

    def get_arv_multiple(self, zipcode, city):
        """Get ARV multiple for a specific zip/city"""
        # Try zip code first (more specific)
        if self.zip_multiples is not None:
            zip_match = self.zip_multiples[self.zip_multiples['Zip'] == str(zipcode)]
            if len(zip_match) > 0:
                return {
                    'conservative': zip_match.iloc[0]['Conservative_Multiple'],
                    'moderate': zip_match.iloc[0]['Moderate_Multiple'],
                    'aggressive': zip_match.iloc[0]['Aggressive_Multiple'],
                    'source': f'Zip {zipcode}',
                    'sample_size': zip_match.iloc[0]['Sample_Size']
                }

        # Fall back to city level
        if self.city_multiples is not None and pd.notna(city):
            city_match = self.city_multiples[self.city_multiples['City'].str.lower() == str(city).lower()]
            if len(city_match) > 0:
                return {
                    'conservative': city_match.iloc[0]['Conservative'],
                    'moderate': city_match.iloc[0]['Median'],
                    'aggressive': city_match.iloc[0]['Aggressive'],
                    'source': f'City {city}',
                    'sample_size': city_match.iloc[0]['Count']
                }

        # Default to overall average if no match
        return {
            'conservative': 2.59,
            'moderate': 2.82,
            'aggressive': 3.22,
            'source': 'Default (overall average)',
            'sample_size': 598
        }

    def calculate_arv_estimates(self, df, assessed_value_col='Total Assessed Value'):
        """
        Calculate ARV estimates for listings

        Parameters:
        - df: DataFrame with listing data
        - assessed_value_col: Name of column with assessed value (if available)
        """

        print("\n" + "="*100)
        print("CALCULATING ARV ESTIMATES FOR LISTINGS")
        print("="*100)

        # Clean the data
        df_clean = df.copy()

        # Clean zip codes (take first 5 digits)
        if 'Zip' in df_clean.columns:
            df_clean['Zip_Clean'] = df_clean['Zip'].astype(str).str[:5]
        else:
            print("WARNING: No Zip column found")
            df_clean['Zip_Clean'] = None

        # Clean list price - try multiple column names
        price_columns = ['List Price', 'Price', 'MLS Amount', 'Last Sale Amount', 'Sale Price']
        price_col_found = None

        for col in price_columns:
            if col in df_clean.columns:
                df_clean['List_Price_Clean'] = df_clean[col].apply(self.clean_price)
                price_col_found = col
                print(f"Using '{col}' as price column")
                break

        if price_col_found is None:
            print(f"ERROR: No price column found. Tried: {', '.join(price_columns)}")
            return df_clean

        # Get assessed value if available
        if assessed_value_col in df_clean.columns:
            df_clean['Assessed_Value_Clean'] = df_clean[assessed_value_col].apply(self.clean_price)
            has_assessed = True
        else:
            print(f"Note: No '{assessed_value_col}' column found. Will estimate from list price.")
            # Estimate assessed value as ~35% of list price (typical ratio)
            df_clean['Assessed_Value_Clean'] = df_clean['List_Price_Clean'] * 0.35
            has_assessed = False

        # Get multiples for each property
        results = []
        for idx, row in df_clean.iterrows():
            multiples = self.get_arv_multiple(row['Zip_Clean'], row.get('City', None))

            # Calculate ARV based on assessed value
            assessed = row['Assessed_Value_Clean']
            if pd.notna(assessed) and assessed > 0:
                arv_conservative = assessed * multiples['conservative']
                arv_moderate = assessed * multiples['moderate']
                arv_aggressive = assessed * multiples['aggressive']
            else:
                arv_conservative = None
                arv_moderate = None
                arv_aggressive = None

            results.append({
                'ARV_Conservative': arv_conservative,
                'ARV_Moderate': arv_moderate,
                'ARV_Aggressive': arv_aggressive,
                'ARV_Multiple_Source': multiples['source'],
                'ARV_Sample_Size': multiples['sample_size'],
                'Multiple_Conservative': multiples['conservative'],
                'Multiple_Moderate': multiples['moderate'],
                'Multiple_Aggressive': multiples['aggressive']
            })

        results_df = pd.DataFrame(results)
        df_output = pd.concat([df_clean, results_df], axis=1)

        # Calculate potential profit (ARV - List Price - estimated rehab)
        # Assume 15% of list price as average rehab cost
        df_output['Est_Rehab_Cost'] = df_output['List_Price_Clean'] * 0.15
        df_output['Potential_Profit_Conservative'] = df_output['ARV_Conservative'] - df_output['List_Price_Clean'] - df_output['Est_Rehab_Cost']
        df_output['Potential_Profit_Moderate'] = df_output['ARV_Moderate'] - df_output['List_Price_Clean'] - df_output['Est_Rehab_Cost']
        df_output['Potential_Profit_Aggressive'] = df_output['ARV_Aggressive'] - df_output['List_Price_Clean'] - df_output['Est_Rehab_Cost']

        # Calculate ROI
        df_output['ROI_Conservative'] = (df_output['Potential_Profit_Conservative'] / (df_output['List_Price_Clean'] + df_output['Est_Rehab_Cost']) * 100).round(1)
        df_output['ROI_Moderate'] = (df_output['Potential_Profit_Moderate'] / (df_output['List_Price_Clean'] + df_output['Est_Rehab_Cost']) * 100).round(1)
        df_output['ROI_Aggressive'] = (df_output['Potential_Profit_Aggressive'] / (df_output['List_Price_Clean'] + df_output['Est_Rehab_Cost']) * 100).round(1)

        # Flag good deals (moderate ROI > 20%)
        df_output['Is_Good_Deal'] = df_output['ROI_Moderate'] > 20

        print(f"\nProcessed {len(df_output)} listings")
        print(f"Good deals found (ROI > 20%): {df_output['Is_Good_Deal'].sum()}")

        if not has_assessed:
            print("\nWARNING: Assessed values were estimated at 35% of list price.")
            print("For more accurate results, include actual assessed values in your data.")

        return df_output

    def print_summary(self, df_results):
        """Print summary of ARV calculations"""

        print("\n" + "="*100)
        print("ARV ESTIMATION SUMMARY")
        print("="*100)

        # Check if Is_Good_Deal column exists
        if 'Is_Good_Deal' not in df_results.columns:
            print("\nNo deal analysis available (missing required columns)")
            return

        good_deals = df_results[df_results['Is_Good_Deal'] == True].copy()

        if len(good_deals) > 0:
            print(f"\nTop 10 Deals by Moderate ROI:")
            print("-"*100)

            cols = ['City', 'Zip_Clean', 'List_Price_Clean', 'ARV_Moderate',
                   'Potential_Profit_Moderate', 'ROI_Moderate', 'Days On Market']

            display_cols = cols if 'Days On Market' in good_deals.columns else cols[:-1]

            top_deals = good_deals.nlargest(10, 'ROI_Moderate')[display_cols]
            print(top_deals.to_string(index=False))

            print(f"\n" + "="*100)
            print("DEAL STATISTICS")
            print("="*100)
            print(f"\nAverage ARV (Moderate): ${good_deals['ARV_Moderate'].mean():,.0f}")
            print(f"Average Potential Profit: ${good_deals['Potential_Profit_Moderate'].mean():,.0f}")
            print(f"Average ROI: {good_deals['ROI_Moderate'].mean():.1f}%")

        else:
            print("\nNo deals found with ROI > 20%")
            print("Try adjusting criteria or check if assessed values are accurate")

    def save_results(self, df_results, output_file='data/listings_with_arv.csv'):
        """Save results to CSV"""
        df_results.to_csv(output_file, index=False)
        print(f"\nSaved results to {output_file}")

def main():
    # Example usage
    calculator = ListingARVCalculator()

    # Try to load listings data
    # You can modify this to load from your specific file
    print("\nLooking for listings data...")

    # Check for common file names
    possible_files = [
        'data/listings.csv',
        'data/mls_listings.csv',
        'data/properties.csv',
        'Property Export Decatur+List.xlsx'
    ]

    df = None
    for file in possible_files:
        try:
            if file.endswith('.xlsx'):
                df = pd.read_excel(file)
            else:
                df = pd.read_csv(file)
            print(f"Loaded data from {file}")
            break
        except FileNotFoundError:
            continue

    if df is None:
        print("\nNo listings data found. Please provide a CSV/Excel file with listings.")
        print("\nExpected columns:")
        print("  Required: City, Zip, List Price (or Price)")
        print("  Optional: Total Assessed Value, Days On Market")
        return

    # Calculate ARV estimates
    df_results = calculator.calculate_arv_estimates(df)

    # Print summary
    calculator.print_summary(df_results)

    # Save results
    calculator.save_results(df_results)

if __name__ == "__main__":
    main()
