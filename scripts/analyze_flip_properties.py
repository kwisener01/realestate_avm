import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import re

class FlipPropertyAnalyzer:
    """Analyze properties for flip potential and calculate ARV multiples"""

    def __init__(self, data_file='data/decatur_properties.csv'):
        self.data_file = data_file
        self.df = None
        self.county_name = self._extract_county_name(data_file)

    def _extract_county_name(self, file_path):
        """Extract county name from file path for output naming"""
        import os
        filename = os.path.basename(file_path)
        # Try to extract county name from filename
        if 'decatur' in filename.lower():
            return 'dekalb'
        elif 'cobb' in filename.lower():
            return 'cobb'
        elif 'fulton' in filename.lower():
            return 'fulton'
        else:
            return 'unknown'

    def load_data(self):
        """Load property data"""
        print(f"Loading data from {self.data_file}...")
        # Detect file type and use appropriate reader
        if self.data_file.endswith('.xlsx') or self.data_file.endswith('.xls'):
            self.df = pd.read_excel(self.data_file)
        else:
            self.df = pd.read_csv(self.data_file)
        print(f"Loaded {len(self.df)} properties\n")
        return self.df

    def parse_date(self, date_str):
        """Parse date string to datetime"""
        if pd.isna(date_str):
            return None
        try:
            return pd.to_datetime(date_str)
        except:
            return None

    def identify_potential_flips(self):
        """
        Identify potential flip properties based on:
        1. Recent sales (Last Sale Date within last 2 years)
        2. Properties that sold quickly or at significant markup
        3. Properties with MLS status indicating recent activity
        """
        print("=" * 80)
        print("ANALYZING POTENTIAL FLIP PROPERTIES")
        print("=" * 80)

        df = self.df.copy()

        # Parse dates
        df['last_sale_date'] = df['Last Sale Recording Date'].apply(self.parse_date)
        df['mls_date'] = df['MLS Date'].apply(self.parse_date)

        # Calculate time on market if both dates available
        df['time_on_market_days'] = (df['last_sale_date'] - df['mls_date']).dt.days

        # Clean numeric columns
        df['last_sale_price'] = pd.to_numeric(df['Last Sale Amount'], errors='coerce')
        df['mls_price'] = pd.to_numeric(df['MLS Amount'], errors='coerce')
        df['assessed_value'] = pd.to_numeric(df['Total Assessed Value_clean'], errors='coerce')
        df['est_value'] = pd.to_numeric(df['Est. Value_clean'], errors='coerce')

        # Filter for properties with complete sale data
        flip_candidates = df[
            (df['last_sale_date'].notna()) &
            (df['last_sale_price'].notna()) &
            (df['last_sale_price'] > 0)
        ].copy()

        print(f"\nProperties with sale data: {len(flip_candidates)}")

        # Calculate potential flip metrics
        flip_candidates['sale_to_assessed_ratio'] = flip_candidates['last_sale_price'] / flip_candidates['assessed_value']
        flip_candidates['sale_to_estimate_ratio'] = flip_candidates['last_sale_price'] / flip_candidates['est_value']

        # Identify likely flips (sold significantly above assessed value in recent times)
        current_year = datetime.now().year
        flip_candidates['years_since_sale'] = current_year - flip_candidates['last_sale_date'].dt.year

        # Criteria for potential flips:
        # 1. Sold within last 3 years
        # 2. Sold for significantly more than assessed value (20%+ markup)
        # 3. Quick sale if MLS data available

        recent_flips = flip_candidates[
            (flip_candidates['years_since_sale'] <= 3) &
            (flip_candidates['sale_to_assessed_ratio'] > 1.2)
        ].copy()

        print(f"Potential flips (sold within 3 years at 20%+ above assessed): {len(recent_flips)}")

        if len(recent_flips) > 0:
            print("\n" + "=" * 80)
            print("FLIP PROPERTY STATISTICS")
            print("=" * 80)

            print(f"\nSale to Assessed Value Ratios:")
            print(f"  Mean:   {recent_flips['sale_to_assessed_ratio'].mean():.2f}x")
            print(f"  Median: {recent_flips['sale_to_assessed_ratio'].median():.2f}x")
            print(f"  Min:    {recent_flips['sale_to_assessed_ratio'].min():.2f}x")
            print(f"  Max:    {recent_flips['sale_to_assessed_ratio'].max():.2f}x")

            # Calculate ARV estimation multiples
            print(f"\nARV ESTIMATION MULTIPLES (Last Sale Price / Assessed Value):")
            print(f"  Conservative (25th percentile): {recent_flips['sale_to_assessed_ratio'].quantile(0.25):.2f}x")
            print(f"  Moderate (50th percentile):     {recent_flips['sale_to_assessed_ratio'].quantile(0.50):.2f}x")
            print(f"  Aggressive (75th percentile):   {recent_flips['sale_to_assessed_ratio'].quantile(0.75):.2f}x")

            # Group by property type if available
            if 'Property Type' in recent_flips.columns:
                print(f"\n" + "=" * 80)
                print("ARV MULTIPLES BY PROPERTY TYPE")
                print("=" * 80)

                property_type_stats = recent_flips.groupby('Property Type').agg({
                    'sale_to_assessed_ratio': ['count', 'mean', 'median', 'min', 'max']
                }).round(2)

                print(property_type_stats)

            # Analyze by city/location
            if 'City' in recent_flips.columns:
                print(f"\n" + "=" * 80)
                print("ARV MULTIPLES BY CITY")
                print("=" * 80)

                city_stats = recent_flips.groupby('City').agg({
                    'sale_to_assessed_ratio': ['count', 'mean', 'median']
                }).round(2)

                city_stats = city_stats[city_stats[('sale_to_assessed_ratio', 'count')] >= 3]
                print(city_stats)

        # Save results
        output_file = 'data/flip_analysis_results.csv'
        recent_flips.to_csv(output_file, index=False)
        print(f"\nSaved detailed results to {output_file}")

        return recent_flips

    def analyze_mls_flip_activity(self):
        """Analyze properties based on MLS activity for quick flips"""
        print("\n" + "=" * 80)
        print("ANALYZING MLS ACTIVITY FOR FLIP PATTERNS")
        print("=" * 80)

        df = self.df.copy()

        # Parse dates
        df['last_sale_date'] = df['Last Sale Recording Date'].apply(self.parse_date)
        df['mls_date'] = df['MLS Date'].apply(self.parse_date)

        # Find properties with both purchase and listing dates
        mls_flips = df[
            (df['last_sale_date'].notna()) &
            (df['mls_date'].notna()) &
            (df['MLS Status'] == 'SOLD')
        ].copy()

        # Calculate time between purchase and listing
        # Note: If MLS date is AFTER sale date, property was bought then listed
        # If sale is after MLS, property was listed then sold

        # Clean prices
        mls_flips['last_sale_price'] = pd.to_numeric(mls_flips['Last Sale Amount'], errors='coerce')
        mls_flips['mls_price'] = pd.to_numeric(mls_flips['MLS Amount'], errors='coerce')

        mls_flips = mls_flips[
            (mls_flips['last_sale_price'] > 0) &
            (mls_flips['mls_price'] > 0)
        ].copy()

        if len(mls_flips) > 0:
            # Calculate profit if both prices available
            mls_flips['potential_profit'] = mls_flips['mls_price'] - mls_flips['last_sale_price']
            mls_flips['profit_margin'] = (mls_flips['potential_profit'] / mls_flips['last_sale_price'] * 100)

            print(f"\nProperties with MLS and sale data: {len(mls_flips)}")

            # Filter for actual flips (positive profit)
            actual_flips = mls_flips[mls_flips['potential_profit'] > 0].copy()

            if len(actual_flips) > 0:
                print(f"\nProperties sold at profit: {len(actual_flips)}")
                print(f"\nProfit Statistics:")
                print(f"  Mean profit:   ${actual_flips['potential_profit'].mean():,.0f}")
                print(f"  Median profit: ${actual_flips['potential_profit'].median():,.0f}")
                print(f"  Mean margin:   {actual_flips['profit_margin'].mean():.1f}%")
                print(f"  Median margin: {actual_flips['profit_margin'].median():.1f}%")

                print(f"\nARV Multiples (MLS Price / Purchase Price):")
                actual_flips['arv_multiple'] = actual_flips['mls_price'] / actual_flips['last_sale_price']
                print(f"  Conservative (25th): {actual_flips['arv_multiple'].quantile(0.25):.2f}x")
                print(f"  Moderate (50th):     {actual_flips['arv_multiple'].quantile(0.50):.2f}x")
                print(f"  Aggressive (75th):   {actual_flips['arv_multiple'].quantile(0.75):.2f}x")

                # Save MLS flip results
                output_file = 'data/mls_flip_analysis.csv'
                actual_flips.to_csv(output_file, index=False)
                print(f"\nSaved MLS flip analysis to {output_file}")

                return actual_flips

        print("\nNo clear flip patterns found in MLS data")
        return pd.DataFrame()

    def generate_report(self):
        """Generate comprehensive flip analysis report"""
        print("\n" + "=" * 80)
        print("REAL ESTATE FLIP ANALYSIS REPORT")
        print("=" * 80)

        self.load_data()

        # Run analyses
        flip_properties = self.identify_potential_flips()
        mls_flips = self.analyze_mls_flip_activity()

        print("\n" + "=" * 80)
        print("RECOMMENDED ARV ESTIMATION STRATEGY")
        print("=" * 80)

        if len(flip_properties) > 0:
            median_multiple = flip_properties['sale_to_assessed_ratio'].median()
            mean_multiple = flip_properties['sale_to_assessed_ratio'].mean()

            print(f"\nBased on {len(flip_properties)} analyzed properties:")
            print(f"\n1. Use {median_multiple:.2f}x assessed value for moderate ARV estimates")
            print(f"2. Use {mean_multiple:.2f}x assessed value for average ARV estimates")
            print(f"3. Adjust based on property condition and location")
            print(f"\nFor conservative estimates, use the assessed value plus 20-30%")
            print(f"For aggressive estimates in hot markets, use up to 2.0-2.5x assessed value")

        print("\n" + "=" * 80)
        print("Analysis complete!")
        print("=" * 80)

def main():
    import sys

    # Check for command-line argument
    if len(sys.argv) > 1:
        data_file = sys.argv[1]
        print(f"Using custom data file: {data_file}")
    else:
        data_file = 'data/decatur_properties.csv'
        print(f"Using default data file: {data_file}")

    analyzer = FlipPropertyAnalyzer(data_file=data_file)
    analyzer.generate_report()

if __name__ == "__main__":
    main()
