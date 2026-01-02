import pandas as pd
import numpy as np
from datetime import datetime, timedelta

class CobbSoldAnalyzer:
    """Analyze Cobb County sold properties for flip multiples"""

    def __init__(self, data_file):
        self.data_file = data_file
        self.df = None

    def load_data(self):
        """Load property data"""
        print(f"Loading data from {self.data_file}...")
        self.df = pd.read_excel(self.data_file)
        print(f"Loaded {len(self.df)} properties\n")
        return self.df

    def clean_and_filter(self):
        """Clean data and filter out invalid properties"""
        print("Cleaning and filtering data...")
        df = self.df.copy()

        initial_count = len(df)

        # Remove rentals and low-value properties
        df = df[pd.to_numeric(df['MLS Amount'], errors='coerce') >= 50000]
        rentals_removed = initial_count - len(df)

        # Remove very high-value outliers
        df = df[pd.to_numeric(df['MLS Amount'], errors='coerce') <= 5000000]

        # Remove properties without assessed value
        df = df[pd.to_numeric(df['Total Assessed Value'], errors='coerce') > 0]

        # Convert date columns
        df['MLS Date'] = pd.to_datetime(df['MLS Date'], errors='coerce')
        df['Last Sale Recording Date'] = pd.to_datetime(df['Last Sale Recording Date'], errors='coerce')

        # Remove properties without valid dates
        df = df[df['MLS Date'].notna()]

        print(f"Removed {rentals_removed} low-value properties (< $50k)")
        print(f"Kept {len(df)} properties for analysis\n")

        self.df = df
        return df

    def identify_flips(self):
        """Identify properties that were flipped (bought and resold within timeframe)"""
        print("Identifying flip properties...")
        df = self.df.copy()

        # Filter for properties with both purchase and sale dates
        flips = df[df['Last Sale Recording Date'].notna()].copy()

        # Calculate time between purchase and resale
        flips['days_held'] = (flips['MLS Date'] - flips['Last Sale Recording Date']).dt.days

        # Focus on properties held 30 days to 3 years (typical flip timeframe)
        flips = flips[(flips['days_held'] >= 30) & (flips['days_held'] <= 1095)]

        # Calculate ARV multiple (Sale Price / Assessed Value)
        flips['assessed_value'] = pd.to_numeric(flips['Total Assessed Value'], errors='coerce')
        flips['sale_price'] = pd.to_numeric(flips['MLS Amount'], errors='coerce')
        flips['last_purchase_price'] = pd.to_numeric(flips['Last Sale Amount'], errors='coerce')

        flips['arv_multiple'] = flips['sale_price'] / flips['assessed_value']

        # Filter for reasonable multiples (0.5x to 10x)
        flips = flips[(flips['arv_multiple'] >= 0.5) & (flips['arv_multiple'] <= 10)]

        # Remove duplicates by address
        flips = flips.drop_duplicates(subset=['Address', 'Zip'])

        print(f"Found {len(flips)} valid flip properties")
        print(f"ARV multiples range: {flips['arv_multiple'].min():.2f}x to {flips['arv_multiple'].max():.2f}x")
        print(f"Median ARV multiple: {flips['arv_multiple'].median():.2f}x\n")

        return flips

    def calculate_multiples_by_zip(self, flips):
        """Calculate ARV multiples by zip code"""
        print("Calculating multiples by zip code...")

        # Group by zip code
        zip_stats = flips.groupby('Zip').agg({
            'arv_multiple': ['count', 'mean', 'median', 'std', 'min', 'max'],
            'sale_price': 'median',
            'City': 'first'
        }).reset_index()

        zip_stats.columns = ['Zip', 'Sample_Size', 'Mean_Multiple', 'Median_Multiple',
                             'StdDev', 'Min_Multiple', 'Max_Multiple', 'Median_Price', 'City']

        # Filter for minimum sample size
        zip_stats = zip_stats[zip_stats['Sample_Size'] >= 5]

        # Calculate conservative (25th percentile) and aggressive (75th percentile)
        for idx, row in zip_stats.iterrows():
            zip_code = row['Zip']
            zip_multiples = flips[flips['Zip'] == zip_code]['arv_multiple']
            zip_stats.loc[idx, 'Conservative_Multiple'] = zip_multiples.quantile(0.25)
            zip_stats.loc[idx, 'Moderate_Multiple'] = zip_multiples.median()
            zip_stats.loc[idx, 'Aggressive_Multiple'] = zip_multiples.quantile(0.75)

        # Add county identifier
        zip_stats['County'] = 'Cobb'

        # Sort by moderate multiple (descending)
        zip_stats = zip_stats.sort_values('Moderate_Multiple', ascending=False)

        print(f"Generated multiples for {len(zip_stats)} zip codes")
        print(f"\nTop 5 zip codes by median multiple:")
        for _, row in zip_stats.head(5).iterrows():
            print(f"  {row['Zip']} ({row['City']}): {row['Moderate_Multiple']:.2f}x ({int(row['Sample_Size'])} properties)")

        return zip_stats

    def calculate_multiples_by_city(self, flips):
        """Calculate ARV multiples by city"""
        print("\nCalculating multiples by city...")

        # Group by city
        city_stats = flips.groupby('City').agg({
            'arv_multiple': ['count', 'mean', 'median', 'std', 'min', 'max'],
            'sale_price': 'median'
        }).reset_index()

        city_stats.columns = ['City', 'Sample_Size', 'Mean_Multiple', 'Median_Multiple',
                              'StdDev', 'Min_Multiple', 'Max_Multiple', 'Median_Price']

        # Filter for minimum sample size
        city_stats = city_stats[city_stats['Sample_Size'] >= 5]

        # Calculate conservative and aggressive
        for idx, row in city_stats.iterrows():
            city = row['City']
            city_multiples = flips[flips['City'] == city]['arv_multiple']
            city_stats.loc[idx, 'Conservative_Multiple'] = city_multiples.quantile(0.25)
            city_stats.loc[idx, 'Aggressive_Multiple'] = city_multiples.quantile(0.75)

        # Add county identifier
        city_stats['County'] = 'Cobb'

        # Sort by median multiple (descending)
        city_stats = city_stats.sort_values('Median_Multiple', ascending=False)

        print(f"Generated multiples for {len(city_stats)} cities")

        return city_stats

    def generate_report(self):
        """Generate full analysis report"""
        self.load_data()
        self.clean_and_filter()
        flips = self.identify_flips()

        zip_multiples = self.calculate_multiples_by_zip(flips)
        city_multiples = self.calculate_multiples_by_city(flips)

        # Save outputs
        zip_multiples.to_csv('data/cobb_sold_multiples_by_area.csv', index=False)
        city_multiples.to_csv('data/cobb_sold_multiples_by_city.csv', index=False)

        print(f"\nSaved to:")
        print(f"  - data/cobb_sold_multiples_by_area.csv ({len(zip_multiples)} zip codes)")
        print(f"  - data/cobb_sold_multiples_by_city.csv ({len(city_multiples)} cities)")

        return zip_multiples, city_multiples

if __name__ == "__main__":
    analyzer = CobbSoldAnalyzer('Property Export Cobb+County+Sold.xlsx')
    analyzer.generate_report()
