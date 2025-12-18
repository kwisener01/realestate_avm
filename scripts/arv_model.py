import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder
import re

class ARVEstimator:
    def __init__(self):
        self.model = None
        self.city_encoder = LabelEncoder()
        self.county_encoder = LabelEncoder()

    def clean_price(self, price_str):
        """Convert price string to numeric"""
        if pd.isna(price_str):
            return None
        cleaned = re.sub(r'[,$]', '', str(price_str))
        try:
            return float(cleaned)
        except:
            return None

    def extract_distance(self, dist_str):
        """Extract numeric distance from string like '50.8 mi'"""
        if pd.isna(dist_str):
            return None
        match = re.search(r'([\d.]+)', str(dist_str))
        if match:
            return float(match.group(1))
        return None

    def estimate_arv_multiplier(self, distance, days_on_market, city, county):
        """
        Estimate ARV multiplier based on location and market factors

        Logic:
        - Closer to Atlanta (lower distance) = higher ARV potential
        - Longer days on market = more distressed = higher ARV upside
        - Certain areas have better appreciation potential
        """
        # Base multiplier for fix-and-flip
        base_multiplier = 1.75

        # Distance factor: Closer to Atlanta = higher multiplier
        # 0-10 mi: +0.25, 10-20 mi: +0.15, 20-30 mi: +0.10, 30-40 mi: +0.05, 40+ mi: 0
        if distance <= 10:
            distance_bonus = 0.25
        elif distance <= 20:
            distance_bonus = 0.15
        elif distance <= 30:
            distance_bonus = 0.10
        elif distance <= 40:
            distance_bonus = 0.05
        else:
            distance_bonus = 0.0

        # Days on market factor: More days = more distressed = higher potential
        # 0-30 days: 0, 30-90 days: +0.05, 90-180 days: +0.10, 180+ days: +0.15
        if days_on_market >= 180:
            dom_bonus = 0.15
        elif days_on_market >= 90:
            dom_bonus = 0.10
        elif days_on_market >= 30:
            dom_bonus = 0.05
        else:
            dom_bonus = 0.0

        # High-value areas (based on typical Atlanta metro appreciation)
        high_value_cities = ['Atlanta', 'Decatur', 'Brookhaven', 'Sandy Springs',
                            'Alpharetta', 'Roswell', 'Marietta']
        moderate_value_cities = ['Lithonia', 'Riverdale', 'College Park', 'East Point',
                                'Forest Park', 'Smyrna', 'Dunwoody']

        city_bonus = 0.0
        if city in high_value_cities:
            city_bonus = 0.10
        elif city in moderate_value_cities:
            city_bonus = 0.05

        # Calculate final multiplier
        final_multiplier = base_multiplier + distance_bonus + dom_bonus + city_bonus

        # Cap between 1.5 and 2.3
        final_multiplier = max(1.5, min(2.3, final_multiplier))

        return final_multiplier

    def prepare_data(self, df):
        """Prepare data for model"""
        # Clean prices
        df['list_price_clean'] = df['List Price'].apply(self.clean_price)

        # Extract distance
        df['distance_clean'] = df['Distance'].apply(self.extract_distance)

        # Fill missing values
        df['distance_clean'] = df['distance_clean'].fillna(df['distance_clean'].median())
        df['Days On Market'] = df['Days On Market'].fillna(0)

        return df

    def calculate_arv(self, df):
        """Calculate ARV for all properties"""
        print("Calculating ARV for properties...")

        # Prepare data
        df = self.prepare_data(df)

        # Calculate ARV for each property
        arvs = []
        multipliers = []

        for idx, row in df.iterrows():
            distance = row['distance_clean']
            dom = row['Days On Market']
            city = row['City']
            county = row['County']
            list_price = row['list_price_clean']

            if pd.isna(list_price) or list_price == 0:
                arvs.append(None)
                multipliers.append(None)
                continue

            # Calculate multiplier
            multiplier = self.estimate_arv_multiplier(distance, dom, city, county)

            # Calculate ARV
            arv = list_price * multiplier

            arvs.append(int(arv))
            multipliers.append(round(multiplier, 2))

        df['ARV'] = arvs
        df['ARV_Multiplier'] = multipliers

        return df

    def determine_deal_status(self, df):
        """Determine if property is a deal (List Price <= 50% of ARV)"""
        print("Determining deal status...")

        deal_statuses = []

        for idx, row in df.iterrows():
            list_price = row['list_price_clean']
            arv = row['ARV']

            if pd.isna(list_price) or pd.isna(arv) or arv == 0:
                deal_statuses.append('Unknown')
                continue

            # Check if list price is <= 50% of ARV
            if list_price <= (arv * 0.5):
                deal_statuses.append('Deal')
            else:
                deal_statuses.append('No Deal')

        df['Deal Status'] = deal_statuses

        return df

def main():
    # Load data
    print("Loading data...")
    df = pd.read_csv('../data/listing_agents.csv')
    print(f"Loaded {len(df)} properties\n")

    # Initialize estimator
    estimator = ARVEstimator()

    # Calculate ARV
    df = estimator.calculate_arv(df)

    # Determine deal status
    df = estimator.determine_deal_status(df)

    # Print summary
    print("\n" + "=" * 70)
    print("RESULTS SUMMARY")
    print("=" * 70)

    # Overall statistics
    total_properties = len(df)
    properties_with_arv = df['ARV'].notna().sum()

    print(f"\nTotal Properties: {total_properties}")
    print(f"Properties with ARV: {properties_with_arv}")

    # Deal statistics
    deal_counts = df['Deal Status'].value_counts()
    print(f"\nDeal Status Breakdown:")
    for status, count in deal_counts.items():
        percentage = (count / total_properties) * 100
        print(f"  {status}: {count} ({percentage:.1f}%)")

    # ARV statistics
    print(f"\nARV Statistics:")
    print(f"  Average ARV: ${df['ARV'].mean():,.0f}")
    print(f"  Median ARV: ${df['ARV'].median():,.0f}")
    print(f"  Min ARV: ${df['ARV'].min():,.0f}")
    print(f"  Max ARV: ${df['ARV'].max():,.0f}")

    # Average multiplier
    print(f"\nAverage ARV Multiplier: {df['ARV_Multiplier'].mean():.2f}x")

    # Show some examples of deals
    deals = df[df['Deal Status'] == 'Deal'].head(10)
    if len(deals) > 0:
        print(f"\n{'-' * 70}")
        print("TOP 10 DEALS:")
        print(f"{'-' * 70}")
        for idx, row in deals.iterrows():
            print(f"\n{row['Address']}, {row['City']}")
            print(f"  List Price: ${row['list_price_clean']:,.0f}")
            print(f"  ARV: ${row['ARV']:,.0f}")
            print(f"  Multiplier: {row['ARV_Multiplier']:.2f}x")
            print(f"  Distance: {row['distance_clean']:.1f} mi")
            print(f"  Days on Market: {row['Days On Market']}")

    # Save results
    output_cols = ['Address', 'City', 'County', 'Zip Code', 'List Price',
                   'Deal Status', 'ARV', 'Distance', 'Days On Market',
                   'ARV_Multiplier', 'List Agent Full Name', 'List Office Name']

    df_output = df[output_cols].copy()

    # Save to CSV
    df_output.to_csv('../data/arv_results.csv', index=False)
    print(f"\n{'-' * 70}")
    print(f"Results saved to: data/arv_results.csv")

    # Also save full data with all columns
    df.to_csv('../data/arv_results_full.csv', index=False)
    print(f"Full results saved to: data/arv_results_full.csv")
    print(f"{'-' * 70}\n")

if __name__ == '__main__':
    main()
