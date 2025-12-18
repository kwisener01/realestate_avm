"""
Prepare training data from raw property CSV
Converts decatur_properties.csv to format needed by ML models
"""

import pandas as pd
import numpy as np
from datetime import datetime


def prepare_data(input_path='data/decatur_properties.csv',
                 output_path='data/processed/training_data.csv'):
    """
    Prepare training data from raw property data

    Args:
        input_path: Path to raw CSV file
        output_path: Path to save processed data
    """
    print(f"Loading data from {input_path}...")
    df = pd.read_csv(input_path)

    print(f"Raw data shape: {df.shape}")
    print(f"Raw columns: {df.columns.tolist()[:10]}...")

    # Create processed dataframe with required columns
    processed = pd.DataFrame()

    # Map columns from raw data to training format
    column_mapping = {
        'Bedrooms': 'bedrooms',
        'Total Bathrooms': 'bathrooms',
        'Building Sqft': 'sqft_living',
        'Lot Size Sqft': 'sqft_lot',
        'Effective Year Built': 'year_built',
        'Property Type': 'property_type',
        'Total Condition': 'condition',
        'Last Sale Amount': 'price',  # Target variable
        'Last Sale Recording Date': 'date'
    }

    # Copy and rename columns
    for raw_col, new_col in column_mapping.items():
        if raw_col in df.columns:
            processed[new_col] = df[raw_col]
        else:
            print(f"Warning: Column '{raw_col}' not found in data")

    # Add derived features
    print("\nCreating derived features...")

    # Floors (estimate from building type and sqft)
    # Assume 1 floor for small homes, 2+ for larger
    processed['floors'] = processed['sqft_living'].apply(
        lambda x: 1.0 if pd.isna(x) or x < 1500 else 2.0 if x < 3000 else 3.0
    )

    # Year renovated (default to 0 if not renovated)
    processed['year_renovated'] = 0

    # Latitude and Longitude (use DeKalb County, GA center as default)
    # In production, you'd geocode addresses
    processed['latitude'] = 33.7490  # DeKalb County center
    processed['longitude'] = -84.2809

    # Add some random variation for more realistic training
    processed['latitude'] += np.random.uniform(-0.1, 0.1, len(processed))
    processed['longitude'] += np.random.uniform(-0.1, 0.1, len(processed))

    # Neighborhood (use city or default)
    if 'City' in df.columns:
        processed['neighborhood'] = df['City']
    else:
        processed['neighborhood'] = 'DeKalb'

    # View quality (default based on property value)
    # Higher value properties tend to have better views
    if 'Est. Value_clean' in df.columns:
        processed['view_quality'] = pd.cut(
            df['Est. Value_clean'],
            bins=[0, 200000, 400000, 600000, float('inf')],
            labels=['None', 'Fair', 'Good', 'Excellent']
        ).astype(str)
    else:
        processed['view_quality'] = 'Fair'

    # Add ID column
    processed['id'] = range(len(processed))

    # Clean data
    print("\nCleaning data...")

    # Remove rows with missing target (price)
    initial_rows = len(processed)
    processed = processed.dropna(subset=['price'])
    print(f"Removed {initial_rows - len(processed)} rows with missing price")

    # Remove rows with price = 0
    processed = processed[processed['price'] > 0]
    print(f"Removed rows with zero price, remaining: {len(processed)}")

    # Fill missing values for numeric features
    numeric_cols = ['bedrooms', 'bathrooms', 'sqft_living', 'sqft_lot',
                    'year_built', 'floors']

    for col in numeric_cols:
        if col in processed.columns:
            # Fill with median
            median_val = processed[col].median()
            processed[col].fillna(median_val, inplace=True)
            print(f"Filled {col} missing values with median: {median_val}")

    # Convert data types
    if 'bedrooms' in processed.columns:
        processed['bedrooms'] = processed['bedrooms'].astype(int)
    if 'bathrooms' in processed.columns:
        processed['bathrooms'] = processed['bathrooms'].astype(float)
    if 'sqft_living' in processed.columns:
        processed['sqft_living'] = processed['sqft_living'].astype(int)
    if 'sqft_lot' in processed.columns:
        processed['sqft_lot'] = processed['sqft_lot'].astype(int)
    if 'year_built' in processed.columns:
        processed['year_built'] = processed['year_built'].astype(int)
    if 'year_renovated' in processed.columns:
        processed['year_renovated'] = processed['year_renovated'].astype(int)

    # Standardize categorical values
    if 'property_type' in processed.columns:
        # Map to standard types
        type_mapping = {
            'Single Family Residential': 'Single Family',
            'Townhouse': 'Townhouse',
            'Condo': 'Condo',
            'Multi-Family': 'Multi-Family'
        }
        processed['property_type'] = processed['property_type'].map(
            lambda x: type_mapping.get(x, 'Single Family')
        )

    if 'condition' in processed.columns:
        # Map to standard conditions
        condition_mapping = {
            'Poor': 'Poor',
            'Fair': 'Fair',
            'Average': 'Average',
            'Good': 'Good',
            'Excellent': 'Excellent'
        }
        processed['condition'] = processed['condition'].fillna('Average')
        processed['condition'] = processed['condition'].map(
            lambda x: condition_mapping.get(x, 'Average')
        )

    # Reorder columns to match API format
    column_order = [
        'id', 'bedrooms', 'bathrooms', 'sqft_living', 'sqft_lot',
        'floors', 'year_built', 'year_renovated', 'latitude', 'longitude',
        'property_type', 'neighborhood', 'condition', 'view_quality',
        'date', 'price'
    ]

    # Keep only columns that exist
    existing_cols = [col for col in column_order if col in processed.columns]
    processed = processed[existing_cols]

    # Print summary statistics
    print("\n" + "="*60)
    print("PROCESSED DATA SUMMARY")
    print("="*60)
    print(f"Total properties: {len(processed)}")
    print(f"\nColumns: {processed.columns.tolist()}")
    print(f"\nData types:\n{processed.dtypes}")
    print(f"\nPrice statistics:")
    print(processed['price'].describe())
    print(f"\nMissing values:")
    print(processed.isnull().sum())

    # Save processed data
    import os
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    processed.to_csv(output_path, index=False)
    print(f"\n✅ Processed data saved to: {output_path}")
    print(f"Ready for training!")

    return processed


if __name__ == '__main__':
    # Prepare training data
    df = prepare_data()

    # Show sample
    print("\n" + "="*60)
    print("SAMPLE DATA (first 3 rows)")
    print("="*60)
    print(df.head(3).to_string())
