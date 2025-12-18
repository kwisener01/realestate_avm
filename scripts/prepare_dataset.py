"""
Data preparation script
Cleans and preprocesses raw property data for model training
"""

import pandas as pd
import numpy as np
import argparse
import os
from pathlib import Path


def clean_property_data(df: pd.DataFrame) -> pd.DataFrame:
    """Clean and preprocess property data"""

    print("Initial dataset shape:", df.shape)

    # Remove duplicates
    df = df.drop_duplicates(subset=['id'], keep='first')
    print(f"After removing duplicates: {df.shape}")

    # Handle missing values
    # For numeric columns, fill with median
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        if df[col].isnull().sum() > 0:
            df[col].fillna(df[col].median(), inplace=True)

    # For categorical columns, fill with mode or 'unknown'
    categorical_cols = df.select_dtypes(include=['object']).columns
    for col in categorical_cols:
        if col not in ['id', 'date', 'description', 'image_path']:
            if df[col].isnull().sum() > 0:
                df[col].fillna(df[col].mode()[0] if not df[col].mode().empty else 'unknown', inplace=True)

    # Remove outliers for price (outside 1st and 99th percentile)
    if 'price' in df.columns:
        q1 = df['price'].quantile(0.01)
        q99 = df['price'].quantile(0.99)
        df = df[(df['price'] >= q1) & (df['price'] <= q99)]
        print(f"After removing price outliers: {df.shape}")

    # Remove properties with invalid data
    if 'bedrooms' in df.columns:
        df = df[df['bedrooms'] > 0]
    if 'bathrooms' in df.columns:
        df = df[df['bathrooms'] > 0]
    if 'sqft_living' in df.columns:
        df = df[df['sqft_living'] > 0]

    print(f"Final cleaned dataset shape: {df.shape}")

    return df


def create_sample_dataset(output_path: str, n_samples: int = 1000):
    """
    Create a sample dataset for demonstration purposes
    In production, replace this with actual data loading
    """
    print(f"Creating sample dataset with {n_samples} samples...")

    np.random.seed(42)

    data = {
        'id': [f'PROP_{i:06d}' for i in range(n_samples)],
        'date': pd.date_range(start='2020-01-01', periods=n_samples, freq='D'),

        # Numeric features
        'bedrooms': np.random.randint(1, 6, n_samples),
        'bathrooms': np.random.uniform(1, 4, n_samples).round(1),
        'sqft_living': np.random.randint(800, 5000, n_samples),
        'sqft_lot': np.random.randint(2000, 15000, n_samples),
        'floors': np.random.choice([1, 1.5, 2, 2.5, 3], n_samples),
        'year_built': np.random.randint(1950, 2023, n_samples),
        'year_renovated': np.random.choice([0] * 700 + list(range(2000, 2023)), n_samples),

        # Location features
        'latitude': np.random.uniform(47.0, 47.8, n_samples),
        'longitude': np.random.uniform(-122.5, -121.5, n_samples),

        # Categorical features
        'property_type': np.random.choice(['Single Family', 'Townhouse', 'Condo', 'Multi-Family'], n_samples),
        'neighborhood': np.random.choice(['Downtown', 'Suburbs', 'Waterfront', 'Rural', 'Urban'], n_samples),
        'condition': np.random.choice(['Poor', 'Fair', 'Average', 'Good', 'Excellent'], n_samples),
        'view_quality': np.random.choice(['None', 'Fair', 'Good', 'Excellent'], n_samples),

        # Text description
        'description': [
            f"Beautiful {np.random.choice(['modern', 'classic', 'updated', 'charming'])} home with "
            f"{np.random.randint(1, 6)} bedrooms and {np.random.uniform(1, 4):.1f} bathrooms. "
            f"Features {np.random.choice(['hardwood floors', 'granite countertops', 'stainless appliances', 'open floor plan'])}. "
            f"Located in {np.random.choice(['quiet', 'vibrant', 'family-friendly', 'convenient'])} neighborhood."
            for _ in range(n_samples)
        ],

        # Image path (placeholder)
        'image_path': [f'data/raw/images/property_{i:06d}.jpg' for i in range(n_samples)]
    }

    df = pd.DataFrame(data)

    # Generate realistic prices based on features
    base_price = 200000
    price = (
        base_price +
        df['bedrooms'] * 50000 +
        df['bathrooms'] * 30000 +
        df['sqft_living'] * 150 +
        (df['year_built'] - 1950) * 1000 +
        (df['condition'].map({'Poor': -50000, 'Fair': -20000, 'Average': 0, 'Good': 20000, 'Excellent': 50000})) +
        (df['view_quality'].map({'None': 0, 'Fair': 20000, 'Good': 40000, 'Excellent': 80000})) +
        np.random.normal(0, 50000, n_samples)
    )

    df['price'] = price.clip(lower=100000)

    # Save dataset
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"Sample dataset saved to {output_path}")

    return df


def split_dataset(df: pd.DataFrame, output_dir: str, train_split: float = 0.7, val_split: float = 0.15):
    """Split dataset into train, validation, and test sets"""

    # Shuffle data
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)

    n = len(df)
    train_end = int(n * train_split)
    val_end = int(n * (train_split + val_split))

    train_df = df[:train_end]
    val_df = df[train_end:val_end]
    test_df = df[val_end:]

    # Save splits
    os.makedirs(output_dir, exist_ok=True)
    train_df.to_csv(os.path.join(output_dir, 'train.csv'), index=False)
    val_df.to_csv(os.path.join(output_dir, 'val.csv'), index=False)
    test_df.to_csv(os.path.join(output_dir, 'test.csv'), index=False)

    print(f"\nDataset split:")
    print(f"  Train: {len(train_df)} samples")
    print(f"  Validation: {len(val_df)} samples")
    print(f"  Test: {len(test_df)} samples")


def main(args):
    if args.create_sample:
        # Create sample dataset
        df = create_sample_dataset(args.raw_data_path, args.n_samples)
    else:
        # Load existing data
        print(f"Loading data from {args.raw_data_path}...")
        df = pd.read_csv(args.raw_data_path)

    # Clean data
    print("\nCleaning data...")
    df = clean_property_data(df)

    # Save cleaned data
    processed_path = os.path.join(args.output_dir, 'processed_data.csv')
    df.to_csv(processed_path, index=False)
    print(f"\nProcessed data saved to {processed_path}")

    # Split dataset
    if args.split:
        print("\nSplitting dataset...")
        split_dataset(df, args.output_dir, args.train_split, args.val_split)

    print("\nData preparation complete!")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Prepare property dataset')
    parser.add_argument('--raw_data_path', type=str, default='data/raw/properties.csv',
                       help='Path to raw data CSV')
    parser.add_argument('--output_dir', type=str, default='data/processed',
                       help='Directory to save processed data')
    parser.add_argument('--create_sample', action='store_true',
                       help='Create sample dataset for demonstration')
    parser.add_argument('--n_samples', type=int, default=1000,
                       help='Number of samples to generate (if create_sample=True)')
    parser.add_argument('--split', action='store_true',
                       help='Split data into train/val/test sets')
    parser.add_argument('--train_split', type=float, default=0.7,
                       help='Fraction for training set')
    parser.add_argument('--val_split', type=float, default=0.15,
                       help='Fraction for validation set')

    args = parser.parse_args()
    main(args)
