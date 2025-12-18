import pandas as pd
import re

def clean_price(price_str):
    """Convert price string to numeric"""
    if pd.isna(price_str):
        return None
    cleaned = re.sub(r'[,$]', '', str(price_str))
    try:
        return float(cleaned)
    except:
        return None

def main():
    # Load Decatur property export
    print("=" * 80)
    print("LOADING DECATUR PROPERTY EXPORT DATA")
    print("=" * 80)

    file_path = '../Property Export Decatur+List.xlsx'

    try:
        # Read Excel file
        df = pd.read_excel(file_path)

        print(f"\nLoaded {len(df)} properties")
        print(f"\nColumns ({len(df.columns)}):")
        for i, col in enumerate(df.columns, 1):
            print(f"  {i}. {col}")

        # Show data types
        print(f"\n{'-' * 80}")
        print("Data Types:")
        print(df.dtypes.to_string())

        # Show sample data
        print(f"\n{'-' * 80}")
        print("Sample Data (first 5 rows):")
        print(df.head().to_string())

        # Check for key columns we need
        print(f"\n{'-' * 80}")
        print("KEY COLUMN ANALYSIS:")

        key_columns = ['price', 'list price', 'sold', 'sale', 'address', 'beds', 'baths', 'sqft']

        for key in key_columns:
            matches = [col for col in df.columns if key.lower() in col.lower()]
            if matches:
                print(f"\n{key.upper()}: Found columns: {matches}")
                # Show sample values
                for col in matches[:2]:  # Show first 2 matching columns
                    sample_vals = df[col].dropna().head(3).tolist()
                    print(f"  {col}: {sample_vals}")

        # Statistics
        print(f"\n{'-' * 80}")
        print("DATASET STATISTICS:")

        # Look for price columns
        price_cols = [col for col in df.columns if 'price' in col.lower() or 'value' in col.lower()]
        for col in price_cols:
            try:
                df[f'{col}_clean'] = df[col].apply(clean_price)
                stats = df[f'{col}_clean'].describe()
                print(f"\n{col}:")
                print(f"  Count: {stats['count']:.0f}")
                print(f"  Mean: ${stats['mean']:,.0f}")
                print(f"  Median: ${stats['50%']:,.0f}")
                print(f"  Min: ${stats['min']:,.0f}")
                print(f"  Max: ${stats['max']:,.0f}")
            except:
                pass

        # Save to CSV for easier processing
        csv_path = '../data/decatur_properties.csv'
        df.to_csv(csv_path, index=False)

        print(f"\n{'-' * 80}")
        print(f"Saved to: {csv_path}")
        print("=" * 80)

        return df

    except Exception as e:
        print(f"\n[ERROR] Failed to load file: {e}")
        import traceback
        traceback.print_exc()
        return None

if __name__ == '__main__':
    main()
