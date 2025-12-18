import pandas as pd
import csv

def upload_to_google_sheets():
    """
    Upload the results to Google Sheets

    Since we have editor access, we'll prepare the data and provide
    instructions for upload
    """
    # Load the final data
    df = pd.read_csv('../data/final_output_for_sheets.csv')

    print("="  * 70)
    print("UPLOAD INSTRUCTIONS")
    print("=" * 70)

    print("\nOption 1: Direct Import (Recommended)")
    print("-" * 70)
    print("1. Open your Google Sheet:")
    print("   https://docs.google.com/spreadsheets/d/1ypK_SACOonFlBM1MvWFqNLElwwSTUuinDHlSY5vBMjM/edit")
    print("\n2. Go to File > Import")
    print("3. Upload the file: data/final_output_for_sheets.csv")
    print("4. Select 'Replace current sheet'")
    print("5. Click 'Import data'")

    print("\n\nOption 2: Manual Column Addition")
    print("-" * 70)
    print("1. I can create a simpler file with just the two new columns")
    print("2. You can copy/paste them into columns V and W")

    # Create a simplified file with just the new columns
    df_simple = df[['Deal Status', 'ARV']].copy()
    df_simple.to_csv('../data/new_columns_only.csv', index=False)

    print("\nCreated: data/new_columns_only.csv with just the new columns")

    # Show sample data
    print("\n\nSample Data Preview:")
    print("-" * 70)
    print(df[['Address', 'List Price', 'Deal Status', 'ARV']].head(10).to_string(index=False))

    print("\n\n" + "=" * 70)
    print("DEAL HIGHLIGHTS")
    print("=" * 70)

    deals = df[df['Deal Status'] == 'Deal'].sort_values('List Price').head(10)
    print("\nTop 10 Best Deals (Lowest List Price):")
    print("-" * 70)
    for idx, row in deals.iterrows():
        print(f"\n{row['Address']}, {row['City']}")
        print(f"  List: {row['List Price']} | ARV: {row['ARV']}")
        print(f"  Distance: {row['Distance']} | Days: {row['Days On Market']}")

if __name__ == '__main__':
    upload_to_google_sheets()
