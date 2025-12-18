import pandas as pd
import requests

def update_google_sheet(sheet_id, data_file):
    """
    Update Google Sheet with ARV and Deal Status columns
    """
    # Load the results
    df = pd.read_csv(data_file)

    print(f"Loaded {len(df)} properties from results")

    # Prepare data for upload
    # We need to create columns: List Price | Deal Status | ARV

    # Format ARV as currency string
    df['ARV_formatted'] = df['ARV'].apply(lambda x: f"${x:,.0f}" if pd.notna(x) else "")

    # Create output dataframe with all original columns plus new ones
    # We'll insert Deal Status and ARV right after List Price

    # Load original data to preserve all columns
    df_original = pd.read_csv('../data/listing_agents.csv')

    # Add the new columns
    df_original['Deal Status'] = df['Deal Status']
    df_original['ARV'] = df['ARV_formatted']

    # Reorder columns to put Deal Status and ARV after List Price
    cols = list(df_original.columns)

    # Find index of List Price
    list_price_idx = cols.index('List Price')

    # Remove Deal Status and ARV from their current positions
    cols.remove('Deal Status')
    cols.remove('ARV')

    # Insert after List Price
    cols.insert(list_price_idx + 1, 'Deal Status')
    cols.insert(list_price_idx + 2, 'ARV')

    # Reorder dataframe
    df_final = df_original[cols]

    # Save to CSV for manual upload
    output_file = '../data/final_output_for_sheets.csv'
    df_final.to_csv(output_file, index=False)

    print(f"\nData prepared and saved to: {output_file}")
    print(f"\nColumns order:")
    for i, col in enumerate(cols, 1):
        print(f"  {i}. {col}")

    # Print statistics
    print(f"\n{'=' * 70}")
    print("SUMMARY:")
    print(f"{'=' * 70}")
    print(f"Total Properties: {len(df_final)}")
    print(f"Deals: {(df_final['Deal Status'] == 'Deal').sum()}")
    print(f"No Deals: {(df_final['Deal Status'] == 'No Deal').sum()}")

    return df_final

def main():
    sheet_id = '1ypK_SACOonFlBM1MvWFqNLElwwSTUuinDHlSY5vBMjM'
    data_file = '../data/arv_results_full.csv'

    df = update_google_sheet(sheet_id, data_file)

    print(f"\n{'=' * 70}")
    print("NEXT STEPS:")
    print(f"{'=' * 70}")
    print("\nThe data is ready in: data/final_output_for_sheets.csv")
    print("\nYou can:")
    print("  1. Open the file and copy all data")
    print("  2. Paste it into your Google Sheet")
    print("\nOr I can upload it programmatically if you provide API credentials.")

if __name__ == '__main__':
    main()
