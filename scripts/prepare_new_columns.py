import pandas as pd

def prepare_columns_for_append():
    """
    Prepare just the Deal Status and ARV columns to append to existing sheet
    """
    # Load the full results
    df = pd.read_csv('../data/arv_results_full.csv')

    # Create a dataframe with just the two new columns
    df_new_cols = pd.DataFrame()
    df_new_cols['Deal Status'] = df['Deal Status']

    # Format ARV as currency
    df_new_cols['ARV'] = df['ARV'].apply(lambda x: f"${x:,.0f}" if pd.notna(x) else "")

    # Save to CSV
    output_file = '../data/columns_to_paste.csv'
    df_new_cols.to_csv(output_file, index=False)

    print("=" * 70)
    print("COLUMNS READY TO PASTE")
    print("=" * 70)

    print(f"\nCreated file: {output_file}")
    print(f"Contains {len(df_new_cols)} rows")

    print("\n\nINSTRUCTIONS:")
    print("-" * 70)
    print("1. Open your Google Sheet:")
    print("   https://docs.google.com/spreadsheets/d/1ypK_SACOonFlBM1MvWFqNLElwwSTUuinDHlSY5vBMjM/edit")
    print("\n2. Click on column U (or the first empty column after 'List Price')")
    print("\n3. Open the CSV file: data/columns_to_paste.csv")
    print("   - Select all data (Ctrl+A)")
    print("   - Copy (Ctrl+C)")
    print("\n4. Go back to Google Sheets")
    print("   - Click on cell U1 (first empty cell in row 1)")
    print("   - Paste (Ctrl+V)")
    print("\n5. Done! The columns will be added to the right without changing existing data")

    # Show preview
    print("\n\n" + "=" * 70)
    print("DATA PREVIEW (First 15 rows):")
    print("=" * 70)
    print(df_new_cols.head(15).to_string(index=True))

    # Show statistics
    print("\n\n" + "=" * 70)
    print("STATISTICS:")
    print("=" * 70)
    deal_counts = df_new_cols['Deal Status'].value_counts()
    for status, count in deal_counts.items():
        percentage = (count / len(df_new_cols)) * 100
        print(f"{status}: {count} ({percentage:.1f}%)")

    return df_new_cols

if __name__ == '__main__':
    prepare_columns_for_append()
