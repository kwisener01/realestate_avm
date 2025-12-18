import pandas as pd
import gspread
from google.oauth2.service_account import Credentials
import os

def update_google_sheet_directly():
    """
    Update Google Sheet by appending Deal Status and ARV columns
    """
    try:
        # Try to authenticate with gspread
        # Note: This requires Google API credentials

        print("=" * 70)
        print("GOOGLE SHEETS UPDATE")
        print("=" * 70)

        # Check for credentials file
        creds_file = '../credentials.json'

        if not os.path.exists(creds_file):
            print("\n[INFO] No credentials file found.")
            print("\nTo update Google Sheets programmatically, you need:")
            print("1. Google Cloud service account credentials")
            print("2. Save them as 'credentials.json' in the project root")
            print("\nAlternatively, I can prepare the data for manual paste...")

            # Prepare data for manual update
            df = pd.read_csv('../data/arv_results_full.csv')

            # Create columns to paste
            df_cols = pd.DataFrame()
            df_cols['Deal Status'] = df['Deal Status']
            df_cols['ARV'] = df['ARV'].apply(lambda x: f"${x:,.0f}" if pd.notna(x) else "")

            # Save
            df_cols.to_csv('../data/paste_these_columns.csv', index=False)

            print("\n" + "-" * 70)
            print("MANUAL UPDATE OPTION:")
            print("-" * 70)
            print("\n1. I've created: data/paste_these_columns.csv")
            print(f"   Contains {len(df_cols)} rows with 2 columns")
            print("\n2. Open this file and your Google Sheet side-by-side")
            print("\n3. In the CSV:")
            print("   - Select all data (Ctrl+A)")
            print("   - Copy (Ctrl+C)")
            print("\n4. In Google Sheets:")
            print("   - Click cell U1 (first empty column)")
            print("   - Paste (Ctrl+V)")
            print("\nYour existing data will not be touched!")

            # Show preview
            print("\n\n" + "=" * 70)
            print("DATA PREVIEW:")
            print("=" * 70)
            print(df_cols.head(10).to_string(index=True))

            print(f"\n\nDeals: {(df_cols['Deal Status'] == 'Deal').sum()}")
            print(f"No Deals: {(df_cols['Deal Status'] == 'No Deal').sum()}")

            return

        # If credentials exist, use them
        scopes = [
            'https://www.googleapis.com/auth/spreadsheets',
            'https://www.googleapis.com/auth/drive'
        ]

        creds = Credentials.from_service_account_file(creds_file, scopes=scopes)
        client = gspread.authorize(creds)

        # Open the spreadsheet
        sheet_id = '1ypK_SACOonFlBM1MvWFqNLElwwSTUuinDHlSY5vBMjM'
        sheet = client.open_by_key(sheet_id)
        worksheet = sheet.get_worksheet(0)  # First sheet

        print("\n[SUCCESS] Connected to Google Sheets!")

        # Load results
        df = pd.read_csv('../data/arv_results_full.csv')

        # Prepare columns
        deal_status = df['Deal Status'].tolist()
        arv = df['ARV'].apply(lambda x: f"${x:,.0f}" if pd.notna(x) else "").tolist()

        # Find the next empty column (after List Price which is column T/20)
        # Get current number of columns
        current_cols = worksheet.col_count
        next_col = current_cols + 1

        print(f"\n[INFO] Current sheet has {current_cols} columns")
        print(f"[INFO] Adding columns at position {next_col} and {next_col + 1}")

        # Expand sheet if needed
        if next_col + 1 > current_cols:
            needed_cols = next_col + 1
            print(f"[INFO] Expanding sheet to {needed_cols} columns...")
            worksheet.resize(rows=worksheet.row_count, cols=needed_cols)
            print(f"[SUCCESS] Sheet expanded!")

        # Add headers
        worksheet.update_cell(1, next_col, 'Deal Status')
        worksheet.update_cell(1, next_col + 1, 'ARV')

        print("\n[INFO] Headers added")

        # Add data in batches for efficiency
        print("[INFO] Uploading Deal Status column...")
        cell_list = []
        for i, value in enumerate(deal_status, start=2):
            cell_list.append(gspread.Cell(i, next_col, value))

        worksheet.update_cells(cell_list, value_input_option='RAW')

        print("[INFO] Uploading ARV column...")
        cell_list = []
        for i, value in enumerate(arv, start=2):
            cell_list.append(gspread.Cell(i, next_col + 1, value))

        worksheet.update_cells(cell_list, value_input_option='RAW')

        print("\n" + "=" * 70)
        print("[SUCCESS] Google Sheet updated!")
        print("=" * 70)
        print(f"\nAdded columns:")
        print(f"  Column {next_col}: Deal Status")
        print(f"  Column {next_col + 1}: ARV")
        print(f"\nTotal rows updated: {len(df)}")

    except Exception as e:
        print(f"\n[ERROR] {str(e)}")
        print("\nPlease use the manual update method described above.")

if __name__ == '__main__':
    update_google_sheet_directly()
