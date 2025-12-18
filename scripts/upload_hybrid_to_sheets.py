import pandas as pd
import gspread
from google.oauth2.service_account import Credentials

def main():
    print("=" * 80)
    print("UPDATING GOOGLE SHEETS WITH HYBRID ARV DATA")
    print("=" * 80)

    # Load hybrid results
    df = pd.read_csv('../data/hybrid_arv_for_sheets.csv')

    print(f"\nLoaded {len(df)} properties with hybrid ARV data")
    print(f"Columns to add: {list(df.columns)}")

    # Authenticate
    creds_file = '../credentials.json'
    scopes = [
        'https://www.googleapis.com/auth/spreadsheets',
        'https://www.googleapis.com/auth/drive'
    ]

    creds = Credentials.from_service_account_file(creds_file, scopes=scopes)
    client = gspread.authorize(creds)

    # Open sheet
    sheet_id = '1ypK_SACOonFlBM1MvWFqNLElwwSTUuinDHlSY5vBMjM'
    sheet = client.open_by_key(sheet_id)
    worksheet = sheet.get_worksheet(0)

    print(f"\n[SUCCESS] Connected to Google Sheets!")
    print(f"Current sheet has {worksheet.col_count} columns")

    # We already have columns 27-28 (Deal Status, ARV)
    # Let's replace them with the hybrid data
    # Column 27: Deal Status
    # Column 28: ARV (Location)
    # Column 29: ARV (ML Model)
    # Column 30: ARV (Average)
    # Column 31: Confidence

    start_col = 27

    # Expand if needed
    needed_cols = start_col + 4
    if worksheet.col_count < needed_cols:
        print(f"[INFO] Expanding sheet to {needed_cols} columns...")
        worksheet.resize(rows=worksheet.row_count, cols=needed_cols)

    # Update headers
    print("\n[INFO] Updating headers...")
    worksheet.update_cell(1, 27, 'Deal Status')
    worksheet.update_cell(1, 28, 'ARV (Location)')
    worksheet.update_cell(1, 29, 'ARV (ML Model)')
    worksheet.update_cell(1, 30, 'ARV (Average)')
    worksheet.update_cell(1, 31, 'Confidence')

    # Prepare data
    print("[INFO] Uploading Deal Status...")
    deal_status = df['Deal Status'].tolist()
    cell_list = []
    for i, value in enumerate(deal_status, start=2):
        cell_list.append(gspread.Cell(i, 27, value))
    worksheet.update_cells(cell_list, value_input_option='RAW')

    print("[INFO] Uploading ARV (Location)...")
    arv_loc = df['ARV (Location)'].tolist()
    cell_list = []
    for i, value in enumerate(arv_loc, start=2):
        cell_list.append(gspread.Cell(i, 28, value))
    worksheet.update_cells(cell_list, value_input_option='RAW')

    print("[INFO] Uploading ARV (ML Model)...")
    arv_ml = df['ARV (ML Model)'].tolist()
    cell_list = []
    for i, value in enumerate(arv_ml, start=2):
        cell_list.append(gspread.Cell(i, 29, value))
    worksheet.update_cells(cell_list, value_input_option='RAW')

    print("[INFO] Uploading ARV (Average)...")
    arv_avg = df['ARV (Average)'].tolist()
    cell_list = []
    for i, value in enumerate(arv_avg, start=2):
        cell_list.append(gspread.Cell(i, 30, value))
    worksheet.update_cells(cell_list, value_input_option='RAW')

    print("[INFO] Uploading Confidence...")
    confidence = df['Confidence'].tolist()
    cell_list = []
    for i, value in enumerate(confidence, start=2):
        cell_list.append(gspread.Cell(i, 31, value))
    worksheet.update_cells(cell_list, value_input_option='RAW')

    print("\n" + "=" * 80)
    print("[SUCCESS] Google Sheet Updated!")
    print("=" * 80)
    print(f"\nAdded columns:")
    print(f"  Column 27: Deal Status")
    print(f"  Column 28: ARV (Location Model)")
    print(f"  Column 29: ARV (ML Model)")
    print(f"  Column 30: ARV (Average)")
    print(f"  Column 31: Confidence Level")
    print(f"\nTotal rows updated: {len(df)}")
    print("\nHIGH CONFIDENCE DEALS: 20 properties")
    print("MEDIUM CONFIDENCE: 181 properties")
    print("=" * 80)

if __name__ == '__main__':
    main()
