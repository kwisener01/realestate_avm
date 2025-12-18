import pandas as pd
import re

def clean_price(price_str):
    """Convert price string to numeric"""
    if pd.isna(price_str):
        return None
    cleaned = re.sub(r'[,$]', '', str(price_str))
    try:
        return int(float(cleaned))
    except:
        return None

def main():
    print("=" * 80)
    print("CREATING HYBRID ARV MODEL - SHOWING BOTH ESTIMATES")
    print("=" * 80)

    # Load both model results
    df_old = pd.read_csv('../data/arv_results_full.csv')
    df_ml = pd.read_csv('../data/arv_ml_results.csv')
    df_original = pd.read_csv('../data/listing_agents.csv')

    print(f"\nLoaded data for {len(df_original)} properties")

    # Create combined dataframe
    df = df_original.copy()

    # Add location model results
    df['ARV_Location'] = df_old['ARV'].apply(clean_price)
    df['Deal_Location'] = df_old['Deal Status']

    # Add ML model results
    df['ARV_ML'] = df_ml['ARV_ML']
    df['Deal_ML'] = df_ml['Deal_Status_ML']

    # Clean list price
    df['list_price_clean'] = df['List Price'].apply(clean_price)

    # Calculate average ARV
    df['ARV_Average'] = ((df['ARV_Location'] + df['ARV_ML']) / 2).astype(int)

    # Determine confidence level
    def get_confidence(row):
        if row['Deal_Location'] == 'Deal' and row['Deal_ML'] == 'Deal':
            return 'HIGH - Both Agree'
        elif row['Deal_Location'] == 'Deal' or row['Deal_ML'] == 'Deal':
            return 'MEDIUM - One Model'
        else:
            return 'LOW - Neither'

    df['Confidence'] = df.apply(get_confidence, axis=1)

    # Create Deal Status column showing both
    def get_deal_status(row):
        if row['Deal_Location'] == 'Deal' and row['Deal_ML'] == 'Deal':
            return 'DEAL ✓✓'
        elif row['Deal_Location'] == 'Deal':
            return 'Deal (Location)'
        elif row['Deal_ML'] == 'Deal':
            return 'Deal (ML)'
        else:
            return 'No Deal'

    df['Deal_Status_Final'] = df.apply(get_deal_status, axis=1)

    # Format ARV columns
    df['ARV_Location_Fmt'] = df['ARV_Location'].apply(lambda x: f"${x:,}" if pd.notna(x) else "")
    df['ARV_ML_Fmt'] = df['ARV_ML'].apply(lambda x: f"${x:,}" if pd.notna(x) else "")
    df['ARV_Average_Fmt'] = df['ARV_Average'].apply(lambda x: f"${x:,}" if pd.notna(x) else "")

    # Statistics
    print("\n" + "=" * 80)
    print("RESULTS SUMMARY")
    print("=" * 80)

    high_conf = (df['Confidence'] == 'HIGH - Both Agree').sum()
    medium_conf = (df['Confidence'] == 'MEDIUM - One Model').sum()
    low_conf = (df['Confidence'] == 'LOW - Neither').sum()

    print(f"\nDeal Confidence Breakdown:")
    print(f"  HIGH (Both Models Agree):   {high_conf} properties ({high_conf/len(df)*100:.1f}%)")
    print(f"  MEDIUM (One Model):         {medium_conf} properties ({medium_conf/len(df)*100:.1f}%)")
    print(f"  LOW (No Deal):              {low_conf} properties ({low_conf/len(df)*100:.1f}%)")

    print(f"\nModel Comparison:")
    location_deals = (df['Deal_Location'] == 'Deal').sum()
    ml_deals = (df['Deal_ML'] == 'Deal').sum()
    print(f"  Location Model Deals:       {location_deals} ({location_deals/len(df)*100:.1f}%)")
    print(f"  ML Model Deals:             {ml_deals} ({ml_deals/len(df)*100:.1f}%)")
    print(f"  Both Agree (High Conf):     {high_conf} ({high_conf/len(df)*100:.1f}%)")

    # Show high confidence deals
    print("\n" + "=" * 80)
    print("TOP 15 HIGH CONFIDENCE DEALS (Both Models Agree)")
    print("=" * 80)

    high_conf_deals = df[df['Confidence'] == 'HIGH - Both Agree'].copy()

    # Calculate spread
    high_conf_deals['Spread'] = high_conf_deals['ARV_Average'] - high_conf_deals['list_price_clean']
    high_conf_deals = high_conf_deals.sort_values('Spread', ascending=False)

    print(f"\n{'Address':<35} {'List Price':<12} {'ARV (Loc)':<12} {'ARV (ML)':<12} {'Avg ARV':<12} {'Profit':<12}")
    print("-" * 107)

    for idx, row in high_conf_deals.head(15).iterrows():
        print(f"{row['Address'][:35]:<35} "
              f"${row['list_price_clean']:>10,}  "
              f"${row['ARV_Location']:>10,}  "
              f"${row['ARV_ML']:>10,}  "
              f"${row['ARV_Average']:>10,}  "
              f"${row['Spread']:>10,}")

    # Prepare for Google Sheets upload
    print("\n" + "=" * 80)
    print("PREPARING DATA FOR GOOGLE SHEETS")
    print("=" * 80)

    # Create columns for sheet
    df_sheet = pd.DataFrame()
    df_sheet['Deal Status'] = df['Deal_Status_Final']
    df_sheet['ARV (Location)'] = df['ARV_Location_Fmt']
    df_sheet['ARV (ML Model)'] = df['ARV_ML_Fmt']
    df_sheet['ARV (Average)'] = df['ARV_Average_Fmt']
    df_sheet['Confidence'] = df['Confidence']

    # Save
    output_file = '../data/hybrid_arv_for_sheets.csv'
    df_sheet.to_csv(output_file, index=False)

    print(f"\nColumns to add to Google Sheets:")
    print(f"  1. Deal Status - Shows if deal and which model(s)")
    print(f"  2. ARV (Location) - Location-based estimate")
    print(f"  3. ARV (ML Model) - Machine learning estimate")
    print(f"  4. ARV (Average) - Average of both")
    print(f"  5. Confidence - Agreement level")

    print(f"\nFile saved: {output_file}")

    # Also save full results
    df.to_csv('../data/hybrid_arv_full_results.csv', index=False)
    print(f"Full results: data/hybrid_arv_full_results.csv")

    print("\n" + "=" * 80)
    print(f"✓ {high_conf} HIGH CONFIDENCE DEALS ready for review!")
    print("=" * 80)

if __name__ == '__main__':
    main()
