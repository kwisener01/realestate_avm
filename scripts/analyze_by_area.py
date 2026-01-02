import pandas as pd
import numpy as np

def analyze_arv_by_area():
    """Analyze ARV multiples by geographic area (city and zip code)"""

    print("="*100)
    print("ARV MULTIPLES BY GEOGRAPHIC AREA")
    print("="*100)

    # Load the flip analysis data
    df = pd.read_csv('data/flip_analysis_results.csv')

    # Clean zip codes
    df['Zip'] = df['Zip'].astype(str).str[:5]

    # City-level analysis
    print("\n" + "="*100)
    print("CITY-LEVEL ARV MULTIPLES (Minimum 5 properties)")
    print("="*100)

    city_stats = df.groupby('City').agg({
        'sale_to_assessed_ratio': ['count', 'mean', 'median', 'std', 'min', 'max'],
        'last_sale_price': 'median'
    }).round(2)

    city_stats.columns = ['Count', 'Mean', 'Median', 'StdDev', 'Min', 'Max', 'Median Sale Price']
    city_stats = city_stats[city_stats['Count'] >= 5].sort_values('Median', ascending=False)

    print(city_stats.to_string())

    # Create city tier classifications
    print("\n" + "="*100)
    print("CITY TIER CLASSIFICATION")
    print("="*100)

    city_stats['Tier'] = pd.cut(
        city_stats['Median'],
        bins=[0, 2.6, 2.8, 3.0, 20],
        labels=['Lower Appreciation', 'Moderate Appreciation', 'High Appreciation', 'Very High Appreciation']
    )

    for tier in ['Very High Appreciation', 'High Appreciation', 'Moderate Appreciation', 'Lower Appreciation']:
        cities_in_tier = city_stats[city_stats['Tier'] == tier]
        if len(cities_in_tier) > 0:
            print(f"\n{tier} (Median ARV Multiple):")
            for city, row in cities_in_tier.iterrows():
                print(f"  - {city}: {row['Median']:.2f}x (n={int(row['Count'])}, price=${row['Median Sale Price']:,.0f})")

    # Zip code analysis
    print("\n" + "="*100)
    print("ZIP CODE-LEVEL ARV MULTIPLES (Minimum 10 properties)")
    print("="*100)

    zip_stats = df.groupby(['Zip', 'City']).agg({
        'sale_to_assessed_ratio': ['count', 'mean', 'median', 'min', 'max'],
        'last_sale_price': 'median'
    }).round(2)

    zip_stats.columns = ['Count', 'Mean', 'Median', 'Min', 'Max', 'Median Price']
    zip_stats = zip_stats[zip_stats['Count'] >= 10].sort_values('Median', ascending=False)

    print(zip_stats.to_string())

    # Create zip-level recommendation table
    print("\n" + "="*100)
    print("RECOMMENDED ARV MULTIPLES BY ZIP CODE")
    print("="*100)
    print(f"{'Zip':<8} {'City':<18} {'Sample':<8} {'Conservative':<13} {'Moderate':<13} {'Aggressive':<13} {'Median Price':<15}")
    print("-"*100)

    for (zipcode, city), row in zip_stats.iterrows():
        # Calculate conservative (10% below median), moderate (median), aggressive (median + 0.5 std dev)
        conservative = row['Median'] * 0.9
        moderate = row['Median']
        # Cap aggressive at 75th percentile approximation
        aggressive = min(row['Median'] * 1.15, row['Max'] * 0.9)

        print(f"{zipcode:<8} {city:<18} {int(row['Count']):<8} {conservative:.2f}x{'':<8} {moderate:.2f}x{'':<8} {aggressive:.2f}x{'':<8} ${row['Median Price']:>10,.0f}")

    # Neighborhood characteristics
    print("\n" + "="*100)
    print("MARKET INSIGHTS BY AREA")
    print("="*100)

    # High-value markets (median sale price > $500k)
    high_value = city_stats[city_stats['Median Sale Price'] > 500000].sort_values('Median Sale Price', ascending=False)
    if len(high_value) > 0:
        print("\nHigh-Value Markets (Median Sale > $500k):")
        for city, row in high_value.iterrows():
            print(f"  {city}: {row['Median']:.2f}x multiple, ${row['Median Sale Price']:,.0f} median price")
            print(f"    Best for: Experienced investors with larger capital")

    # Entry-level markets (median sale price < $300k)
    entry_level = city_stats[city_stats['Median Sale Price'] < 300000].sort_values('Median', ascending=False)
    if len(entry_level) > 0:
        print("\nEntry-Level Markets (Median Sale < $300k):")
        for city, row in entry_level.iterrows():
            print(f"  {city}: {row['Median']:.2f}x multiple, ${row['Median Sale Price']:,.0f} median price")
            print(f"    Best for: New investors, cash flow focus")

    # Most consistent markets (low standard deviation)
    consistent = city_stats.nsmallest(5, 'StdDev')
    print("\nMost Consistent Markets (Lowest Volatility):")
    for city, row in consistent.iterrows():
        print(f"  {city}: {row['Median']:.2f}x ± {row['StdDev']:.2f} (StdDev)")
        print(f"    Best for: Risk-averse investors")

    # Save area-specific multiples
    output_data = []
    for (zipcode, city), row in zip_stats.iterrows():
        output_data.append({
            'Zip': zipcode,
            'City': city,
            'Sample_Size': int(row['Count']),
            'Conservative_Multiple': round(row['Median'] * 0.9, 2),
            'Moderate_Multiple': round(row['Median'], 2),
            'Aggressive_Multiple': round(min(row['Median'] * 1.15, row['Max'] * 0.9), 2),
            'Median_Sale_Price': round(row['Median Price'], 0),
            'Min_Multiple': row['Min'],
            'Max_Multiple': row['Max']
        })

    output_df = pd.DataFrame(output_data)
    output_df.to_csv('data/arv_multiples_by_area.csv', index=False)
    print(f"\n\nSaved area-specific multiples to data/arv_multiples_by_area.csv")

    # Also save city-level data
    city_output = city_stats.reset_index()
    city_output['Conservative'] = (city_output['Median'] * 0.9).round(2)
    city_output['Aggressive'] = city_output.apply(lambda x: round(min(x['Median'] * 1.15, x['Max'] * 0.9), 2), axis=1)
    city_output.to_csv('data/arv_multiples_by_city.csv', index=False)
    print(f"Saved city-level multiples to data/arv_multiples_by_city.csv")

if __name__ == "__main__":
    analyze_arv_by_area()
