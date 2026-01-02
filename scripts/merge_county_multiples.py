"""
Merge multiple county ARV multiples into unified lookup tables

This script combines ARV multiples from different counties (DeKalb, Cobb, Fulton, etc.)
into single unified files that the API can use for lookups.

Usage:
    python scripts/merge_county_multiples.py
"""

import pandas as pd
import os
import glob

def merge_multiples():
    """Merge all county-specific multiples into unified files"""

    print("="*100)
    print("MERGING COUNTY ARV MULTIPLES")
    print("="*100)

    # Find all county-specific files
    area_files = glob.glob('data/*_multiples_by_area.csv')
    city_files = glob.glob('data/*_multiples_by_city.csv')

    print(f"\nFound area files: {len(area_files)}")
    for f in area_files:
        county = os.path.basename(f).replace('_multiples_by_area.csv', '')
        df = pd.read_csv(f)
        print(f"  - {county}: {len(df)} zip codes")

    print(f"\nFound city files: {len(city_files)}")
    for f in city_files:
        county = os.path.basename(f).replace('_multiples_by_city.csv', '')
        df = pd.read_csv(f)
        print(f"  - {county}: {len(df)} cities")

    # Merge area/zip code multiples
    print("\n" + "="*100)
    print("STEP 1: Merging Zip Code Multiples")
    print("="*100)

    all_area_dfs = []
    for file in area_files:
        county_name = os.path.basename(file).replace('_multiples_by_area.csv', '')
        df = pd.read_csv(file)
        df['County'] = county_name.title()
        all_area_dfs.append(df)
        print(f"  Added {len(df)} zip codes from {county_name}")

    if all_area_dfs:
        merged_area = pd.concat(all_area_dfs, ignore_index=True)

        # Remove duplicates (keep first occurrence)
        duplicates = merged_area[merged_area.duplicated(subset=['Zip'], keep=False)]
        if len(duplicates) > 0:
            print(f"\n  WARNING: Found {len(duplicates)} duplicate zip codes")
            print(duplicates[['Zip', 'City', 'County', 'Sample_Size']].to_string())
            print(f"\n  Keeping first occurrence (usually DeKalb)")

        merged_area = merged_area.drop_duplicates(subset=['Zip'], keep='first')

        # Save merged file
        output_file = 'data/arv_multiples_by_area.csv'
        merged_area.to_csv(output_file, index=False)
        print(f"\n  Saved: {output_file}")
        print(f"  Total zip codes: {len(merged_area)}")

        # Show coverage by county
        print(f"\n  Coverage by county:")
        coverage = merged_area.groupby('County').size()
        for county, count in coverage.items():
            print(f"    {county}: {count} zip codes")

    # Merge city multiples
    print("\n" + "="*100)
    print("STEP 2: Merging City Multiples")
    print("="*100)

    all_city_dfs = []
    for file in city_files:
        county_name = os.path.basename(file).replace('_multiples_by_city.csv', '')
        df = pd.read_csv(file)
        df['County'] = county_name.title()
        all_city_dfs.append(df)
        print(f"  Added {len(df)} cities from {county_name}")

    if all_city_dfs:
        merged_city = pd.concat(all_city_dfs, ignore_index=True)

        # Remove duplicates (keep first occurrence)
        duplicates = merged_city[merged_city.duplicated(subset=['City'], keep=False)]
        if len(duplicates) > 0:
            print(f"\n  WARNING: Found {len(duplicates)} duplicate cities")
            print(duplicates[['City', 'County', 'Count']].to_string())
            print(f"\n  Keeping first occurrence")

        merged_city = merged_city.drop_duplicates(subset=['City'], keep='first')

        # Save merged file
        output_file = 'data/arv_multiples_by_city.csv'
        merged_city.to_csv(output_file, index=False)
        print(f"\n  Saved: {output_file}")
        print(f"  Total cities: {len(merged_city)}")

        # Show coverage by county
        print(f"\n  Coverage by county:")
        coverage = merged_city.groupby('County').size()
        for county, count in coverage.items():
            print(f"    {county}: {count} cities")

    # Summary
    print("\n" + "="*100)
    print("MERGE COMPLETE!")
    print("="*100)

    print(f"\nUnified ARV lookup tables:")
    print(f"  - data/arv_multiples_by_area.csv ({len(merged_area)} zip codes)")
    print(f"  - data/arv_multiples_by_city.csv ({len(merged_city)} cities)")

    print(f"\nNext steps:")
    print(f"  1. Test with a property from new county:")
    print(f"     python scripts/explain_arv_calculation.py")
    print(f"\n  2. Commit and deploy:")
    print(f"     git add data/arv_multiples_*.csv")
    print(f"     git commit -m 'feat: add Cobb County ARV multiples'")
    print(f"     git push origin master")

    # Show sample of new coverage
    print(f"\n" + "="*100)
    print("SAMPLE OF MERGED DATA")
    print("="*100)

    print(f"\nZip Codes (showing first 10):")
    print(merged_area[['Zip', 'City', 'County', 'Moderate_Multiple', 'Sample_Size']].head(10).to_string(index=False))

    print(f"\nCities:")
    print(merged_city[['City', 'County', 'Median', 'Count']].to_string(index=False))

    return True


def main():
    success = merge_multiples()

    if success:
        print("\n" + "="*100)
        print("Ready to deploy!")
        print("="*100)

if __name__ == "__main__":
    main()
