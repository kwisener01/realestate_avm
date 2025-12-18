import pandas as pd
import re

def extract_distance(dist_str):
    """Extract numeric distance from string like '50.8 mi'"""
    if pd.isna(dist_str):
        return None
    match = re.search(r'([\d.]+)', str(dist_str))
    if match:
        return float(match.group(1))
    return None

def clean_price(price_str):
    """Convert price string to numeric"""
    if pd.isna(price_str):
        return None
    cleaned = re.sub(r'[,$]', '', str(price_str))
    try:
        return float(cleaned)
    except:
        return None

def show_arv_calculation(row):
    """Show detailed ARV calculation breakdown"""

    # Extract data
    address = row['Address']
    city = row['City']
    distance = extract_distance(row['Distance'])
    dom = row['Days On Market']
    list_price = clean_price(row['List Price'])

    print("\n" + "=" * 80)
    print(f"PROPERTY: {address}, {city}")
    print("=" * 80)

    print(f"\nInput Data:")
    print(f"  List Price: ${list_price:,.0f}")
    print(f"  Distance from Atlanta: {distance:.1f} miles")
    print(f"  Days on Market: {dom}")
    print(f"  City: {city}")
    print(f"  County: {row['County']}")

    print(f"\n{'-' * 80}")
    print("ARV CALCULATION BREAKDOWN:")
    print("-" * 80)

    # Base multiplier
    base_multiplier = 1.75
    print(f"\n1. BASE MULTIPLIER: {base_multiplier}x")
    print(f"   (Standard fix-and-flip baseline)")

    # Distance bonus
    if distance <= 10:
        distance_bonus = 0.25
        distance_reason = "0-10 mi (Very close to Atlanta - high demand)"
    elif distance <= 20:
        distance_bonus = 0.15
        distance_reason = "10-20 mi (Close to Atlanta - good demand)"
    elif distance <= 30:
        distance_bonus = 0.10
        distance_reason = "20-30 mi (Moderate distance)"
    elif distance <= 40:
        distance_bonus = 0.05
        distance_reason = "30-40 mi (Further out)"
    else:
        distance_bonus = 0.0
        distance_reason = "40+ mi (Distant from city center)"

    print(f"\n2. DISTANCE BONUS: +{distance_bonus}x")
    print(f"   {distance_reason}")

    # Days on market bonus
    if dom >= 180:
        dom_bonus = 0.15
        dom_reason = "180+ days (Very distressed - high negotiation potential)"
    elif dom >= 90:
        dom_bonus = 0.10
        dom_reason = "90-180 days (Distressed - good negotiation room)"
    elif dom >= 30:
        dom_bonus = 0.05
        dom_reason = "30-90 days (Some negotiation potential)"
    else:
        dom_bonus = 0.0
        dom_reason = "0-30 days (Fresh listing)"

    print(f"\n3. DAYS ON MARKET BONUS: +{dom_bonus}x")
    print(f"   {dom_reason}")

    # City bonus
    high_value_cities = ['Atlanta', 'Decatur', 'Brookhaven', 'Sandy Springs',
                        'Alpharetta', 'Roswell', 'Marietta']
    moderate_value_cities = ['Lithonia', 'Riverdale', 'College Park', 'East Point',
                            'Forest Park', 'Smyrna', 'Dunwoody']

    city_bonus = 0.0
    if city in high_value_cities:
        city_bonus = 0.10
        city_reason = "High-value market (strong appreciation)"
    elif city in moderate_value_cities:
        city_bonus = 0.05
        city_reason = "Moderate-value market (steady appreciation)"
    else:
        city_bonus = 0.0
        city_reason = "Standard market"

    print(f"\n4. CITY PREMIUM: +{city_bonus}x")
    print(f"   {city_reason}")

    # Calculate final multiplier
    final_multiplier = base_multiplier + distance_bonus + dom_bonus + city_bonus

    # Cap between 1.5 and 2.3
    uncapped_multiplier = final_multiplier
    final_multiplier = max(1.5, min(2.3, final_multiplier))

    print(f"\n{'-' * 80}")
    print(f"TOTAL MULTIPLIER:")
    print(f"  {base_multiplier} (base) + {distance_bonus} (distance) + {dom_bonus} (DOM) + {city_bonus} (city)")
    print(f"  = {uncapped_multiplier}x")
    if uncapped_multiplier != final_multiplier:
        print(f"  Capped to: {final_multiplier}x (range: 1.5x - 2.3x)")
    print("-" * 80)

    # Calculate ARV
    arv = list_price * final_multiplier

    print(f"\nARV CALCULATION:")
    print(f"  List Price × Multiplier = ARV")
    print(f"  ${list_price:,.0f} × {final_multiplier}x = ${arv:,.0f}")

    # Deal status
    deal_threshold = arv * 0.5
    is_deal = list_price <= deal_threshold

    print(f"\n{'-' * 80}")
    print(f"DEAL ANALYSIS:")
    print(f"  50% of ARV = ${deal_threshold:,.0f}")
    print(f"  List Price = ${list_price:,.0f}")
    print(f"  List Price <= 50% of ARV? {is_deal}")
    print(f"  -> DEAL STATUS: {'[DEAL]' if is_deal else '[NO DEAL]'}")

    if is_deal:
        spread = arv - list_price
        roi = (spread / list_price) * 100
        print(f"\n  Potential Profit: ${spread:,.0f}")
        print(f"  ROI Potential: {roi:.1f}%")

    print("=" * 80)

def main():
    # Load data
    df = pd.read_csv('../data/arv_results_full.csv')

    print("\n" * 2)
    print("+" + "=" * 78 + "+")
    print("|" + " " * 20 + "ARV CALCULATION BREAKDOWN" + " " * 33 + "|")
    print("+" + "=" * 78 + "+")

    # Show examples from different categories

    # 1. A good deal
    print("\n\n" + "#" * 80)
    print("EXAMPLE 1: GOOD DEAL (Close to Atlanta, Long DOM)")
    print("#" * 80)
    deal = df[df['Deal Status'] == 'Deal'].iloc[0]
    show_arv_calculation(deal)

    # 2. A high-value area deal
    print("\n\n" + "#" * 80)
    print("EXAMPLE 2: ATLANTA PROPERTY (Premium Location)")
    print("#" * 80)
    atlanta_deals = df[(df['City'] == 'Atlanta') & (df['Deal Status'] == 'Deal')]
    if len(atlanta_deals) > 0:
        show_arv_calculation(atlanta_deals.iloc[0])

    # 3. A no deal
    print("\n\n" + "#" * 80)
    print("EXAMPLE 3: NO DEAL (Fresh Listing, Far from Atlanta)")
    print("#" * 80)
    no_deal = df[df['Deal Status'] == 'No Deal'].iloc[0]
    show_arv_calculation(no_deal)

    # 4. A borderline case
    print("\n\n" + "#" * 80)
    print("EXAMPLE 4: BORDERLINE CASE (Mid-range)")
    print("#" * 80)
    mid_range = df[(df['list_price_clean'] > 150000) & (df['list_price_clean'] < 200000)].iloc[5]
    show_arv_calculation(mid_range)

    print("\n\n" + "=" * 80)
    print("KEY TAKEAWAYS:")
    print("=" * 80)
    print("\n1. Closer to Atlanta = Higher ARV potential (+0.25x max)")
    print("2. Longer on market = More distressed = Better deal (+0.15x max)")
    print("3. Premium cities get additional bonus (+0.10x max)")
    print("4. Final multiplier ranges from 1.5x to 2.3x")
    print("5. DEAL = List Price <= 50% of ARV")
    print("\n" + "=" * 80 + "\n")

if __name__ == '__main__':
    main()
