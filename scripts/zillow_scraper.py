import requests
from bs4 import BeautifulSoup
import pandas as pd
import time
import re
from urllib.parse import quote
import json

class ZillowScraper:
    def __init__(self):
        self.headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
            'Accept-Language': 'en-US,en;q=0.5',
            'Accept-Encoding': 'gzip, deflate, br',
            'Connection': 'keep-alive',
        }
        self.session = requests.Session()

    def clean_price(self, price_str):
        """Extract numeric price from string like '$350,000' or '$350K'"""
        if not price_str:
            return None
        # Remove $, commas, and extract number
        cleaned = re.sub(r'[,$]', '', str(price_str))
        # Handle K (thousands) notation
        if 'K' in cleaned.upper():
            cleaned = cleaned.upper().replace('K', '000')
        # Handle M (millions) notation
        if 'M' in cleaned.upper():
            number = float(cleaned.upper().replace('M', ''))
            return int(number * 1000000)
        try:
            return int(float(cleaned))
        except:
            return None

    def search_property(self, address, city, state, zipcode):
        """Search for property on Zillow and return property data"""
        try:
            # Format address for URL
            search_address = f"{address}, {city}, {state} {zipcode}"
            encoded_address = quote(search_address)

            # Zillow search URL
            url = f"https://www.zillow.com/homes/{encoded_address}_rb/"

            print(f"Searching: {search_address}")

            # Make request
            response = self.session.get(url, headers=self.headers, timeout=10)

            if response.status_code != 200:
                print(f"  [ERROR] Status code: {response.status_code}")
                return None

            soup = BeautifulSoup(response.content, 'html.parser')

            # Try to find the property data in the page
            property_data = {
                'address': search_address,
                'zestimate': None,
                'beds': None,
                'baths': None,
                'sqft': None,
                'year_built': None,
                'lot_size': None,
                'property_type': None,
                'status': None
            }

            # Look for JSON data in script tags (Zillow embeds property data)
            scripts = soup.find_all('script', type='application/json')
            for script in scripts:
                try:
                    data = json.loads(script.string)
                    # Navigate through the JSON structure to find property info
                    if isinstance(data, dict):
                        # Try to extract zestimate and property details
                        # This structure may vary, so we'll try multiple paths
                        self._extract_from_json(data, property_data)
                except:
                    continue

            # Fallback: try to scrape from HTML elements
            if not property_data['zestimate']:
                # Try to find Zestimate in text
                zestimate_elem = soup.find(text=re.compile(r'Zestimate', re.I))
                if zestimate_elem:
                    parent = zestimate_elem.find_parent()
                    if parent:
                        price_text = parent.get_text()
                        price_match = re.search(r'\$[\d,]+', price_text)
                        if price_match:
                            property_data['zestimate'] = self.clean_price(price_match.group())

            # Look for property facts
            facts = soup.find_all(['span', 'div'], class_=re.compile(r'fact|detail|stat', re.I))
            for fact in facts:
                text = fact.get_text().lower()
                if 'bed' in text:
                    beds_match = re.search(r'(\d+)\s*bed', text)
                    if beds_match:
                        property_data['beds'] = int(beds_match.group(1))
                if 'bath' in text:
                    baths_match = re.search(r'(\d+\.?\d*)\s*bath', text)
                    if baths_match:
                        property_data['baths'] = float(baths_match.group(1))
                if 'sqft' in text or 'sq ft' in text:
                    sqft_match = re.search(r'([\d,]+)\s*sq', text)
                    if sqft_match:
                        property_data['sqft'] = self.clean_price(sqft_match.group(1))

            if property_data['zestimate']:
                print(f"  [OK] Found - Zestimate: ${property_data['zestimate']:,}")
            else:
                print(f"  [WARNING] No Zestimate found")

            return property_data

        except Exception as e:
            print(f"  [ERROR] Error: {str(e)}")
            return None

    def _extract_from_json(self, data, property_data, depth=0):
        """Recursively search JSON for property data"""
        if depth > 10:  # Prevent infinite recursion
            return

        if isinstance(data, dict):
            # Look for common keys
            if 'zestimate' in data:
                property_data['zestimate'] = self.clean_price(data.get('zestimate'))
            if 'price' in data and not property_data['zestimate']:
                property_data['zestimate'] = self.clean_price(data.get('price'))
            if 'bedrooms' in data:
                property_data['beds'] = data.get('bedrooms')
            if 'bathrooms' in data:
                property_data['baths'] = data.get('bathrooms')
            if 'livingArea' in data:
                property_data['sqft'] = data.get('livingArea')
            if 'yearBuilt' in data:
                property_data['year_built'] = data.get('yearBuilt')
            if 'lotSize' in data:
                property_data['lot_size'] = data.get('lotSize')
            if 'homeType' in data:
                property_data['property_type'] = data.get('homeType')

            # Recurse into nested dicts
            for value in data.values():
                if isinstance(value, (dict, list)):
                    self._extract_from_json(value, property_data, depth + 1)

        elif isinstance(data, list):
            for item in data:
                if isinstance(item, (dict, list)):
                    self._extract_from_json(item, property_data, depth + 1)

def main():
    # Load the data
    df = pd.read_csv('../data/listing_agents.csv')

    print(f"Loaded {len(df)} properties")
    print("\nStarting Zillow scraping...")
    print("=" * 50)

    scraper = ZillowScraper()
    results = []

    # Process first 5 properties as a test
    test_limit = 5
    print(f"\nTesting with first {test_limit} properties...\n")

    for idx, row in df.head(test_limit).iterrows():
        address = row['Address']
        city = row['City']
        zipcode = row['Zip Code']

        # Extract state from address if present, otherwise assume GA
        state = 'GA'

        print(f"[{idx + 1}/{test_limit}] {address}")

        # Search property
        property_data = scraper.search_property(address, city, state, zipcode)

        if property_data:
            result = {
                'original_index': idx,
                'address': address,
                'city': city,
                'zipcode': zipcode,
                'list_price': row['List Price'],
                **property_data
            }
            results.append(result)

        # Be respectful - rate limit
        time.sleep(2)
        print()

    # Save results
    results_df = pd.DataFrame(results)
    results_df.to_csv('../data/zillow_scraped_test.csv', index=False)

    print("=" * 50)
    print(f"\nCompleted! Scraped {len(results)} properties")
    print(f"Results saved to: data/zillow_scraped_test.csv")
    print("\nSummary:")
    print(f"  Properties with Zestimate: {results_df['zestimate'].notna().sum()}")
    print(f"  Properties with beds/baths: {results_df['beds'].notna().sum()}")
    print(f"  Properties with sqft: {results_df['sqft'].notna().sum()}")

if __name__ == '__main__':
    main()
