import pandas as pd
import time
import re
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.chrome.options import Options
from selenium.common.exceptions import TimeoutException, NoSuchElementException
import json

class MultiSourceScraper:
    def __init__(self, headless=True):
        self.headless = headless
        self.driver = None

    def init_driver(self):
        """Initialize Chrome driver"""
        chrome_options = Options()
        if self.headless:
            chrome_options.add_argument('--headless')
        chrome_options.add_argument('--no-sandbox')
        chrome_options.add_argument('--disable-dev-shm-usage')
        chrome_options.add_argument('--disable-blink-features=AutomationControlled')
        chrome_options.add_argument('--user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36')
        chrome_options.add_experimental_option("excludeSwitches", ["enable-automation"])
        chrome_options.add_experimental_option('useAutomationExtension', False)

        self.driver = webdriver.Chrome(options=chrome_options)
        self.driver.execute_script("Object.defineProperty(navigator, 'webdriver', {get: () => undefined})")

    def clean_price(self, price_str):
        """Extract numeric price from string"""
        if not price_str:
            return None
        cleaned = re.sub(r'[,$]', '', str(price_str))
        if 'K' in cleaned.upper():
            cleaned = cleaned.upper().replace('K', '000')
        if 'M' in cleaned.upper():
            number = float(cleaned.upper().replace('M', ''))
            return int(number * 1000000)
        try:
            return int(float(cleaned))
        except:
            return None

    def scrape_zillow(self, address, city, state, zipcode):
        """Scrape Zillow for property data"""
        try:
            search_url = f"https://www.zillow.com/homes/{address.replace(' ', '-')}-{city.replace(' ', '-')}-{state}-{zipcode}_rb/"

            print(f"  [Zillow] Trying: {search_url}")
            self.driver.get(search_url)
            time.sleep(3)  # Wait for page load

            data = {
                'source': 'zillow',
                'zestimate': None,
                'beds': None,
                'baths': None,
                'sqft': None
            }

            # Try to find Zestimate
            try:
                # Look for price elements
                price_elements = self.driver.find_elements(By.CSS_SELECTOR, '[data-testid="price"], .ds-estimate-value, .zestimate')
                for elem in price_elements:
                    text = elem.text
                    if '$' in text and 'Zestimate' in self.driver.page_source:
                        price = self.clean_price(text)
                        if price:
                            data['zestimate'] = price
                            break

                # Try alternative selectors
                if not data['zestimate']:
                    try:
                        zest_elem = self.driver.find_element(By.XPATH, "//*[contains(text(), 'Zestimate')]/following-sibling::*[contains(text(), '$')]")
                        data['zestimate'] = self.clean_price(zest_elem.text)
                    except:
                        pass

            except Exception as e:
                print(f"    Could not find Zestimate: {e}")

            # Try to find property details
            try:
                facts = self.driver.find_elements(By.CSS_SELECTOR, '[data-testid="bed-bath-beyond"], .ds-bed-bath-living-area')
                for fact in facts:
                    text = fact.text.lower()
                    beds_match = re.search(r'(\d+)\s*bd', text)
                    if beds_match:
                        data['beds'] = int(beds_match.group(1))
                    baths_match = re.search(r'(\d+\.?\d*)\s*ba', text)
                    if baths_match:
                        data['baths'] = float(baths_match.group(1))
                    sqft_match = re.search(r'([\d,]+)\s*sqft', text)
                    if sqft_match:
                        data['sqft'] = self.clean_price(sqft_match.group(1))
            except Exception as e:
                print(f"    Could not find property facts: {e}")

            if data['zestimate']:
                print(f"    [OK] Zestimate: ${data['zestimate']:,}")
                return data
            else:
                print(f"    [FAIL] No data found")
                return None

        except Exception as e:
            print(f"    [ERROR] {e}")
            return None

    def scrape_realtor(self, address, city, state, zipcode):
        """Scrape Realtor.com for property data"""
        try:
            # Format address for Realtor.com URL
            formatted_address = f"{address.replace(' ', '_')},{city.replace(' ', '_')},{state}_{zipcode}"
            search_url = f"https://www.realtor.com/realestateandhomes-detail/{formatted_address}"

            print(f"  [Realtor] Trying: {search_url}")
            self.driver.get(search_url)
            time.sleep(3)

            data = {
                'source': 'realtor',
                'zestimate': None,
                'beds': None,
                'baths': None,
                'sqft': None
            }

            # Try to find estimated value
            try:
                price_elem = self.driver.find_element(By.CSS_SELECTOR, '[data-testid="price"], .price')
                price_text = price_elem.text
                data['zestimate'] = self.clean_price(price_text)
            except:
                # Try alternative selectors
                try:
                    price_elem = self.driver.find_element(By.XPATH, "//*[contains(@class, 'price') or contains(@data-testid, 'price')]")
                    data['zestimate'] = self.clean_price(price_elem.text)
                except:
                    pass

            # Property details
            try:
                details = self.driver.find_elements(By.CSS_SELECTOR, '[data-testid="property-meta"], .property-meta')
                for detail in details:
                    text = detail.text.lower()
                    beds_match = re.search(r'(\d+)\s*bed', text)
                    if beds_match:
                        data['beds'] = int(beds_match.group(1))
                    baths_match = re.search(r'(\d+\.?\d*)\s*bath', text)
                    if baths_match:
                        data['baths'] = float(baths_match.group(1))
                    sqft_match = re.search(r'([\d,]+)\s*sqft', text)
                    if sqft_match:
                        data['sqft'] = self.clean_price(sqft_match.group(1))
            except:
                pass

            if data['zestimate']:
                print(f"    [OK] Price: ${data['zestimate']:,}")
                return data
            else:
                print(f"    [FAIL] No data found")
                return None

        except Exception as e:
            print(f"    [ERROR] {e}")
            return None

    def scrape_redfin(self, address, city, state, zipcode):
        """Scrape Redfin for property data"""
        try:
            # Redfin search
            search_url = f"https://www.redfin.com/city/{city.replace(' ', '-')}/{state.upper()}/{zipcode}"

            print(f"  [Redfin] Trying search: {search_url}")
            self.driver.get(search_url)
            time.sleep(3)

            # Try to find the specific property by searching
            try:
                search_box = self.driver.find_element(By.CSS_SELECTOR, 'input[type="search"], input[placeholder*="address"]')
                search_box.clear()
                search_box.send_keys(f"{address}, {city}, {state} {zipcode}")
                time.sleep(2)
                # Click first result if available
                first_result = self.driver.find_element(By.CSS_SELECTOR, '.SearchBox__option, .search-result')
                first_result.click()
                time.sleep(3)
            except:
                print(f"    [FAIL] Could not search")
                return None

            data = {
                'source': 'redfin',
                'zestimate': None,
                'beds': None,
                'baths': None,
                'sqft': None
            }

            # Try to find price
            try:
                price_elem = self.driver.find_element(By.CSS_SELECTOR, '.statsValue, [data-rf-test-id="abp-price"]')
                data['zestimate'] = self.clean_price(price_elem.text)
            except:
                pass

            # Property details
            try:
                stats = self.driver.find_elements(By.CSS_SELECTOR, '.stat-block, .KeyDetails')
                for stat in stats:
                    text = stat.text.lower()
                    beds_match = re.search(r'(\d+)\s*bed', text)
                    if beds_match:
                        data['beds'] = int(beds_match.group(1))
                    baths_match = re.search(r'(\d+\.?\d*)\s*bath', text)
                    if baths_match:
                        data['baths'] = float(baths_match.group(1))
                    sqft_match = re.search(r'([\d,]+)\s*sq', text)
                    if sqft_match:
                        data['sqft'] = self.clean_price(sqft_match.group(1))
            except:
                pass

            if data['zestimate']:
                print(f"    [OK] Price: ${data['zestimate']:,}")
                return data
            else:
                print(f"    [FAIL] No data found")
                return None

        except Exception as e:
            print(f"    [ERROR] {e}")
            return None

    def scrape_property(self, address, city, state, zipcode):
        """Try multiple sources to get property data"""
        print(f"\nSearching: {address}, {city}, {state} {zipcode}")

        # Try Zillow first
        data = self.scrape_zillow(address, city, state, zipcode)
        if data and data['zestimate']:
            return data

        # Try Realtor.com
        data = self.scrape_realtor(address, city, state, zipcode)
        if data and data['zestimate']:
            return data

        # Try Redfin
        data = self.scrape_redfin(address, city, state, zipcode)
        if data and data['zestimate']:
            return data

        print("  [FAIL] No data from any source")
        return None

    def close(self):
        if self.driver:
            self.driver.quit()

def main():
    # Load data
    df = pd.read_csv('../data/listing_agents.csv')
    print(f"Loaded {len(df)} properties\n")

    # Initialize scraper
    scraper = MultiSourceScraper(headless=False)  # Set to False to see browser
    scraper.init_driver()

    results = []
    test_limit = 10  # Test with first 10 properties

    print(f"Testing with first {test_limit} properties...")
    print("=" * 70)

    try:
        for idx, row in df.head(test_limit).iterrows():
            address = row['Address']
            city = row['City']
            zipcode = str(row['Zip Code'])
            state = 'GA'

            print(f"\n[{idx + 1}/{test_limit}]", end=" ")

            # Scrape property
            property_data = scraper.scrape_property(address, city, state, zipcode)

            if property_data:
                result = {
                    'index': idx,
                    'address': address,
                    'city': city,
                    'zipcode': zipcode,
                    'list_price': row['List Price'],
                    **property_data
                }
                results.append(result)

            # Rate limiting
            time.sleep(2)

    finally:
        scraper.close()

    # Save results
    if results:
        results_df = pd.DataFrame(results)
        results_df.to_csv('../data/multi_source_scraped.csv', index=False)

        print("\n" + "=" * 70)
        print(f"\nCompleted! Scraped {len(results)}/{test_limit} properties")
        print(f"Success rate: {len(results)/test_limit*100:.1f}%")
        print(f"\nResults saved to: data/multi_source_scraped.csv")
        print("\nSummary:")
        print(f"  Properties with price: {results_df['zestimate'].notna().sum()}")
        print(f"  Properties with beds: {results_df['beds'].notna().sum()}")
        print(f"  Properties with baths: {results_df['baths'].notna().sum()}")
        print(f"  Properties with sqft: {results_df['sqft'].notna().sum()}")

        # Show sources used
        print(f"\nSources:")
        for source in results_df['source'].value_counts().items():
            print(f"  {source[0]}: {source[1]}")
    else:
        print("\n[ERROR] No properties scraped successfully")

if __name__ == '__main__':
    main()
