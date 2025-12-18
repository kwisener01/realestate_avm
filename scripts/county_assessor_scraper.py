import pandas as pd
import requests
from bs4 import BeautifulSoup
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.chrome.options import Options
import time
import re

class CountyAssessorScraper:
    """
    Scraper for Georgia county tax assessor websites to get sales comps
    """

    def __init__(self, headless=True):
        self.headless = headless
        self.driver = None

        # County assessor URLs (Top Georgia counties)
        self.county_urls = {
            'Dekalb - GA': 'https://www.qpublic.net/ga/dekalb/',
            'Fulton - GA': 'https://www.qpublic.net/ga/fulton/',
            'Clayton - GA': 'https://www.claytoncountyga.gov/government/tax-assessors/property-search',
            'Gwinnett - GA': 'https://www.gwinnettassessor.com/',
            'Cobb - GA': 'https://www.cobbtax.org/',
            'Henry - GA': 'https://www.qpublic.net/ga/henry/',
            'Douglas - GA': 'https://www.qpublic.net/ga/douglas/',
            'Rockdale - GA': 'https://www.qpublic.net/ga/rockdale/',
            'Newton - GA': 'https://www.qpublic.net/ga/newton/',
            'Paulding - GA': 'https://www.qpublic.net/ga/paulding/',
        }

    def init_driver(self):
        """Initialize Chrome driver"""
        chrome_options = Options()
        if self.headless:
            chrome_options.add_argument('--headless')
        chrome_options.add_argument('--no-sandbox')
        chrome_options.add_argument('--disable-dev-shm-usage')
        chrome_options.add_argument('--user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36')

        self.driver = webdriver.Chrome(options=chrome_options)

    def clean_price(self, price_str):
        """Extract numeric price from string"""
        if not price_str:
            return None
        cleaned = re.sub(r'[,$]', '', str(price_str))
        try:
            return int(float(cleaned))
        except:
            return None

    def scrape_qpublic_county(self, address, county):
        """
        Scrape qpublic.net counties (most Georgia counties use this)
        DeKalb, Fulton, Henry, Douglas, Rockdale, Newton, Paulding all use qpublic
        """
        try:
            base_url = self.county_urls.get(county)
            if not base_url:
                print(f"  [SKIP] No scraper for {county}")
                return None

            if 'qpublic.net' not in base_url:
                return None  # Different scraper needed

            print(f"  [QPublic] Searching {county}...")

            # Navigate to search page
            search_url = f"{base_url}search.html"
            self.driver.get(search_url)
            time.sleep(2)

            # Find and fill address search box
            try:
                # Try different search field IDs
                search_field = None
                for field_id in ['address_quick', 'searchInput', 'quick_search']:
                    try:
                        search_field = self.driver.find_element(By.ID, field_id)
                        break
                    except:
                        continue

                if not search_field:
                    # Try by name
                    search_field = self.driver.find_element(By.NAME, 'address')

                search_field.clear()
                search_field.send_keys(address)

                # Submit search
                search_field.submit()
                time.sleep(3)

                # Check if we got results
                # Look for property details or sales history
                sales_data = []

                # Try to find sales history table
                try:
                    # Look for "Sales History" or similar
                    sales_section = self.driver.find_elements(By.XPATH,
                        "//*[contains(text(), 'Sales') or contains(text(), 'Transfer')]")

                    if sales_section:
                        # Find table with sales data
                        tables = self.driver.find_elements(By.TAG_NAME, 'table')
                        for table in tables:
                            rows = table.find_elements(By.TAG_NAME, 'tr')
                            for row in rows:
                                cells = row.find_elements(By.TAG_NAME, 'td')
                                if len(cells) >= 2:
                                    # Try to extract date and price
                                    for i, cell in enumerate(cells):
                                        text = cell.text
                                        # Look for dollar amounts
                                        if '$' in text:
                                            price = self.clean_price(text)
                                            if price and price > 10000:  # Valid sale price
                                                sales_data.append({
                                                    'sale_price': price,
                                                    'raw_data': text
                                                })
                except Exception as e:
                    print(f"    Could not parse sales: {e}")

                # Also try to get property value/assessment
                assessed_value = None
                try:
                    value_elements = self.driver.find_elements(By.XPATH,
                        "//*[contains(text(), 'Market Value') or contains(text(), 'Assessed')]")

                    for elem in value_elements:
                        parent = elem.find_element(By.XPATH, '..')
                        text = parent.text
                        if '$' in text:
                            assessed_value = self.clean_price(text)
                            break
                except:
                    pass

                result = {
                    'county': county,
                    'address': address,
                    'sales_history': sales_data,
                    'assessed_value': assessed_value,
                    'comp_count': len(sales_data)
                }

                if sales_data or assessed_value:
                    print(f"    [OK] Found {len(sales_data)} sales, assessed: ${assessed_value or 0:,}")
                    return result
                else:
                    print(f"    [FAIL] No data found")
                    return None

            except Exception as e:
                print(f"    [ERROR] Search failed: {e}")
                return None

        except Exception as e:
            print(f"  [ERROR] {e}")
            return None

    def scrape_property(self, address, county):
        """Main scraper that routes to appropriate county scraper"""

        # Most Georgia counties use qpublic.net
        if 'qpublic.net' in self.county_urls.get(county, ''):
            return self.scrape_qpublic_county(address, county)
        else:
            # For non-qpublic counties, we'd need custom scrapers
            print(f"  [INFO] {county} requires custom scraper (not yet implemented)")
            return None

    def close(self):
        if self.driver:
            self.driver.quit()

def main():
    # Load data
    df = pd.read_csv('../data/listing_agents.csv')

    print("=" * 80)
    print("COUNTY ASSESSOR SCRAPER - SALES COMPS")
    print("=" * 80)

    # Focus on top counties for testing
    top_counties = ['Dekalb - GA', 'Fulton - GA', 'Henry - GA', 'Douglas - GA', 'Rockdale - GA']
    df_test = df[df['County'].isin(top_counties)].head(5)  # Test with 5 properties

    print(f"\nTesting with {len(df_test)} properties from top counties")
    print(f"Counties: {', '.join(top_counties)}\n")

    # Initialize scraper
    scraper = CountyAssessorScraper(headless=False)
    scraper.init_driver()

    results = []

    try:
        for idx, row in df_test.iterrows():
            address = row['Address']
            county = row['County']

            print(f"\n[{idx + 1}/{len(df_test)}] {address}, {county}")

            result = scraper.scrape_property(address, county)

            if result:
                result['original_index'] = idx
                result['list_price'] = row['List Price']
                results.append(result)

            time.sleep(3)  # Be respectful

    finally:
        scraper.close()

    # Save results
    if results:
        df_results = pd.DataFrame(results)
        df_results.to_csv('../data/county_comps_test.csv', index=False)

        print("\n" + "=" * 80)
        print(f"RESULTS: Found comps for {len(results)}/{len(df_test)} properties")
        print(f"Saved to: data/county_comps_test.csv")
        print("=" * 80)
    else:
        print("\n[WARNING] No comps found")

if __name__ == '__main__':
    main()
