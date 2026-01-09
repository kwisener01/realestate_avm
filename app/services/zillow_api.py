import requests
import json
import os
from typing import Optional, Dict
from urllib.parse import quote

class ZillowAPIService:
    """Service for fetching property data from Zillow scraper API (HasData.com)"""

    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or os.getenv('ZILLOW_API_KEY', '16c519c0-be1f-4e1d-9ff4-76702fc1f43a')
        # Use Listing API - works with address search and returns Zestimate directly
        self.base_url = "https://api.hasdata.com/scrape/zillow/listing"

    def _construct_zillow_url(self, address: str, city: str, state: str, zipcode: str = None) -> str:
        """
        Construct a Zillow property URL from address components.
        Example: https://www.zillow.com/homedetails/301-E-79th-St-APT-23S-New-York-NY-10075_zpid/

        Note: Without ZPID, this may not work perfectly. The API works best with full URLs
        including the Zillow Property ID (zpid), but we attempt to construct a search URL.
        """
        # Clean and format address components
        address_clean = address.replace(' ', '-').replace(',', '').replace('.', '')
        city_clean = city.replace(' ', '-')

        # Construct full address string
        if zipcode:
            full_address = f"{address_clean}-{city_clean}-{state}-{zipcode}"
        else:
            full_address = f"{address_clean}-{city_clean}-{state}"

        # Use /homedetails/ format which is what HasData expects
        zillow_url = f"https://www.zillow.com/homedetails/{full_address}/"
        return zillow_url

    def _make_request_by_url(self, zillow_url: str) -> Optional[Dict]:
        """Make a request to the Property API using a Zillow URL with ZPID"""
        property_api_url = "https://api.hasdata.com/scrape/zillow/property"

        try:
            headers = {
                'Content-Type': 'application/json',
                'x-api-key': self.api_key
            }

            params = {
                'url': zillow_url
            }

            print(f"  Making API request to Property API with URL")
            print(f"  Zillow URL parameter: {zillow_url}")

            response = requests.get(
                property_api_url,
                headers=headers,
                params=params,
                timeout=15
            )

            print(f"  API Response Status: {response.status_code}")

            if response.status_code == 200:
                data = response.json()
                print(f"  API Response Keys: {list(data.keys()) if isinstance(data, dict) else 'Not a dict'}")
                return data
            else:
                print(f"  ERROR: API status {response.status_code}")
                return None

        except Exception as e:
            print(f"  ERROR: {type(e).__name__}: {e}")
            return None

    def _make_request_by_address(self, address: str, city: str, state: str, zipcode: str = None) -> Optional[Dict]:
        """Make a request to the Listing API using address search (works without ZPID!)"""
        try:
            headers = {
                'Content-Type': 'application/json',
                'x-api-key': self.api_key
            }

            # Build search keyword from address components
            if zipcode:
                search_keyword = f"{address}, {city}, {state} {zipcode}"
            else:
                search_keyword = f"{address}, {city}, {state}"

            params = {
                'keyword': search_keyword,
                'type': 'forSale'
            }

            print(f"  Making API request to Listing API")
            print(f"  Search keyword: {search_keyword}")

            response = requests.get(
                self.base_url,
                headers=headers,
                params=params,
                timeout=15
            )

            print(f"  API Response Status: {response.status_code}")

            if response.status_code == 200:
                data = response.json()
                print(f"  API Response Keys: {list(data.keys()) if isinstance(data, dict) else 'Not a dict'}")
                # Listing API returns property data directly when searching by specific address
                return data
            else:
                print(f"  ERROR: API status {response.status_code}")
                print(f"  Response: {response.text[:300]}")
                return None

        except requests.exceptions.Timeout:
            print(f"  ERROR: API timeout after 15s")
            return None
        except Exception as e:
            print(f"  ERROR: {type(e).__name__}: {e}")
            return None

    def get_property_by_address(self, address: str, city: str = None, state: str = "GA", zipcode: str = None) -> Optional[Dict]:
        """
        Get property details by address using Listing API

        Args:
            address: Street address (e.g., "3541 Santa Leta Dr")
            city: City name (e.g., "Ellenwood")
            state: State code (default: "GA")
            zipcode: ZIP code (optional, improves accuracy)

        Returns:
            Property data including Zestimate, or None if not found
        """
        if not city:
            print("City is required for address search")
            return None

        # Use Listing API with address search (works without ZPID!)
        return self._make_request_by_address(address, city, state, zipcode)

    def get_value_estimate(
        self,
        address: str,
        city: str = None,
        state: str = "GA",
        zipcode: str = None,
        property_type: str = "Single Family",
        bedrooms: Optional[int] = None,
        bathrooms: Optional[float] = None,
        square_footage: Optional[int] = None,
        zillow_url: str = None,
        **kwargs
    ) -> Optional[float]:
        """
        Get property value estimate (Zestimate) - compatible with RentCast interface

        Args:
            address: Street address
            city: City name
            state: State code (default: "GA")
            zipcode: ZIP code (optional)
            property_type: Property type (not used by Zillow scraper)
            bedrooms: Number of bedrooms (not used by Zillow scraper)
            bathrooms: Number of bathrooms (not used by Zillow scraper)
            square_footage: Square footage (not used by Zillow scraper)
            zillow_url: Direct Zillow URL (with ZPID) - if provided, uses this instead of constructing
            **kwargs: Additional parameters (for compatibility)

        Returns:
            Zestimate value as float, or None if not available
        """
        # If zillow_url is provided directly, use Property API with URL
        if zillow_url and zillow_url.strip():
            property_data = self._make_request_by_url(zillow_url.strip())
        else:
            # Use Listing API with address search
            property_data = self.get_property_by_address(address, city, state, zipcode)

        if not property_data:
            print(f"  No property data returned from API")
            return None

        # Extract Zestimate from response
        try:
            print(f"  Attempting to extract Zestimate from response...")

            # HasData API specific structure: data is nested under 'property' key
            if 'property' in property_data and isinstance(property_data['property'], dict):
                prop = property_data['property']

                # First try to get Zestimate (preferred)
                if 'zestimate' in prop and prop['zestimate']:
                    value = float(prop['zestimate'])
                    print(f"  ✅ Found Zestimate in property: ${value:,.0f}")
                    return value

                # Fall back to list price if no Zestimate
                if 'price' in prop and prop['price']:
                    value = float(prop['price'])
                    print(f"  ✅ Found list price in property (no Zestimate): ${value:,.0f}")
                    return value

            # Legacy fallback: try root level (for other APIs or response formats)
            if 'zestimate' in property_data:
                value = float(property_data['zestimate'])
                print(f"  ✅ Found Zestimate in root: ${value:,.0f}")
                return value
            elif 'price' in property_data:
                value = float(property_data['price'])
                print(f"  ✅ Found price in root: ${value:,.0f}")
                return value

            # If nothing found
            print(f"  ⚠️  WARNING: Could not find Zestimate or price in response")
            print(f"  Response keys: {list(property_data.keys())[:10]}")
            if 'property' in property_data:
                print(f"  Property keys: {list(property_data['property'].keys())[:15]}")

        except (KeyError, ValueError, TypeError) as e:
            print(f"  ❌ ERROR extracting Zestimate: {e}")
            return None

        return None

    def get_property_data(
        self,
        address: str,
        city: str = None,
        state: str = "GA",
        zipcode: str = None,
        **kwargs
    ) -> Optional[Dict]:
        """
        Get full property data - compatible with RentCast interface

        Args:
            address: Street address
            city: City name
            state: State code (default: "GA")
            zipcode: ZIP code (optional)
            **kwargs: Additional parameters (for compatibility)

        Returns:
            Complete property data dict, or None if not available
        """
        return self.get_property_by_address(address, city, state, zipcode)

    def get_comparables(self, address: str, city: str = None, state: str = "GA", zipcode: str = None) -> Optional[list]:
        """
        Get comparable properties (if available in response)

        Args:
            address: Street address
            city: City name
            state: State code (default: "GA")
            zipcode: ZIP code (optional)

        Returns:
            List of comparable properties, or None if not available
        """
        data = self.get_property_data(address, city, state, zipcode)

        if data:
            # Try to extract comps from various possible locations
            if 'comparables' in data:
                return data['comparables']
            elif 'comps' in data:
                return data['comps']
            elif 'nearbyHomes' in data:
                return data['nearbyHomes']

        return None


# Example usage
if __name__ == "__main__":
    zillow = ZillowAPIService()

    # Test: Get property estimate
    print("Testing Zillow API integration...")
    print("\n1. Getting property by address:")

    address = "500 Gayle Dr"
    city = "Acworth"
    state = "GA"

    property_data = zillow.get_property_by_address(address, city, state)
    if property_data:
        print(f"Property data: {json.dumps(property_data, indent=2)[:500]}...")
    else:
        print("Property data not available")

    print("\n2. Getting value estimate:")
    estimate = zillow.get_value_estimate(address, city, state)
    if estimate:
        print(f"Zestimate for {address}, {city}: ${estimate:,.0f}")
    else:
        print("Zestimate not available")
