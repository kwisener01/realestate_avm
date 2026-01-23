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
        return_details: bool = False,
        **kwargs
    ) -> Optional[float] | Optional[Dict]:
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
            return_details: If True, returns dict with zestimate, sqft, beds, baths
            **kwargs: Additional parameters (for compatibility)

        Returns:
            If return_details=False: Zestimate value as float, or None
            If return_details=True: Dict with zestimate, sqft, bedrooms, bathrooms
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

        # Extract Zestimate and property details from response
        try:
            print(f"  Attempting to extract Zestimate from response...")

            zestimate = None
            sqft = None
            beds = None
            baths = None

            # Determine property object from response
            prop = None

            # HasData API: single property result
            if 'property' in property_data and isinstance(property_data['property'], dict):
                prop = property_data['property']
            # HasData API: search results (array of properties)
            elif 'properties' in property_data and len(property_data['properties']) > 0:
                prop = property_data['properties'][0]
                print(f"  Using first property from search results")

            if prop:
                # Extract Zestimate (preferred) or list price
                if 'zestimate' in prop and prop['zestimate']:
                    zestimate = float(prop['zestimate'])
                    print(f"  SUCCESS: Found Zestimate: ${zestimate:,.0f}")
                elif 'price' in prop and prop['price']:
                    zestimate = float(prop['price'])
                    print(f"  SUCCESS: Found list price (no Zestimate): ${zestimate:,.0f}")

                # Extract square footage (try multiple possible field names)
                # Note: area=0 means no data, so we check for value > 0
                for sqft_field in ['area', 'livingArea', 'livingAreaSqFt', 'sqft', 'livingAreaValue', 'finishedSqFt']:
                    if sqft_field in prop and prop[sqft_field] and int(float(prop[sqft_field])) > 0:
                        sqft = int(float(prop[sqft_field]))
                        print(f"  SUCCESS: Found sqft ({sqft_field}): {sqft:,}")
                        break

                # Extract bedrooms
                for bed_field in ['beds', 'bedrooms', 'bedroomCount']:
                    if bed_field in prop and prop[bed_field]:
                        beds = int(prop[bed_field])
                        print(f"  SUCCESS: Found bedrooms: {beds}")
                        break

                # Extract bathrooms
                for bath_field in ['baths', 'bathrooms', 'bathroomCount']:
                    if bath_field in prop and prop[bath_field]:
                        baths = float(prop[bath_field])
                        print(f"  SUCCESS: Found bathrooms: {baths}")
                        break

                # If sqft is missing but we have a Zillow URL, make follow-up call to Property API
                if sqft is None and 'url' in prop and prop['url'] and not zillow_url:
                    print(f"  No sqft from Listing API, trying Property API...")
                    prop_url = prop['url']
                    full_data = self._make_request_by_url(prop_url)
                    if full_data and 'property' in full_data:
                        full_prop = full_data['property']
                        # Get sqft from Property API
                        for sqft_field in ['area', 'livingArea', 'livingAreaSqFt', 'sqft']:
                            if sqft_field in full_prop and full_prop[sqft_field] and int(float(full_prop[sqft_field])) > 0:
                                sqft = int(float(full_prop[sqft_field]))
                                print(f"  SUCCESS: Found sqft from Property API ({sqft_field}): {sqft:,}")
                                break
                        # Also get Zestimate if we only had list price
                        if 'zestimate' in full_prop and full_prop['zestimate']:
                            zestimate = float(full_prop['zestimate'])
                            print(f"  SUCCESS: Found Zestimate from Property API: ${zestimate:,.0f}")

            # Legacy fallback: try root level (for other APIs or response formats)
            if not zestimate:
                if 'zestimate' in property_data:
                    zestimate = float(property_data['zestimate'])
                    print(f"  SUCCESS: Found Zestimate in root: ${zestimate:,.0f}")
                elif 'price' in property_data:
                    zestimate = float(property_data['price'])
                    print(f"  SUCCESS: Found price in root: ${zestimate:,.0f}")

            # If nothing found
            if not zestimate:
                print(f"  WARNING:  WARNING: Could not find Zestimate or price in response")
                print(f"  Response keys: {list(property_data.keys())[:10]}")
                if 'property' in property_data:
                    print(f"  Property keys: {list(property_data['property'].keys())[:15]}")

            # Return based on return_details flag
            if return_details:
                return {
                    'zestimate': zestimate,
                    'sqft': sqft,
                    'bedrooms': beds,
                    'bathrooms': baths
                }
            else:
                return zestimate

        except (KeyError, ValueError, TypeError) as e:
            print(f"  ERROR: ERROR extracting Zestimate: {e}")
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
