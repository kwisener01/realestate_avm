import requests
import json
import os
from typing import Optional, Dict
from urllib.parse import quote

class ZillowAPIService:
    """Service for fetching property data from Zillow scraper API (api.data.com)"""

    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or os.getenv('ZILLOW_API_KEY', '16c519c0-be1f-4e1d-9ff4-76702fc1f43a')
        self.base_url = "https://api.data.com/scrape/zillow/property"

    def _construct_zillow_url(self, address: str, city: str, state: str, zipcode: str = None) -> str:
        """
        Construct a Zillow property URL from address components.
        Example: https://www.zillow.com/homes/301-E-79th-St-APT-23S-New-York-NY-10075
        """
        # Clean and format address components
        address_clean = address.replace(' ', '-').replace(',', '')
        city_clean = city.replace(' ', '-')

        # Construct full address string
        if zipcode:
            full_address = f"{address_clean}-{city_clean}-{state}-{zipcode}"
        else:
            full_address = f"{address_clean}-{city_clean}-{state}"

        zillow_url = f"https://www.zillow.com/homes/{full_address}"
        return zillow_url

    def _make_request(self, zillow_url: str) -> Optional[Dict]:
        """Make a request to the Zillow scraper API"""
        try:
            headers = {
                'Content-Type': 'application/json',
                'x-api-key': self.api_key
            }

            params = {
                'url': zillow_url
            }

            response = requests.get(
                self.base_url,
                headers=headers,
                params=params,
                timeout=5  # Reduced from 30s - fail fast if API isn't responding
            )

            if response.status_code == 200:
                return response.json()
            else:
                print(f"Zillow API error: {response.status_code} - {response.text[:200]}")
                return None

        except Exception as e:
            print(f"Error fetching from Zillow API: {e}")
            return None

    def get_property_by_address(self, address: str, city: str = None, state: str = "GA", zipcode: str = None) -> Optional[Dict]:
        """
        Get property details by address

        Args:
            address: Street address (e.g., "500 Gayle Dr")
            city: City name (e.g., "Acworth")
            state: State code (default: "GA")
            zipcode: ZIP code (optional, improves accuracy)

        Returns:
            Property data including Zestimate, or None if not found
        """
        if not city:
            print("City is required for Zillow URL construction")
            return None

        # Construct Zillow URL
        zillow_url = self._construct_zillow_url(address, city, state, zipcode)
        print(f"Fetching property from: {zillow_url}")

        return self._make_request(zillow_url)

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
            **kwargs: Additional parameters (for compatibility)

        Returns:
            Zestimate value as float, or None if not available
        """
        property_data = self.get_property_by_address(address, city, state, zipcode)

        if not property_data:
            return None

        # Extract Zestimate from response
        try:
            # Common Zillow scraper response patterns:
            if 'zestimate' in property_data:
                return float(property_data['zestimate'])
            elif 'price' in property_data:
                return float(property_data['price'])
            elif 'estimatedValue' in property_data:
                return float(property_data['estimatedValue'])
            else:
                # Try to find value in nested structure
                for key in ['data', 'property', 'result', 'propertyDetails']:
                    if key in property_data and isinstance(property_data[key], dict):
                        nested = property_data[key]
                        if 'zestimate' in nested:
                            return float(nested['zestimate'])
                        if 'price' in nested:
                            return float(nested['price'])
                        if 'estimatedValue' in nested:
                            return float(nested['estimatedValue'])

        except (KeyError, ValueError, TypeError) as e:
            print(f"Error extracting Zestimate: {e}")
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
