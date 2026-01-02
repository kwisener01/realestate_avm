import http.client
import json
import os
from typing import Optional, Dict
from urllib.parse import urlencode, quote

class ZillowAPIService:
    """Service for fetching property data from Zillow via RapidAPI"""

    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or os.getenv('RAPIDAPI_KEY', 'c37d73d47cmsh6834d2a90b8cafbp13a56ajsn7ad8c20790bc')
        self.host = "private-zillow.p.rapidapi.com"

    def _make_request(self, endpoint: str, params: Dict = None) -> Dict:
        """Make a request to the Zillow API"""
        try:
            conn = http.client.HTTPSConnection(self.host)

            headers = {
                'x-rapidapi-key': self.api_key,
                'x-rapidapi-host': self.host
            }

            # Build query string if params provided
            query_string = ""
            if params:
                query_string = "?" + urlencode(params)

            conn.request("GET", f"{endpoint}{query_string}", headers=headers)
            res = conn.getresponse()
            data = res.read()

            return json.loads(data.decode("utf-8"))

        except Exception as e:
            print(f"Error fetching from Zillow API: {e}")
            return None

    def get_property_by_address(self, address: str, city: str = None, state: str = "GA") -> Optional[Dict]:
        """
        Get property details by address

        Args:
            address: Street address (e.g., "500 Gayle Dr")
            city: City name (e.g., "Acworth")
            state: State code (default: "GA")

        Returns:
            Property data including Zestimate, or None if not found
        """
        # Construct full address
        full_address = f"{address}, {city}, {state}" if city else f"{address}, {state}"

        params = {
            "address": full_address
        }

        return self._make_request("/property", params)

    def get_zestimate(self, address: str, city: str = None, state: str = "GA") -> Optional[float]:
        """
        Get Zillow's estimated property value (Zestimate)

        Args:
            address: Street address
            city: City name
            state: State code (default: "GA")

        Returns:
            Zestimate value as float, or None if not available
        """
        property_data = self.get_property_by_address(address, city, state)

        if not property_data:
            return None

        # Extract Zestimate from response
        # The exact structure depends on the API response format
        try:
            # Common Zillow API response patterns:
            if 'zestimate' in property_data:
                return float(property_data['zestimate'])
            elif 'price' in property_data:
                return float(property_data['price'])
            elif 'estimatedValue' in property_data:
                return float(property_data['estimatedValue'])
            else:
                # Try to find value in nested structure
                for key in ['data', 'property', 'result']:
                    if key in property_data and isinstance(property_data[key], dict):
                        if 'zestimate' in property_data[key]:
                            return float(property_data[key]['zestimate'])
                        if 'price' in property_data[key]:
                            return float(property_data[key]['price'])

        except (KeyError, ValueError, TypeError) as e:
            print(f"Error extracting Zestimate: {e}")
            return None

        return None

    def search_properties(self, city: str, state: str = "GA", min_price: int = None, max_price: int = None) -> Optional[list]:
        """
        Search for properties in a specific area

        Args:
            city: City name
            state: State code (default: "GA")
            min_price: Minimum price filter
            max_price: Maximum price filter

        Returns:
            List of properties, or None if error
        """
        params = {
            "city": city,
            "state": state
        }

        if min_price:
            params["minPrice"] = min_price
        if max_price:
            params["maxPrice"] = max_price

        return self._make_request("/search", params)

    def get_property_details(self, zpid: str) -> Optional[Dict]:
        """
        Get detailed property information by Zillow Property ID (zpid)

        Args:
            zpid: Zillow Property ID

        Returns:
            Detailed property data
        """
        params = {"zpid": zpid}
        return self._make_request("/propertyDetails", params)


# Example usage
if __name__ == "__main__":
    zillow = ZillowAPIService()

    # Test: Get property estimate
    print("Testing Zillow API integration...")
    print("\n1. Getting property by address:")

    address = "500 Gayle Dr"
    city = "Acworth"

    property_data = zillow.get_property_by_address(address, city)
    print(f"Property data: {json.dumps(property_data, indent=2)[:500]}...")

    print("\n2. Getting Zestimate:")
    zestimate = zillow.get_zestimate(address, city)
    if zestimate:
        print(f"Zestimate for {address}, {city}: ${zestimate:,.0f}")
    else:
        print("Zestimate not available")
