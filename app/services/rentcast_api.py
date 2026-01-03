import requests
import os
from typing import Optional, Dict


class RentcastAPIService:
    """Service for fetching property value estimates from Rentcast API"""

    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or os.getenv('RENTCAST_API_KEY', '09941f5d40cc4eb896e8322c691ed644')
        self.base_url = "https://api.rentcast.io/v1"

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
        max_radius: float = 3.0,
        days_old: int = 180,
        comp_count: int = 20
    ) -> Optional[float]:
        """
        Get property value estimate from Rentcast with improved accuracy

        Args:
            address: Street address
            city: City name
            state: State code (default: "GA")
            zipcode: ZIP code (optional)
            property_type: Property type (default: "Single Family")
            bedrooms: Number of bedrooms (improves accuracy)
            bathrooms: Number of bathrooms (improves accuracy)
            square_footage: Square footage (improves accuracy)
            max_radius: Maximum distance for comps in miles (default: 3.0)
            days_old: Maximum age of comps in days (default: 180)
            comp_count: Number of comps to use (default: 20)

        Returns:
            Property value estimate as float, or None if not available
        """
        try:
            # Construct full address
            full_address = f"{address}, {city}, {state}"
            if zipcode:
                full_address += f" {zipcode}"

            headers = {
                'X-Api-Key': self.api_key,
                'accept': 'application/json'
            }

            params = {
                'address': full_address,
                'propertyType': property_type,
                'maxRadius': max_radius,
                'daysOld': days_old,
                'compCount': comp_count
            }

            # Add property attributes if provided (increases accuracy)
            if bedrooms is not None:
                params['bedrooms'] = bedrooms
            if bathrooms is not None:
                params['bathrooms'] = bathrooms
            if square_footage is not None and square_footage > 0:
                params['squareFootage'] = square_footage

            response = requests.get(
                f"{self.base_url}/avm/value",
                headers=headers,
                params=params,
                timeout=10
            )

            if response.status_code == 200:
                data = response.json()

                # Extract value estimate from response
                if 'price' in data:
                    return float(data['price'])
                elif 'value' in data:
                    return float(data['value'])
                elif 'estimate' in data:
                    return float(data['estimate'])

                return None
            else:
                print(f"Rentcast API error: {response.status_code} - {response.text[:200]}")
                return None

        except Exception as e:
            print(f"Error fetching from Rentcast API: {e}")
            return None

    def get_property_data(
        self,
        address: str,
        city: str = None,
        state: str = "GA",
        zipcode: str = None,
        property_type: str = "Single Family",
        bedrooms: Optional[int] = None,
        bathrooms: Optional[float] = None,
        square_footage: Optional[int] = None,
        max_radius: float = 3.0,
        days_old: int = 180,
        comp_count: int = 20
    ) -> Optional[Dict]:
        """
        Get full property data from Rentcast including comps with improved accuracy

        Args:
            address: Street address
            city: City name
            state: State code (default: "GA")
            zipcode: ZIP code (optional)
            property_type: Property type (default: "Single Family")
            bedrooms: Number of bedrooms (improves accuracy)
            bathrooms: Number of bathrooms (improves accuracy)
            square_footage: Square footage (improves accuracy)
            max_radius: Maximum distance for comps in miles (default: 3.0)
            days_old: Maximum age of comps in days (default: 180)
            comp_count: Number of comps to use (default: 20)

        Returns:
            Complete property data dict with value, comparables, etc., or None if not available
        """
        try:
            # Construct full address
            full_address = f"{address}, {city}, {state}"
            if zipcode:
                full_address += f" {zipcode}"

            headers = {
                'X-Api-Key': self.api_key,
                'accept': 'application/json'
            }

            params = {
                'address': full_address,
                'propertyType': property_type,
                'maxRadius': max_radius,
                'daysOld': days_old,
                'compCount': comp_count
            }

            # Add property attributes if provided (increases accuracy)
            if bedrooms is not None:
                params['bedrooms'] = bedrooms
            if bathrooms is not None:
                params['bathrooms'] = bathrooms
            if square_footage is not None and square_footage > 0:
                params['squareFootage'] = square_footage

            response = requests.get(
                f"{self.base_url}/avm/value",
                headers=headers,
                params=params,
                timeout=10
            )

            if response.status_code == 200:
                return response.json()
            else:
                return None

        except Exception as e:
            print(f"Error fetching from Rentcast API: {e}")
            return None

    def get_comparables(self, address: str, city: str = None, state: str = "GA", zipcode: str = None) -> Optional[list]:
        """
        Get comparable properties used for valuation

        Args:
            address: Street address
            city: City name
            state: State code (default: "GA")
            zipcode: ZIP code (optional)

        Returns:
            List of comparable properties sorted by correlation, or None if not available
        """
        data = self.get_property_data(address, city, state, zipcode)

        if data and 'comparables' in data:
            return data['comparables']

        return None


# Example usage
if __name__ == "__main__":
    rentcast = RentcastAPIService()

    # Test: Get property value estimate
    print("Testing Rentcast API integration...")
    print("\n1. Getting value estimate:")

    address = "500 Gayle Dr"
    city = "Acworth"
    state = "GA"

    value = rentcast.get_value_estimate(address, city, state)
    if value:
        print(f"Value estimate for {address}, {city}: ${value:,.0f}")
    else:
        print("Value estimate not available")

    print("\n2. Getting full property data:")
    data = rentcast.get_property_data(address, city, state)
    if data:
        print(f"Property data: {data}")
    else:
        print("Property data not available")
