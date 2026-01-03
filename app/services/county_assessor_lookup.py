"""
County Assessor Record Lookup Service
Fetches property details (sqft, year built, etc.) from county assessor records
"""
import requests
from bs4 import BeautifulSoup
import re
from typing import Optional, Dict
import time


class CountyAssessorLookup:
    """
    Lookup property details from Georgia county assessor records
    """

    def __init__(self):
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        })

    def lookup_property(self, address: str, city: str, county: str, parcel_number: Optional[str] = None) -> Dict:
        """
        Lookup property details from county assessor records

        Args:
            address: Street address
            city: City name
            county: County name (DeKalb, Fulton, Cobb, Gwinnett, Clayton)
            parcel_number: Parcel/APN number (optional, helps with lookup)

        Returns:
            Dict with property details: sqft, year_built, bedrooms, bathrooms, etc.
        """
        county_lower = county.lower().strip()

        try:
            if 'dekalb' in county_lower:
                return self._lookup_dekalb(address, parcel_number)
            elif 'fulton' in county_lower:
                return self._lookup_fulton(address, parcel_number)
            elif 'cobb' in county_lower:
                return self._lookup_cobb(address, parcel_number)
            elif 'gwinnett' in county_lower:
                return self._lookup_gwinnett(address, parcel_number)
            elif 'clayton' in county_lower:
                return self._lookup_clayton(address, parcel_number)
            else:
                return {'error': f'County {county} not supported yet'}
        except Exception as e:
            return {'error': str(e)}

    def _lookup_dekalb(self, address: str, parcel_number: Optional[str] = None) -> Dict:
        """
        Lookup property from DeKalb County assessor
        Uses qPublic.net system
        """
        try:
            if not parcel_number:
                return {'error': 'Parcel number required for DeKalb County'}

            # DeKalb County uses qPublic - Schneider Geospatial
            # Format: https://qpublic.schneidercorp.com/Application.aspx?AppID=775&LayerID=14445&PageTypeID=4&PageID=5554&KeyValue=PARCEL

            # Clean parcel number
            parcel_clean = parcel_number.replace('-', '').replace(' ', '').upper()

            # qPublic URL for DeKalb
            base_url = "https://qpublic.schneidercorp.com/Application.aspx"
            params = {
                'AppID': '775',
                'LayerID': '14445',
                'PageTypeID': '4',
                'PageID': '5554',
                'KeyValue': parcel_clean
            }

            response = self.session.get(base_url, params=params, timeout=15)

            if response.status_code == 200:
                soup = BeautifulSoup(response.text, 'html.parser')

                result = {
                    'source': 'DeKalb County (qPublic)',
                    'parcel_number': parcel_number
                }

                # qPublic uses tables with class "DataletData"
                # Look for Living Area or Square Feet
                for table in soup.find_all('table', class_='DataletData'):
                    rows = table.find_all('tr')
                    for row in rows:
                        cells = row.find_all('td')
                        if len(cells) >= 2:
                            label = cells[0].get_text().strip().lower()
                            value = cells[1].get_text().strip()

                            if 'living area' in label or 'heated area' in label or 'square feet' in label:
                                # Extract numeric value
                                sqft_match = re.search(r'(\d{1,5})', value.replace(',', ''))
                                if sqft_match:
                                    result['sqft'] = int(sqft_match.group(1))
                                    return result

                # Fallback: search entire page text
                page_text = soup.get_text()
                patterns = [
                    r'Living Area[:\s]+(\d{1,5})',
                    r'Heated Area[:\s]+(\d{1,5})',
                    r'Square Feet[:\s]+(\d{1,5})',
                ]

                for pattern in patterns:
                    match = re.search(pattern, page_text.replace(',', ''), re.IGNORECASE)
                    if match:
                        result['sqft'] = int(match.group(1))
                        return result

                return {'error': 'Sqft not found on DeKalb County page'}
            else:
                return {'error': f'DeKalb lookup failed: HTTP {response.status_code}'}

        except Exception as e:
            return {'error': f'DeKalb error: {str(e)}'}

    def _lookup_fulton(self, address: str, parcel_number: Optional[str] = None) -> Dict:
        """
        Lookup property from Fulton County assessor
        Uses qPublic.net system (AppID=897)
        """
        try:
            if not parcel_number:
                return {'error': 'Parcel number required for Fulton County'}

            # Fulton also uses qPublic
            parcel_clean = parcel_number.replace('-', '').replace(' ', '').upper()

            base_url = "https://qpublic.schneidercorp.com/Application.aspx"
            params = {
                'AppID': '897',  # Fulton County AppID
                'LayerID': '17934',
                'PageTypeID': '4',
                'PageID': '7510',
                'KeyValue': parcel_clean
            }

            response = self.session.get(base_url, params=params, timeout=15)

            if response.status_code == 200:
                soup = BeautifulSoup(response.text, 'html.parser')

                result = {
                    'source': 'Fulton County (qPublic)',
                    'parcel_number': parcel_number
                }

                # Same qPublic parsing as DeKalb
                for table in soup.find_all('table', class_='DataletData'):
                    rows = table.find_all('tr')
                    for row in rows:
                        cells = row.find_all('td')
                        if len(cells) >= 2:
                            label = cells[0].get_text().strip().lower()
                            value = cells[1].get_text().strip()

                            if 'living area' in label or 'heated area' in label or 'square feet' in label or 'finished sq' in label:
                                sqft_match = re.search(r'(\d{1,5})', value.replace(',', ''))
                                if sqft_match:
                                    result['sqft'] = int(sqft_match.group(1))
                                    return result

                return {'error': 'Sqft not found on Fulton County page'}
            else:
                return {'error': f'Fulton lookup failed: HTTP {response.status_code}'}

        except Exception as e:
            return {'error': f'Fulton error: {str(e)}'}

    def _lookup_cobb(self, address: str, parcel_number: Optional[str] = None) -> Dict:
        """
        Lookup property from Cobb County assessor
        Uses qPublic.net system (AppID=1051)
        """
        try:
            if not parcel_number:
                return {'error': 'Parcel number required for Cobb County'}

            # Cobb also uses qPublic
            parcel_clean = parcel_number.replace('-', '').replace(' ', '').upper()

            base_url = "https://qpublic.schneidercorp.com/Application.aspx"
            params = {
                'AppID': '1051',  # Cobb County AppID
                'LayerID': '23951',
                'PageTypeID': '4',
                'PageID': '9967',
                'KeyValue': parcel_clean
            }

            response = self.session.get(base_url, params=params, timeout=15)

            if response.status_code == 200:
                soup = BeautifulSoup(response.text, 'html.parser')

                result = {
                    'source': 'Cobb County (qPublic)',
                    'parcel_number': parcel_number
                }

                # Same qPublic parsing as DeKalb/Fulton
                for table in soup.find_all('table', class_='DataletData'):
                    rows = table.find_all('tr')
                    for row in rows:
                        cells = row.find_all('td')
                        if len(cells) >= 2:
                            label = cells[0].get_text().strip().lower()
                            value = cells[1].get_text().strip()

                            if 'living area' in label or 'heated area' in label or 'square feet' in label or 'finished sq' in label:
                                sqft_match = re.search(r'(\d{1,5})', value.replace(',', ''))
                                if sqft_match:
                                    result['sqft'] = int(sqft_match.group(1))
                                    return result

                return {'error': 'Sqft not found on Cobb County page'}
            else:
                return {'error': f'Cobb lookup failed: HTTP {response.status_code}'}

        except Exception as e:
            return {'error': f'Cobb County error: {str(e)}'}

    def _lookup_gwinnett(self, address: str, parcel_number: Optional[str] = None) -> Dict:
        """
        Lookup property from Gwinnett County assessor
        Uses qPublic.net system (AppID=698 for tax records)
        """
        try:
            if not parcel_number:
                return {'error': 'Parcel number required for Gwinnett County'}

            # Gwinnett also uses qPublic
            parcel_clean = parcel_number.replace('-', '').replace(' ', '').upper()

            base_url = "https://qpublic.schneidercorp.com/Application.aspx"
            params = {
                'AppID': '698',  # Gwinnett County AppID (tax records)
                'LayerID': '11403',
                'PageTypeID': '4',
                'PageID': '5857',
                'KeyValue': parcel_clean
            }

            response = self.session.get(base_url, params=params, timeout=15)

            if response.status_code == 200:
                soup = BeautifulSoup(response.text, 'html.parser')

                result = {
                    'source': 'Gwinnett County (qPublic)',
                    'parcel_number': parcel_number
                }

                # Same qPublic parsing as other counties
                for table in soup.find_all('table', class_='DataletData'):
                    rows = table.find_all('tr')
                    for row in rows:
                        cells = row.find_all('td')
                        if len(cells) >= 2:
                            label = cells[0].get_text().strip().lower()
                            value = cells[1].get_text().strip()

                            if 'living area' in label or 'heated area' in label or 'square feet' in label or 'finished sq' in label:
                                sqft_match = re.search(r'(\d{1,5})', value.replace(',', ''))
                                if sqft_match:
                                    result['sqft'] = int(sqft_match.group(1))
                                    return result

                return {'error': 'Sqft not found on Gwinnett County page'}
            else:
                return {'error': f'Gwinnett lookup failed: HTTP {response.status_code}'}

        except Exception as e:
            return {'error': f'Gwinnett County error: {str(e)}'}

    def _lookup_clayton(self, address: str, parcel_number: Optional[str] = None) -> Dict:
        """
        Lookup property from Clayton County assessor
        Uses Beacon system (similar to qPublic, AppID=1234)
        """
        try:
            if not parcel_number:
                return {'error': 'Parcel number required for Clayton County'}

            # Clayton uses Beacon (similar to qPublic)
            parcel_clean = parcel_number.replace('-', '').replace(' ', '').upper()

            base_url = "https://beacon.schneidercorp.com/Application.aspx"
            params = {
                'AppID': '1234',  # Clayton County AppID
                'LayerID': '39180',
                'PageTypeID': '4',
                'PageID': '14578',
                'KeyValue': parcel_clean
            }

            response = self.session.get(base_url, params=params, timeout=15)

            if response.status_code == 200:
                soup = BeautifulSoup(response.text, 'html.parser')

                result = {
                    'source': 'Clayton County (Beacon)',
                    'parcel_number': parcel_number
                }

                # Same parsing as qPublic (Beacon uses same structure)
                for table in soup.find_all('table', class_='DataletData'):
                    rows = table.find_all('tr')
                    for row in rows:
                        cells = row.find_all('td')
                        if len(cells) >= 2:
                            label = cells[0].get_text().strip().lower()
                            value = cells[1].get_text().strip()

                            if 'living area' in label or 'heated area' in label or 'square feet' in label or 'finished sq' in label:
                                sqft_match = re.search(r'(\d{1,5})', value.replace(',', ''))
                                if sqft_match:
                                    result['sqft'] = int(sqft_match.group(1))
                                    return result

                return {'error': 'Sqft not found on Clayton County page'}
            else:
                return {'error': f'Clayton lookup failed: HTTP {response.status_code}'}

        except Exception as e:
            return {'error': f'Clayton County error: {str(e)}'}


def lookup_sqft_from_assessor(address: str, city: str, county: str, parcel_number: Optional[str] = None) -> Optional[int]:
    """
    Convenience function to lookup just square footage

    Args:
        address: Street address
        city: City name
        county: County name
        parcel_number: Parcel/APN number (optional)

    Returns:
        Square footage as integer, or None if not found
    """
    lookup = CountyAssessorLookup()
    result = lookup.lookup_property(address, city, county, parcel_number)

    if 'sqft' in result:
        return result['sqft']

    return None


# Example usage
if __name__ == "__main__":
    lookup = CountyAssessorLookup()

    # Test with a DeKalb County property
    result = lookup.lookup_property(
        address="714 Post Road Dr",
        city="Stone Mountain",
        county="DeKalb",
        parcel_number="16-033-13-130"
    )

    print("DeKalb County Lookup Result:")
    print(result)
