
import requests
import pandas as pd
import json
import os
from datetime import datetime, timedelta
from typing import Dict, List, Optional
import sqlite3
import re
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
import time

class FEATDataExtractor:
    """Extract Hawaii fish auction data from NOAA FEAT system"""
    
    def __init__(self):
        self.feat_base_url = "https://apps-pifsc.fisheries.noaa.gov/FEAT"
        self.hawaii_trends_url = f"{self.feat_base_url}/#/report/hawaii-trends"
        self.db_path = "fish_auction.db"
        
        # Common Hawaii fish species in auctions
        self.target_species = [
            'Yellowfin Tuna', 'Bigeye Tuna', 'Skipjack Tuna',
            'Mahi-mahi', 'Marlin', 'Swordfish', 'Opah', 'Ono'
        ]
    
    def extract_hawaii_trends_data(self, headless: bool = True) -> Dict:
        """Extract data from FEAT Hawaii trends page"""
        try:
            print("🌐 Accessing FEAT Hawaii Trends page...")
            
            # First try simple HTTP request
            simple_result = self._try_simple_request()
            if simple_result.get('success'):
                return simple_result
            
            # If simple request fails, try browser automation
            print("   📱 Attempting browser-based extraction...")
            return self._try_browser_extraction(headless)
            
        except Exception as e:
            return {
                'success': False,
                'error': str(e),
                'recommendation': 'Manual navigation to FEAT required'
            }
    
    def _try_simple_request(self) -> Dict:
        """Try to extract data with simple HTTP requests"""
        try:
            # Check if FEAT has API endpoints
            api_endpoints = [
                f"{self.feat_base_url}/api/hawaii-trends",
                f"{self.feat_base_url}/api/data/hawaii",
                f"{self.feat_base_url}/data/hawaii-trends.json",
                f"{self.feat_base_url}/export/hawaii-trends"
            ]
            
            for endpoint in api_endpoints:
                try:
                    response = requests.get(endpoint, timeout=15)
                    if response.status_code == 200:
                        # Try to parse as JSON
                        if 'json' in response.headers.get('content-type', '').lower():
                            data = response.json()
                            processed_data = self._process_api_data(data)
                            if processed_data:
                                return {
                                    'success': True,
                                    'data_source': endpoint,
                                    'data': processed_data,
                                    'method': 'API'
                                }
                        
                        # Try to parse as CSV
                        if 'csv' in response.headers.get('content-type', '').lower():
                            df = pd.read_csv(response.text)
                            processed_data = self._process_csv_data(df)
                            if not processed_data.empty:
                                return {
                                    'success': True,
                                    'data_source': endpoint,
                                    'data': processed_data,
                                    'method': 'CSV'
                                }
                
                except requests.RequestException:
                    continue
            
            return {'success': False, 'method': 'simple_request'}
            
        except Exception as e:
            return {'success': False, 'error': str(e)}
    
    def _try_browser_extraction(self, headless: bool = True) -> Dict:
        """Try to extract data using browser automation"""
        try:
            print("   🤖 Setting up browser automation...")
            
            # Note: This would require selenium and chromedriver
            # For Replit environment, we'll simulate what this would do
            
            extraction_plan = {
                'success': False,
                'method': 'browser_simulation',
                'steps_needed': [
                    f"Navigate to {self.hawaii_trends_url}",
                    "Wait for page to load completely",
                    "Look for data tables with Hawaii commercial fishing data",
                    "Search for export/download buttons",
                    "Extract table data if visible",
                    "Look for CSV download links",
                    "Check for API endpoints in network requests"
                ],
                'manual_instructions': [
                    "1. Open browser and navigate to FEAT Hawaii Trends",
                    "2. Look for these data elements:",
                    "   - Commercial landing volumes by species",
                    "   - Average prices per pound",
                    "   - Monthly/quarterly trends",
                    "   - Fleet activity metrics",
                    "3. Look for these UI elements:",
                    "   - 'Export' button",
                    "   - 'Download CSV' link", 
                    "   - 'Data' or 'Raw Data' options",
                    "4. If found, download and save to /data directory",
                    "5. Run data processing script on downloaded files"
                ],
                'expected_data_format': {
                    'columns': ['date', 'species', 'volume_lbs', 'avg_price_per_lb', 'revenue'],
                    'frequency': 'monthly or quarterly',
                    'species_coverage': self.target_species
                }
            }
            
            return extraction_plan
            
        except Exception as e:
            return {
                'success': False,
                'error': str(e),
                'method': 'browser_extraction'
            }
    
    def _process_api_data(self, api_data: Dict) -> Optional[pd.DataFrame]:
        """Process data returned from API endpoints"""
        try:
            # Look for common data structures
            if 'data' in api_data:
                data_section = api_data['data']
            elif 'results' in api_data:
                data_section = api_data['results']
            elif 'records' in api_data:
                data_section = api_data['records']
            else:
                data_section = api_data
            
            # Convert to DataFrame if it's a list of records
            if isinstance(data_section, list) and len(data_section) > 0:
                df = pd.DataFrame(data_section)
                return self._standardize_data_format(df)
            
            return None
            
        except Exception as e:
            print(f"Error processing API data: {str(e)}")
            return None
    
    def _process_csv_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Process CSV data from FEAT"""
        return self._standardize_data_format(df)
    
    def _standardize_data_format(self, df: pd.DataFrame) -> pd.DataFrame:
        """Standardize data format for Hawaii auction system"""
        try:
            # Look for relevant columns
            column_mappings = {
                'date': ['date', 'month', 'year', 'period', 'time'],
                'species': ['species', 'fish', 'fish_species', 'spp'],
                'volume': ['volume', 'weight', 'pounds', 'lbs', 'catch'],
                'price': ['price', 'price_per_lb', 'avg_price', 'unit_price'],
                'revenue': ['revenue', 'value', 'total_value', 'ex_vessel_value']
            }
            
            standardized_data = {}
            
            for standard_col, possible_names in column_mappings.items():
                for col_name in df.columns:
                    if any(name.lower() in col_name.lower() for name in possible_names):
                        standardized_data[standard_col] = df[col_name]
                        break
            
            if len(standardized_data) >= 3:  # Need at least date, species, and one metric
                result_df = pd.DataFrame(standardized_data)
                
                # Filter for Hawaii auction species
                if 'species' in result_df.columns:
                    species_filter = result_df['species'].str.contains(
                        '|'.join(self.target_species), 
                        case=False, 
                        na=False
                    )
                    result_df = result_df[species_filter]
                
                return result_df
            
            return pd.DataFrame()
            
        except Exception as e:
            print(f"Error standardizing data: {str(e)}")
            return pd.DataFrame()
    
    def create_feat_access_guide(self) -> str:
        """Create comprehensive guide for accessing FEAT data"""
        guide = [
            "🎯 FEAT HAWAII TRENDS DATA ACCESS GUIDE",
            "=" * 50,
            "",
            "📍 TARGET URL:",
            f"   {self.hawaii_trends_url}",
            "",
            "🔍 WHAT TO LOOK FOR:",
            "",
            "1. HAWAII COMMERCIAL FISHING DATA:",
            "   ✓ Commercial landings by species (pounds)",
            "   ✓ Average price per pound by species",
            "   ✓ Monthly or quarterly trend data",
            "   ✓ Fleet activity and vessel counts",
            "   ✓ Revenue and economic indicators",
            "",
            "2. KEY SPECIES TO FOCUS ON:",
            f"   ✓ {', '.join(self.target_species[:4])}",
            f"   ✓ {', '.join(self.target_species[4:])}",
            "",
            "3. UI ELEMENTS TO SEARCH FOR:",
            "   🔗 'Export Data' button",
            "   🔗 'Download CSV' link",
            "   🔗 'Raw Data' or 'Data Table' options",
            "   🔗 'API Documentation' or 'Data Access' links",
            "   📊 Interactive charts with data export options",
            "",
            "📥 DOWNLOAD INSTRUCTIONS:",
            "",
            "IF YOU FIND DOWNLOADABLE DATA:",
            "1. Download files to your computer",
            "2. Upload files to /data directory in this Repl",
            "3. Run this command to process them:",
            "   python feat_data_extractor.py --process-files",
            "",
            "EXPECTED FILE FORMATS:",
            "   ✓ CSV files (preferred)",
            "   ✓ Excel files (.xlsx, .xls)",
            "   ✓ JSON data exports",
            "",
            "🔧 MANUAL DATA EXTRACTION:",
            "",
            "IF NO DOWNLOAD OPTIONS:",
            "1. Look for data tables on the page",
            "2. Copy table data to clipboard",
            "3. Paste into Excel or Google Sheets",
            "4. Save as CSV file",
            "5. Upload to /data directory",
            "",
            "MINIMUM DATA REQUIREMENTS:",
            "   📅 Date/Period column",
            "   🐟 Species column",
            "   📊 Either volume (pounds) OR price per pound",
            "   📈 At least 6 months of historical data",
            "",
            "🎯 PRIORITY DATA POINTS:",
            "",
            "MOST VALUABLE FOR PRICE PREDICTION:",
            "1. Monthly commercial landings by species",
            "2. Average ex-vessel prices (price paid to fishermen)",
            "3. Seasonal catch patterns",
            "4. Fleet activity indicators",
            "",
            "ALTERNATIVE DATA SOURCES IF FEAT FAILS:",
            "1. Hawaii DLNR Commercial Fishing Reports",
            "2. WPacFIN Purchase Reports",
            "3. NOAA Fisheries Statistics Division",
            "4. Pacific Islands Regional Office data",
            "",
            "📞 CONTACT FOR DATA ACCESS:",
            "   NOAA Pacific Islands Fisheries Science Center",
            "   Phone: (808) 725-5300",
            "   Email: pifsc.data@noaa.gov",
            "",
            "💡 TIPS FOR SUCCESS:",
            "   • Try different browsers if page doesn't load",
            "   • Look for mobile/simplified versions of reports",
            "   • Check page source for hidden API endpoints",
            "   • Use browser developer tools to monitor network requests",
            "   • Contact PIFSC directly if system access is restricted",
            ""
        ]
        
        return "\n".join(guide)
    
    def process_uploaded_files(self, data_directory: str = "data") -> List[Dict]:
        """Process manually uploaded FEAT data files"""
        try:
            import os
            processed_files = []
            
            if not os.path.exists(data_directory):
                return []
            
            # Look for relevant files
            relevant_files = []
            for filename in os.listdir(data_directory):
                if any(term in filename.lower() for term in ['feat', 'hawaii', 'trend', 'commercial']):
                    if filename.endswith(('.csv', '.xlsx', '.xls', '.json')):
                        relevant_files.append(os.path.join(data_directory, filename))
            
            for filepath in relevant_files:
                try:
                    print(f"📁 Processing {os.path.basename(filepath)}...")
                    
                    # Determine file type and read
                    if filepath.endswith('.csv'):
                        df = pd.read_csv(filepath)
                    elif filepath.endswith(('.xlsx', '.xls')):
                        df = pd.read_excel(filepath)
                    elif filepath.endswith('.json'):
                        with open(filepath, 'r') as f:
                            json_data = json.load(f)
                        df = pd.DataFrame(json_data)
                    else:
                        continue
                    
                    # Standardize and validate
                    standardized_df = self._standardize_data_format(df)
                    
                    if not standardized_df.empty:
                        # Store in database
                        self._store_feat_data(standardized_df, filepath)
                        
                        processed_files.append({
                            'filename': os.path.basename(filepath),
                            'rows': len(standardized_df),
                            'columns': list(standardized_df.columns),
                            'species_found': standardized_df.get('species', pd.Series()).unique().tolist(),
                            'date_range': self._get_date_range(standardized_df)
                        })
                        
                        print(f"   ✅ Processed {len(standardized_df)} records")
                    else:
                        print(f"   ⚠️  No relevant data found in file")
                
                except Exception as e:
                    print(f"   ❌ Error processing {filepath}: {str(e)}")
            
            return processed_files
            
        except Exception as e:
            print(f"Error processing uploaded files: {str(e)}")
            return []
    
    def _store_feat_data(self, df: pd.DataFrame, source_file: str):
        """Store FEAT data in database"""
        try:
            conn = sqlite3.connect(self.db_path)
            
            # Prepare data for storage
            df['data_source'] = f'feat_{os.path.basename(source_file)}'
            df['import_date'] = datetime.now().strftime('%Y-%m-%d')
            
            # Store in market_data table (replacing demo data)
            df.to_sql('historical_market_data', conn, if_exists='append', index=False)
            
            print(f"   💾 Stored in database: {len(df)} records")
            
            conn.close()
            
        except Exception as e:
            print(f"Error storing FEAT data: {str(e)}")
    
    def _get_date_range(self, df: pd.DataFrame) -> str:
        """Get date range from DataFrame"""
        try:
            if 'date' in df.columns:
                dates = pd.to_datetime(df['date'], errors='coerce').dropna()
                if not dates.empty:
                    return f"{dates.min().strftime('%Y-%m')} to {dates.max().strftime('%Y-%m')}"
            return "Date range unknown"
        except:
            return "Date range unknown"

def main():
    """Main function for FEAT data extraction"""
    extractor = FEATDataExtractor()
    
    print("🎯 FEAT Hawaii Trends Data Extractor")
    print("=" * 40)
    
    # Generate access guide
    guide = extractor.create_feat_access_guide()
    print(guide)
    
    # Save guide to file
    with open('feat_access_guide.txt', 'w') as f:
        f.write(guide)
    
    print(f"\n📄 Access guide saved to: feat_access_guide.txt")
    
    # Check for existing data files
    processed_files = extractor.process_uploaded_files()
    
    if processed_files:
        print(f"\n✅ Processed {len(processed_files)} existing files:")
        for file_info in processed_files:
            print(f"   📁 {file_info['filename']}: {file_info['rows']} rows")
            print(f"      🗓️  {file_info['date_range']}")
            print(f"      🐟 Species: {', '.join(file_info['species_found'][:3])}...")
    else:
        print(f"\n📋 No existing FEAT data files found.")
        print(f"   Follow the guide above to manually download data from FEAT.")
    
    # Try automated extraction
    print(f"\n🤖 Attempting automated data extraction...")
    result = extractor.extract_hawaii_trends_data()
    
    if result.get('success'):
        print(f"   ✅ Automated extraction successful!")
        print(f"   📊 Method: {result.get('method')}")
    else:
        print(f"   ⚠️  Automated extraction not available")
        print(f"   💡 Manual navigation required - follow guide above")

if __name__ == "__main__":
    main()
