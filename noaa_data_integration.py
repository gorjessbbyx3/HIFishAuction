import requests
import pandas as pd
import json
from datetime import datetime, timedelta
from typing import Dict, List, Optional
import sqlite3
import os

class NOAADataIntegration:
    """Integration service for NOAA fisheries data sources"""
    
    def __init__(self):
        self.inport_base_url = "https://www.fisheries.noaa.gov/inport"
        self.data_gov_base_url = "https://catalog.data.gov/api/3/action"
        self.fisheries_data_url = "https://www.fisheries.noaa.gov/resources/data"
        self.feat_base_url = "https://apps-pifsc.fisheries.noaa.gov/FEAT"
        
        # NOAA InPort data identifiers
        self.data_sources = {
            "honolulu_retail_monitoring": {
                "inport_id": "6349",
                "title": "Honolulu Retail Monitoring Fish Price Data Collection (2016)",
                "description": "Weekly consumer-level fish price data from Honolulu retail markets",
                "format": "CSV",
                "access_method": "data.gov"
            },
            "ufa_auction_sampling": {
                "inport_id": "TBD",
                "title": "UFA Auction Sampling Data (1984–2002)",
                "description": "Landing quantities and price-per-pound data from United Fishing Agency",
                "format": "CSV",
                "access_method": "noaa_contact"
            },
            "wpacfin_purchase_reports": {
                "inport_id": "TBD",
                "title": "WPacFIN Purchase Reports & Creel Surveys",
                "description": "Vendor purchase reports and catch-per-unit-effort data",
                "format": "CSV",
                "access_method": "wpacfin_portal"
            }
        }
        
        self.db_path = "fish_auction.db"
    
    def fetch_ufa_auction_data(self) -> Dict:
        """Provide guidance for accessing UFA auction sampling data"""
        return {
            'status': 'Authentication Required',
            'description': 'UFA Auction Sampling Data (1984-2002) requires PIFSC non-disclosure agreement',
            'contact': 'ashley.tomita@noaa.gov',
            'phone': '(808)725-5693',
            'record_count': '1,923,132 records',
            'data_elements': [
                'Price per pound by species',
                'Quantity sold',
                'Auction date',
                'Buyer information',
                'Fisher information',
                'Area codes',
                'Species codes',
                'Condition codes'
            ],
            'data_steward': 'Keith A Bigelow (keith.bigelow@noaa.gov)',
            'access_procedure': 'Send written request to PIFSC and requires approval by the PIFSC data owner',
            'confidentiality': 'Fisheries confidential data - requires signed non-disclosure statement'
        }
    
    def check_noaa_fisheries_data_catalog(self) -> Dict:
        """Check availability of NOAA fisheries data catalog"""
        try:
            print("Accessing NOAA Fisheries Data Catalog...")
            
            response = requests.get(self.fisheries_data_url, timeout=15)
            
            if response.status_code == 200:
                # Parse the page to find relevant datasets
                content = response.text
                
                # Look for Hawaii-specific datasets
                hawaii_datasets = self._extract_hawaii_fisheries_datasets(content)
                
                return {
                    'status': 'Available',
                    'catalog_accessible': True,
                    'hawaii_specific_datasets': len(hawaii_datasets),
                    'relevant_data_types': [
                        'Commercial fisheries landings',
                        'Fishery-dependent data',
                        'Economic and social data',
                        'Pacific Islands fisheries data',
                        'Stock assessments',
                        'Fleet activity reports'
                    ],
                    'integration_priority': 'High - comprehensive fisheries data source'
                }
            else:
                return {
                    'status': 'Limited Access',
                    'message': f'HTTP {response.status_code} - catalog may require navigation'
                }
                
        except Exception as e:
            return {
                'status': 'Connection Failed',
                'error': str(e),
                'fallback': 'Manual navigation to fisheries data catalog required'
            }
    
    def access_feat_performance_indicators(self) -> Dict:
        """Access FEAT performance indicators for Pacific Islands fisheries"""
        try:
            print("Accessing FEAT Performance Indicators...")
            
            feat_indicators_url = f"{self.feat_base_url}/#/report/performance-indicators"
            response = requests.get(feat_indicators_url, timeout=15)
            
            if response.status_code == 200:
                return {
                    'status': 'FEAT System Accessible',
                    'performance_indicators_available': True,
                    'data_types': [
                        'Commercial fishing fleet performance',
                        'Catch per unit effort (CPUE)',
                        'Revenue and cost analysis',
                        'Fleet composition metrics',
                        'Fishing effort statistics',
                        'Economic performance indicators'
                    ],
                    'hawaii_relevance': {
                        'longline_fleet_metrics': 'Direct impact on fish supply',
                        'cpue_trends': 'Predictor of catch abundance',
                        'economic_indicators': 'Market condition insights',
                        'fleet_activity': 'Supply availability forecasting'
                    },
                    'integration_value': 'High - real-time fleet performance affects auction supply',
                    'next_steps': 'Investigate FEAT API access or data export capabilities'
                }
            else:
                return {
                    'status': 'FEAT Access Issue',
                    'message': f'HTTP {response.status_code} - may require authentication or direct navigation'
                }
                
        except Exception as e:
            return {
                'status': 'FEAT Connection Failed',
                'error': str(e),
                'recommendation': 'Manual access to FEAT system may be required'
            }
    
    def _extract_hawaii_fisheries_datasets(self, html_content: str) -> List[str]:
        """Extract Hawaii-specific fisheries datasets from NOAA catalog"""
        try:
            import re
            
            # Look for Hawaii-related dataset patterns
            hawaii_patterns = [
                r'href="([^"]*hawaii[^"]*)"',
                r'href="([^"]*pacific.*island[^"]*)"',
                r'href="([^"]*honolulu[^"]*)"',
                r'href="([^"]*longline[^"]*)"'
            ]
            
            datasets = []
            for pattern in hawaii_patterns:
                matches = re.findall(pattern, html_content, re.IGNORECASE)
                datasets.extend(matches)
            
            # Clean and filter datasets
            cleaned_datasets = []
            for dataset in datasets:
                if dataset.startswith('/'):
                    dataset = 'https://www.fisheries.noaa.gov' + dataset
                if dataset not in cleaned_datasets:
                    cleaned_datasets.append(dataset)
            
            return cleaned_datasets[:10]  # Return top 10 matches
            
        except Exception as e:
            print(f"Error extracting Hawaii datasets: {str(e)}")
            return []

    def fetch_undercurrent_market_data(self) -> Optional[pd.DataFrame]:
        """Fetch global seafood market data from Undercurrent News"""
        try:
            print("Accessing Undercurrent News global seafood market data...")
            
            # Note: This would typically require subscription access
            # For now, provide structure for integration
            
            return {
                'status': 'Subscription Required',
                'description': 'Undercurrent News provides real-time global seafood pricing',
                'relevance': 'Global tuna market pricing affects Hawaii auction prices',
                'subscription_url': 'https://www.undercurrentnews.com/data/',
                'data_types': [
                    'Weekly tuna prices (Tokyo, US, EU markets)',
                    'Salmon pricing trends',
                    'Shrimp market data',
                    'White fish pricing',
                    'Market analysis and forecasts'
                ],
                'integration_value': 'Provides international market context for local Hawaii prices'
            }
            
        except Exception as e:
            print(f"Error accessing Undercurrent News data: {str(e)}")
            return None
    
    def fetch_honolulu_retail_data(self) -> Optional[pd.DataFrame]:
        """Fetch Honolulu retail monitoring fish price data from Data.gov"""
        try:
            print("Accessing NOAA InPort item 6349: Honolulu Retail Monitoring Fish Price Data...")
            
            # Search for the dataset on Data.gov
            search_url = f"{self.data_gov_base_url}/package_search"
            search_params = {
                "q": "Honolulu fish price retail monitoring 2016",
                "fq": "organization:noaa-gov",
                "rows": 10
            }
            
            response = requests.get(search_url, params=search_params, timeout=30)
            
            if response.status_code == 200:
                search_results = response.json()
                datasets = search_results.get('result', {}).get('results', [])
                
                for dataset in datasets:
                    if "honolulu" in dataset.get('title', '').lower() and "fish" in dataset.get('title', '').lower():
                        print(f"Found dataset: {dataset.get('title')}")
                        
                        # Get dataset details
                        package_url = f"{self.data_gov_base_url}/package_show"
                        package_params = {"id": dataset.get('id')}
                        
                        package_response = requests.get(package_url, params=package_params, timeout=30)
                        if package_response.status_code == 200:
                            package_data = package_response.json()
                            resources = package_data.get('result', {}).get('resources', [])
                            
                            # Look for CSV resources
                            for resource in resources:
                                if resource.get('format', '').upper() == 'CSV':
                                    csv_url = resource.get('url')
                                    if csv_url:
                                        print(f"Downloading CSV data from: {csv_url}")
                                        return self._download_and_process_csv(csv_url, 'honolulu_retail')
            
            print("Dataset not found on Data.gov. Manual access may be required.")
            return None
            
        except Exception as e:
            print(f"Error accessing NOAA data: {str(e)}")
            return None
    
    def _download_and_process_csv(self, csv_url: str, data_type: str) -> Optional[pd.DataFrame]:
        """Download and process CSV data from NOAA sources"""
        try:
            print(f"Processing {data_type} data...")
            
            # Download CSV
            response = requests.get(csv_url, timeout=60)
            if response.status_code == 200:
                # Save raw data
                raw_filename = f"data/raw_{data_type}_{datetime.now().strftime('%Y%m%d')}.csv"
                os.makedirs("data", exist_ok=True)
                
                with open(raw_filename, 'wb') as f:
                    f.write(response.content)
                
                # Process data
                df = pd.read_csv(raw_filename)
                processed_df = self._process_fish_price_data(df, data_type)
                
                # Store in database
                self._store_historical_data(processed_df, data_type)
                
                print(f"Successfully processed {len(processed_df)} records from {data_type}")
                return processed_df
            else:
                print(f"Failed to download CSV: HTTP {response.status_code}")
                return None
                
        except Exception as e:
            print(f"Error processing CSV data: {str(e)}")
            return None
    
    def _process_fish_price_data(self, df: pd.DataFrame, data_type: str) -> pd.DataFrame:
        """Process and standardize fish price data"""
        try:
            print(f"Processing {data_type} with {len(df)} rows and columns: {list(df.columns)}")
            
            # Standardize column names (common variations)
            column_mapping = {
                'Date': 'date',
                'DATE': 'date',
                'Sample_Date': 'date',
                'Species': 'species',
                'SPECIES': 'species',
                'Fish_Species': 'species',
                'Price': 'price_per_lb',
                'PRICE': 'price_per_lb',
                'Price_Per_Pound': 'price_per_lb',
                'Price_per_lb': 'price_per_lb',
                'Weight': 'weight',
                'WEIGHT': 'weight',
                'Volume': 'volume',
                'VOLUME': 'volume',
                'Market': 'market_location',
                'Location': 'market_location',
                'Origin': 'origin',
                'ORIGIN': 'origin',
                'Product_Form': 'product_form',
                'Preservation': 'preservation_method'
            }
            
            # Rename columns
            df = df.rename(columns=column_mapping)
            
            # Ensure required columns exist
            required_columns = ['date', 'species', 'price_per_lb']
            missing_columns = [col for col in required_columns if col not in df.columns]
            
            if missing_columns:
                print(f"Warning: Missing required columns: {missing_columns}")
                print(f"Available columns: {list(df.columns)}")
                return pd.DataFrame()  # Return empty DataFrame
            
            # Data cleaning and standardization
            df = df.copy()
            
            # Standardize date format
            if 'date' in df.columns:
                df['date'] = pd.to_datetime(df['date'], errors='coerce')
                df = df.dropna(subset=['date'])
                df['date'] = df['date'].dt.strftime('%Y-%m-%d')
            
            # Standardize species names
            if 'species' in df.columns:
                species_mapping = {
                    'yellowfin tuna': 'Yellowfin Tuna',
                    'ahi': 'Yellowfin Tuna',
                    'bigeye tuna': 'Bigeye Tuna',
                    'mahi mahi': 'Mahi-mahi',
                    'mahi-mahi': 'Mahi-mahi',
                    'opah': 'Opah',
                    'marlin': 'Marlin',
                    'blue marlin': 'Marlin',
                    'striped marlin': 'Marlin'
                }
                
                df['species'] = df['species'].str.lower().str.strip()
                df['species'] = df['species'].replace(species_mapping)
                df = df[df['species'].notna()]
            
            # Clean price data
            if 'price_per_lb' in df.columns:
                # Remove non-numeric characters and convert to float
                df['price_per_lb'] = pd.to_numeric(df['price_per_lb'], errors='coerce')
                df = df.dropna(subset=['price_per_lb'])
                # Filter out unrealistic prices (< $1 or > $100 per lb)
                df = df[(df['price_per_lb'] >= 1.0) & (df['price_per_lb'] <= 100.0)]
            
            # Add metadata
            df['data_source'] = data_type
            df['import_date'] = datetime.now().strftime('%Y-%m-%d')
            
            return df
            
        except Exception as e:
            print(f"Error processing fish price data: {str(e)}")
            return pd.DataFrame()
    
    def _store_historical_data(self, df: pd.DataFrame, data_type: str):
        """Store processed historical data in database"""
        try:
            conn = sqlite3.connect(self.db_path)
            
            # Create enhanced market_data table if not exists
            cursor = conn.cursor()
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS historical_market_data (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    date TEXT NOT NULL,
                    species TEXT NOT NULL,
                    price_per_lb REAL NOT NULL,
                    volume REAL,
                    weight REAL,
                    market_location TEXT,
                    origin TEXT,
                    product_form TEXT,
                    preservation_method TEXT,
                    data_source TEXT NOT NULL,
                    import_date TEXT NOT NULL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            ''')
            
            # Insert data
            df.to_sql('historical_market_data', conn, if_exists='append', index=False)
            
            print(f"Stored {len(df)} records in historical_market_data table")
            
            conn.commit()
            conn.close()
            
        except Exception as e:
            print(f"Error storing historical data: {str(e)}")
    
    def get_data_integration_status(self) -> Dict:
        """Get status of all NOAA data integrations"""
        status = {}
        
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Check what data we have
            cursor.execute('''
                SELECT data_source, COUNT(*) as record_count, 
                       MIN(date) as earliest_date, MAX(date) as latest_date
                FROM historical_market_data 
                GROUP BY data_source
            ''')
            
            results = cursor.fetchall()
            
            for row in results:
                data_source, count, earliest, latest = row
                status[data_source] = {
                    'record_count': count,
                    'date_range': f"{earliest} to {latest}",
                    'status': 'Available'
                }
            
            conn.close()
            
            # Check for missing data sources
            for source_key, source_info in self.data_sources.items():
                if source_key not in status:
                    status[source_key] = {
                        'record_count': 0,
                        'date_range': 'No data',
                        'status': 'Integration needed',
                        'access_method': source_info['access_method'],
                        'description': source_info['description']
                    }
            
            return status
            
        except Exception as e:
            print(f"Error checking data status: {str(e)}")
            return {}
    
    def generate_integration_report(self) -> str:
        """Generate comprehensive integration status report"""
        status = self.get_data_integration_status()
        
        report = ["HAWAII FISH AUCTION DATA INTEGRATION STATUS", "=" * 60, ""]
        
        # Check new data sources
        fisheries_catalog = self.check_noaa_fisheries_data_catalog()
        feat_status = self.access_feat_performance_indicators()
        
        # NOAA Fisheries Data Catalog
        report.extend([
            "🗃️  NOAA FISHERIES DATA CATALOG",
            f"   Status: {fisheries_catalog['status']}",
            f"   URL: {self.fisheries_data_url}",
            f"   Hawaii Datasets Found: {fisheries_catalog.get('hawaii_specific_datasets', 0)}",
            f"   Integration Priority: {fisheries_catalog.get('integration_priority', 'Unknown')}",
            ""
        ])
        
        # FEAT Performance Indicators
        report.extend([
            "📊 FEAT PERFORMANCE INDICATORS",
            f"   Status: {feat_status['status']}",
            f"   URL: {self.feat_base_url}/#/report/performance-indicators",
            f"   Integration Value: {feat_status.get('integration_value', 'Unknown')}",
            ""
        ])
        
        # UFA Auction Data - Primary Source
        ufa_info = self.fetch_ufa_auction_data()
        report.extend([
            "🎯 PRIMARY DATA SOURCE - UFA AUCTION SAMPLING (1984-2002)",
            f"   Status: {ufa_info['status']}",
            f"   Records: {ufa_info['record_count']}",
            f"   Contact: {ufa_info['contact']} / {ufa_info['phone']}",
            f"   Access: {ufa_info['access_procedure']}",
            f"   Confidentiality: {ufa_info['confidentiality']}",
            ""
        ])
        
        # Global Market Context
        undercurrent_info = self.fetch_undercurrent_market_data()
        if undercurrent_info:
            report.extend([
                "🌏 GLOBAL MARKET CONTEXT - UNDERCURRENT NEWS",
                f"   Status: {undercurrent_info['status']}",
                f"   Relevance: {undercurrent_info['relevance']}",
                f"   Integration Value: {undercurrent_info['integration_value']}",
                ""
            ])
        
        # Available data
        available_sources = [k for k, v in status.items() if v.get('record_count', 0) > 0]
        if available_sources:
            report.append("✅ CURRENTLY INTEGRATED:")
            for source in available_sources:
                info = status[source]
                report.append(f"  • {source}: {info['record_count']} records ({info['date_range']})")
            report.append("")
        
        # Missing data
        missing_sources = [k for k, v in status.items() if v.get('record_count', 0) == 0]
        if missing_sources:
            report.append("❌ PENDING INTEGRATIONS:")
            for source in missing_sources:
                info = status[source]
                report.append(f"  • {source}: {info.get('description', 'No description')}")
                report.append(f"    Access: {info.get('access_method', 'Unknown')}")
            report.append("")
        
        # Critical next steps
        report.extend([
            "🔑 CRITICAL NEXT STEPS:",
            "1. NOAA FISHERIES DATA CATALOG EXPLORATION:",
            f"   - Navigate to {self.fisheries_data_url}",
            "   - Search for 'Hawaii', 'Pacific Islands', and 'longline' datasets",
            "   - Identify downloadable CSV/Excel files for historical data",
            "   - Focus on commercial fisheries landings and fleet statistics",
            "",
            "2. FEAT SYSTEM INTEGRATION:",
            f"   - Access {self.feat_base_url}/#/report/performance-indicators",
            "   - Investigate API endpoints or data export capabilities",
            "   - Extract fleet performance and CPUE data for supply forecasting",
            "   - Monitor economic indicators affecting fish prices",
            "",
            "3. NOAA UFA DATA ACCESS (PRIMARY):",
            "   - Email ashley.tomita@noaa.gov for access to 1.9M auction records",
            "   - Complete PIFSC non-disclosure agreement",
            "   - Request data steward approval (keith.bigelow@noaa.gov)",
            "",
            "4. GLOBAL MARKET INTEGRATION:",
            "   - Evaluate Undercurrent News subscription for international context",
            "   - Assess Pacific tuna market pricing correlations",
            "",
            "5. TECHNICAL IMPLEMENTATION:",
            "   - Design secure data handling for confidential fisheries data",
            "   - Implement automated data validation and quality checks",
            "   - Create prediction model training pipeline with authentic data",
            "",
            "📊 IMPACT: Comprehensive NOAA integration will provide real-time fleet",
            "    performance data and 18+ years of authentic Hawaii auction prices.",
            ""
        ])
        
        return "\n".join(report)
    
    def update_prediction_model_data(self):
        """Update prediction model with authentic historical data"""
        try:
            conn = sqlite3.connect(self.db_path)
            
            # Get aggregated historical data for model training
            query = '''
                SELECT 
                    date,
                    species,
                    AVG(price_per_lb) as avg_price,
                    COUNT(*) as sample_count,
                    STDDEV(price_per_lb) as price_volatility
                FROM historical_market_data 
                WHERE date >= date('now', '-2 years')
                GROUP BY date, species
                HAVING sample_count >= 2
                ORDER BY date
            '''
            
            historical_df = pd.read_sql_query(query, conn)
            conn.close()
            
            if not historical_df.empty:
                print(f"Retrieved {len(historical_df)} aggregated price records for model training")
                
                # Update the existing market_data table with processed data
                conn = sqlite3.connect(self.db_path)
                
                # Clear existing market_data and insert processed historical data
                cursor = conn.cursor()
                cursor.execute('DELETE FROM market_data WHERE source = "noaa_historical"')
                
                for _, row in historical_df.iterrows():
                    cursor.execute('''
                        INSERT INTO market_data (date, species, price_per_lb, volume, source)
                        VALUES (?, ?, ?, ?, ?)
                    ''', (row['date'], row['species'], row['avg_price'], row['sample_count'], 'noaa_historical'))
                
                conn.commit()
                conn.close()
                
                print("Updated prediction model with authentic NOAA historical data")
                return True
            else:
                print("No suitable historical data found for model training")
                return False
                
        except Exception as e:
            print(f"Error updating prediction model: {str(e)}")
            return False