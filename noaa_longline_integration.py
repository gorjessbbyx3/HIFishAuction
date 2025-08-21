import requests
import pandas as pd
import json
from datetime import datetime, timedelta
from typing import Dict, List, Optional
import sqlite3
import os
import re
from io import StringIO

class NOAALonglineIntegration:
    """Integration service for NOAA Pacific Islands longline fishery data"""
    
    def __init__(self):
        self.base_url = "https://www.fisheries.noaa.gov"
        self.data_endpoints = {
            'hawaii_california': '/resource/data/hawaii-and-california-longline-fishery-logbook-summary-reports',
            'american_samoa': '/resource/data/american-samoa-longline-fishery-logbook-summary-reports'
        }
        
        # Species mapping from logbook data to Hawaii auction species
        self.species_mapping = {
            'yellowfin_tuna': ['yellowfin', 'yft', 'ahi'],
            'bigeye_tuna': ['bigeye', 'bet', 'bigeye_tuna'],
            'albacore': ['albacore', 'alb'],
            'skipjack': ['skipjack', 'skj'],
            'blue_marlin': ['blue_marlin', 'blm', 'marlin'],
            'striped_marlin': ['striped_marlin', 'mls'],
            'swordfish': ['swordfish', 'swo'],
            'mahi_mahi': ['mahi', 'dol', 'dolphinfish'],
            'opah': ['opah', 'opa', 'moonfish'],
            'wahoo': ['wahoo', 'wah', 'ono']
        }
        
        self.db_path = "fish_auction.db"
    
    def check_data_availability(self) -> Dict:
        """Check availability of NOAA longline fishery data"""
        try:
            hawaii_status = self._check_endpoint_status('hawaii_california')
            samoa_status = self._check_endpoint_status('american_samoa')
            
            return {
                'status': 'Available',
                'hawaii_california_data': hawaii_status,
                'american_samoa_data': samoa_status,
                'data_types': [
                    'Monthly catch summaries',
                    'Species composition data',
                    'Fishing effort statistics',
                    'CPUE (Catch Per Unit Effort)',
                    'Seasonal catch patterns'
                ],
                'update_frequency': 'Monthly with 2-3 month lag',
                'coverage_area': 'Central and Western Pacific'
            }
            
        except Exception as e:
            return {
                'status': 'Connection Error',
                'error': str(e),
                'fallback': 'Using demonstration data structure'
            }
    
    def _check_endpoint_status(self, endpoint_key: str) -> Dict:
        """Check status of specific NOAA data endpoint"""
        try:
            url = self.base_url + self.data_endpoints[endpoint_key]
            response = requests.get(url, timeout=10)
            
            if response.status_code == 200:
                # Look for download links or data references
                content = response.text
                data_links = self._extract_data_links(content)
                
                return {
                    'status': 'Accessible',
                    'data_files_found': len(data_links),
                    'recent_updates': self._find_recent_data_dates(content),
                    'download_links': data_links[:5]  # First 5 links
                }
            else:
                return {
                    'status': 'HTTP Error',
                    'status_code': response.status_code
                }
                
        except Exception as e:
            return {
                'status': 'Error',
                'error': str(e)
            }
    
    def _extract_data_links(self, html_content: str) -> List[str]:
        """Extract data download links from NOAA page"""
        try:
            # Look for common data file patterns
            patterns = [
                r'href="([^"]*\.csv[^"]*)"',
                r'href="([^"]*\.xlsx?[^"]*)"',
                r'href="([^"]*logbook[^"]*)"',
                r'href="([^"]*summary[^"]*)"'
            ]
            
            links = []
            for pattern in patterns:
                matches = re.findall(pattern, html_content, re.IGNORECASE)
                links.extend(matches)
            
            # Filter and clean links
            cleaned_links = []
            for link in links:
                if link.startswith('/'):
                    link = self.base_url + link
                elif not link.startswith('http'):
                    continue
                cleaned_links.append(link)
            
            return list(set(cleaned_links))  # Remove duplicates
            
        except Exception as e:
            print(f"Error extracting data links: {str(e)}")
            return []
    
    def _find_recent_data_dates(self, content: str) -> List[str]:
        """Find recent data dates mentioned in the content"""
        try:
            # Look for date patterns (YYYY, YYYY-MM, etc.)
            date_patterns = [
                r'\b20[12]\d\b',  # Years 2010-2029
                r'\b20[12]\d-[01]\d\b',  # YYYY-MM format
                r'\b(January|February|March|April|May|June|July|August|September|October|November|December)\s+20[12]\d\b'
            ]
            
            dates = []
            for pattern in date_patterns:
                matches = re.findall(pattern, content, re.IGNORECASE)
                dates.extend(matches)
            
            # Sort and return most recent
            unique_dates = list(set(dates))
            return sorted(unique_dates, reverse=True)[:5]
            
        except Exception as e:
            return []
    
    def fetch_longline_catch_data(self, fishery: str = 'hawaii_california', 
                                 months_back: int = 12) -> Optional[pd.DataFrame]:
        """Fetch longline catch data from NOAA sources"""
        try:
            if fishery not in self.data_endpoints:
                raise ValueError(f"Unknown fishery: {fishery}")
            
            # Check data availability first
            status = self._check_endpoint_status(fishery)
            
            if status.get('status') != 'Accessible':
                print(f"NOAA {fishery} data not accessible: {status}")
                return None
            
            # Attempt to download recent data files
            data_links = status.get('download_links', [])
            
            if not data_links:
                print(f"No data download links found for {fishery}")
                return None
            
            # Try to download and parse data files
            catch_data = []
            for link in data_links[:3]:  # Try first 3 files
                try:
                    file_data = self._download_and_parse_file(link)
                    if file_data is not None:
                        catch_data.append(file_data)
                except Exception as e:
                    print(f"Error processing file {link}: {str(e)}")
                    continue
            
            if catch_data:
                combined_data = pd.concat(catch_data, ignore_index=True)
                return self._process_longline_data(combined_data, fishery)
            else:
                print("No valid data files could be processed")
                return None
            
        except Exception as e:
            print(f"Error fetching longline data: {str(e)}")
            return None
    
    def _download_and_parse_file(self, url: str) -> Optional[pd.DataFrame]:
        """Download and parse a data file from NOAA"""
        try:
            response = requests.get(url, timeout=30)
            
            if response.status_code != 200:
                return None
            
            # Determine file type and parse accordingly
            if url.lower().endswith('.csv'):
                return pd.read_csv(StringIO(response.text))
            elif url.lower().endswith(('.xlsx', '.xls')):
                return pd.read_excel(response.content)
            else:
                # Try parsing as CSV first
                try:
                    return pd.read_csv(StringIO(response.text))
                except:
                    return None
                    
        except Exception as e:
            print(f"Error downloading file {url}: {str(e)}")
            return None
    
    def _process_longline_data(self, raw_data: pd.DataFrame, fishery: str) -> pd.DataFrame:
        """Process and standardize longline fishery data"""
        try:
            processed_data = []
            
            for _, row in raw_data.iterrows():
                # Extract standardized fields
                record = self._extract_standard_fields(row, fishery)
                if record:
                    processed_data.append(record)
            
            if processed_data:
                df = pd.DataFrame(processed_data)
                return self._validate_and_clean_data(df)
            else:
                return pd.DataFrame()
                
        except Exception as e:
            print(f"Error processing longline data: {str(e)}")
            return pd.DataFrame()
    
    def _extract_standard_fields(self, row: pd.Series, fishery: str) -> Optional[Dict]:
        """Extract standard fields from a data row"""
        try:
            # Common field mappings (may need adjustment based on actual data format)
            field_mappings = {
                'date_fields': ['date', 'month', 'year', 'period'],
                'species_fields': ['species', 'species_name', 'spp'],
                'catch_fields': ['catch', 'pounds', 'weight', 'kg'],
                'effort_fields': ['trips', 'sets', 'hooks', 'effort'],
                'vessel_fields': ['vessels', 'vessel_count', 'boats']
            }
            
            record = {
                'fishery': fishery,
                'data_source': 'noaa_longline_logbook'
            }
            
            # Extract date information
            for field in field_mappings['date_fields']:
                if field in row.index and not pd.isna(row[field]):
                    record['period'] = str(row[field])
                    break
            
            # Extract species information
            for field in field_mappings['species_fields']:
                if field in row.index and not pd.isna(row[field]):
                    species = str(row[field]).lower()
                    mapped_species = self._map_species_name(species)
                    if mapped_species:
                        record['species'] = mapped_species
                        break
            
            # Extract catch data
            for field in field_mappings['catch_fields']:
                if field in row.index and not pd.isna(row[field]):
                    try:
                        record['catch_weight'] = float(row[field])
                        break
                    except:
                        continue
            
            # Extract effort data
            for field in field_mappings['effort_fields']:
                if field in row.index and not pd.isna(row[field]):
                    try:
                        record['effort'] = float(row[field])
                        break
                    except:
                        continue
            
            # Only return record if we have minimum required fields
            if 'species' in record and 'catch_weight' in record:
                return record
            else:
                return None
                
        except Exception as e:
            return None
    
    def _map_species_name(self, species_name: str) -> Optional[str]:
        """Map logbook species name to standard Hawaii auction species"""
        species_lower = species_name.lower()
        
        for standard_name, variations in self.species_mapping.items():
            for variation in variations:
                if variation in species_lower:
                    # Convert to Hawaii auction format
                    return standard_name.replace('_', ' ').title()
        
        return None
    
    def _validate_and_clean_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Validate and clean the processed data"""
        try:
            # Remove records with missing critical data
            df = df.dropna(subset=['species', 'catch_weight'])
            
            # Filter for positive catch weights
            if 'catch_weight' in df.columns:
                df = df[df['catch_weight'] > 0]
            
            # Add processing timestamp
            df['processed_date'] = datetime.now().strftime('%Y-%m-%d')
            
            return df
            
        except Exception as e:
            print(f"Error validating data: {str(e)}")
            return df
    
    def store_longline_data(self, data: pd.DataFrame):
        """Store longline fishery data in database"""
        try:
            conn = sqlite3.connect(self.db_path)
            
            # Create longline data table if it doesn't exist
            cursor = conn.cursor()
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS longline_fishery_data (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    fishery TEXT NOT NULL,
                    species TEXT NOT NULL,
                    period TEXT,
                    catch_weight REAL,
                    effort REAL,
                    data_source TEXT NOT NULL,
                    processed_date TEXT NOT NULL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            ''')
            
            # Insert data
            data.to_sql('longline_fishery_data', conn, if_exists='append', index=False)
            
            conn.commit()
            conn.close()
            
            print(f"Stored {len(data)} longline fishery records")
            
        except Exception as e:
            print(f"Error storing longline data: {str(e)}")
    
    def analyze_catch_supply_impact(self, species: str, days_forward: int = 7) -> Dict:
        """Analyze how recent catch data affects expected supply and prices"""
        try:
            conn = sqlite3.connect(self.db_path)
            
            # Get recent catch data for the species
            query = '''
                SELECT * FROM longline_fishery_data 
                WHERE species = ? 
                ORDER BY processed_date DESC, period DESC 
                LIMIT 12
            '''
            
            df = pd.read_sql_query(query, conn, params=[species])
            conn.close()
            
            if df.empty:
                return {
                    'status': 'No Data',
                    'message': f'No recent longline catch data available for {species}',
                    'recommendation': 'Integration with NOAA logbook data needed'
                }
            
            # Analyze catch trends
            catch_analysis = self._analyze_catch_trends(df)
            
            # Predict supply impact
            supply_impact = self._predict_supply_impact(catch_analysis, days_forward)
            
            return {
                'status': 'Analysis Complete',
                'species': species,
                'catch_trend': catch_analysis,
                'supply_forecast': supply_impact,
                'price_impact_prediction': self._predict_price_impact(supply_impact)
            }
            
        except Exception as e:
            return {
                'status': 'Analysis Failed',
                'error': str(e)
            }
    
    def _analyze_catch_trends(self, df: pd.DataFrame) -> Dict:
        """Analyze catch trends from longline data"""
        try:
            total_catch = df['catch_weight'].sum()
            avg_catch = df['catch_weight'].mean()
            recent_catch = df.head(3)['catch_weight'].mean()
            
            # Calculate trend
            if recent_catch > avg_catch * 1.1:
                trend = 'Increasing'
            elif recent_catch < avg_catch * 0.9:
                trend = 'Decreasing'
            else:
                trend = 'Stable'
            
            return {
                'total_recent_catch': round(total_catch, 1),
                'average_catch': round(avg_catch, 1),
                'recent_average': round(recent_catch, 1),
                'trend': trend,
                'data_points': len(df)
            }
            
        except Exception as e:
            return {'trend': 'Unknown', 'error': str(e)}
    
    def _predict_supply_impact(self, catch_analysis: Dict, days_forward: int) -> Dict:
        """Predict supply impact based on catch trends"""
        trend = catch_analysis.get('trend', 'Unknown')
        recent_avg = catch_analysis.get('recent_average', 0)
        
        # Simple supply prediction logic
        if trend == 'Increasing':
            supply_change = '+15% to +25%'
            market_impact = 'Increased supply likely to moderate prices'
        elif trend == 'Decreasing':
            supply_change = '-15% to -25%'
            market_impact = 'Reduced supply likely to increase prices'
        else:
            supply_change = 'No significant change'
            market_impact = 'Supply conditions stable'
        
        return {
            'forecast_period': f'{days_forward} days',
            'expected_supply_change': supply_change,
            'market_impact': market_impact,
            'confidence': 'Moderate' if recent_avg > 0 else 'Low'
        }
    
    def _predict_price_impact(self, supply_impact: Dict) -> str:
        """Predict price impact based on supply forecast"""
        supply_change = supply_impact.get('expected_supply_change', '')
        
        if '+' in supply_change:
            return 'Downward pressure on prices expected due to increased supply'
        elif '-' in supply_change:
            return 'Upward pressure on prices expected due to reduced supply'
        else:
            return 'No significant price impact expected from supply conditions'
    
    def get_integration_summary(self) -> Dict:
        """Get summary of NOAA longline data integration capabilities"""
        return {
            'data_sources': {
                'hawaii_california_longline': 'Hawaii and California longline fishery logbook summaries',
                'american_samoa_longline': 'American Samoa longline fishery logbook summaries'
            },
            'data_types': [
                'Monthly catch summaries by species',
                'Fishing effort statistics',
                'CPUE (Catch Per Unit Effort) data',
                'Fleet composition and activity',
                'Seasonal catch patterns'
            ],
            'integration_benefits': [
                'Real supply-side data for price predictions',
                'Early indicators of catch abundance',
                'Seasonal pattern validation',
                'Fleet activity monitoring'
            ],
            'update_frequency': 'Monthly (2-3 month lag)',
            'coverage_area': 'Central and Western Pacific Ocean',
            'data_format': 'CSV and Excel files',
            'public_access': True,
            'api_available': False
        }