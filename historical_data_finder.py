
import requests
import pandas as pd
import json
from datetime import datetime, timedelta
from typing import Dict, List, Optional
import sqlite3
import re
from bs4 import BeautifulSoup
import time

class HistoricalDataFinder:
    """Comprehensive finder for Hawaii fish auction historical data sources"""
    
    def __init__(self):
        self.data_sources = {
            'feat_hawaii_trends': {
                'url': 'https://apps-pifsc.fisheries.noaa.gov/FEAT/#/report/hawaii-trends',
                'description': 'NOAA FEAT Hawaii fishing trends and performance data',
                'data_types': ['commercial landings', 'fleet activity', 'revenue trends']
            },
            'noaa_inport_6349': {
                'url': 'https://www.fisheries.noaa.gov/inport/item/6349',
                'description': 'Honolulu Retail Monitoring Fish Price Data Collection (2016)',
                'data_types': ['retail prices', 'market monitoring', 'species pricing']
            },
            'pacific_fishery_data': {
                'url': 'https://www.fisheries.noaa.gov/pacific-islands/commercial-fishing/pacific-islands-commercial-fishery-data',
                'description': 'Pacific Islands commercial fishery statistics',
                'data_types': ['landing statistics', 'catch data', 'fleet reports']
            },
            'hawaii_catch_statistics': {
                'url': 'https://dlnr.hawaii.gov/dar/fishing/commercial-fishing/',
                'description': 'Hawaii Department of Land and Natural Resources catch data',
                'data_types': ['state commercial data', 'local fishing reports']
            },
            'wpacfin_data': {
                'url': 'http://www.wpacfin.org/',
                'description': 'Western Pacific Fishery Information Network',
                'data_types': ['purchase reports', 'dealer data', 'price summaries']
            },
            'data_gov_fisheries': {
                'url': 'https://catalog.data.gov/dataset?organization=noaa-gov&tags=fisheries&q=hawaii',
                'description': 'Data.gov NOAA fisheries datasets for Hawaii',
                'data_types': ['various federal datasets', 'downloadable CSV files']
            }
        }
        
        self.db_path = "fish_auction.db"
        
    def search_all_sources(self) -> Dict:
        """Search all known data sources for available historical data"""
        results = {}
        
        print("🔍 Searching for Hawaii fish auction historical data sources...")
        print("=" * 60)
        
        for source_name, source_info in self.data_sources.items():
            print(f"\n📊 Checking {source_name}...")
            
            try:
                if source_name == 'feat_hawaii_trends':
                    results[source_name] = self._check_feat_hawaii_trends()
                elif source_name == 'data_gov_fisheries':
                    results[source_name] = self._search_data_gov_fisheries()
                elif source_name == 'wpacfin_data':
                    results[source_name] = self._check_wpacfin_data()
                else:
                    results[source_name] = self._check_generic_source(source_info)
                    
                # Add status indicator
                if results[source_name].get('accessible', False):
                    print(f"   ✅ Accessible - {results[source_name].get('data_files', 0)} potential files found")
                else:
                    print(f"   ⚠️  Limited access - manual review needed")
                    
            except Exception as e:
                results[source_name] = {
                    'status': 'Error',
                    'error': str(e),
                    'accessible': False
                }
                print(f"   ❌ Error accessing source")
        
        return results
    
    def _check_feat_hawaii_trends(self) -> Dict:
        """Check FEAT Hawaii trends for downloadable data"""
        try:
            # FEAT system may have API endpoints or downloadable reports
            base_url = "https://apps-pifsc.fisheries.noaa.gov/FEAT"
            
            # Try to access the main FEAT page first
            response = requests.get(f"{base_url}/#/", timeout=15)
            
            if response.status_code == 200:
                # Look for data export capabilities
                content = response.text
                
                # Search for download links or API endpoints
                api_patterns = [
                    r'/api/[^"\']+',
                    r'/data/[^"\']+',
                    r'/export/[^"\']+',
                    r'\.csv[^"\']*',
                    r'\.xlsx?[^"\']*'
                ]
                
                potential_endpoints = []
                for pattern in api_patterns:
                    matches = re.findall(pattern, content, re.IGNORECASE)
                    potential_endpoints.extend(matches)
                
                return {
                    'status': 'Accessible',
                    'accessible': True,
                    'potential_endpoints': list(set(potential_endpoints)),
                    'data_types': [
                        'Hawaii longline fleet performance',
                        'Commercial catch statistics',
                        'Revenue and cost analysis',
                        'Fleet composition data',
                        'Fishing effort metrics'
                    ],
                    'next_steps': [
                        'Explore FEAT interface manually for data export options',
                        'Look for CSV download buttons in reports',
                        'Check if API access is available',
                        'Contact PIFSC for bulk data access'
                    ],
                    'priority': 'High - most relevant to Hawaii auction data'
                }
            else:
                return {
                    'status': 'Limited Access',
                    'accessible': False,
                    'message': f'HTTP {response.status_code} - may require authentication'
                }
                
        except Exception as e:
            return {
                'status': 'Connection Failed',
                'accessible': False,
                'error': str(e)
            }
    
    def _search_data_gov_fisheries(self) -> Dict:
        """Search Data.gov for Hawaii fisheries datasets"""
        try:
            search_url = "https://catalog.data.gov/api/3/action/package_search"
            
            # Search for Hawaii fisheries datasets
            search_queries = [
                "hawaii fish auction",
                "hawaii commercial fishing",
                "pacific islands fisheries",
                "honolulu fish market"
            ]
            
            found_datasets = []
            
            for query in search_queries:
                params = {
                    "q": query,
                    "fq": "organization:noaa-gov",
                    "rows": 50
                }
                
                response = requests.get(search_url, params=params, timeout=30)
                
                if response.status_code == 200:
                    data = response.json()
                    datasets = data.get('result', {}).get('results', [])
                    
                    for dataset in datasets:
                        title = dataset.get('title', '').lower()
                        if any(term in title for term in ['hawaii', 'pacific', 'auction', 'price']):
                            found_datasets.append({
                                'title': dataset.get('title'),
                                'id': dataset.get('id'),
                                'notes': dataset.get('notes', '')[:200] + '...',
                                'resources': len(dataset.get('resources', [])),
                                'url': f"https://catalog.data.gov/dataset/{dataset.get('id')}"
                            })
            
            return {
                'status': 'Searchable',
                'accessible': True,
                'datasets_found': len(found_datasets),
                'datasets': found_datasets[:10],  # Top 10 results
                'csv_files_available': sum(1 for d in found_datasets if d['resources'] > 0),
                'next_steps': [
                    'Review dataset details on Data.gov',
                    'Download CSV files from relevant datasets',
                    'Check data quality and format compatibility'
                ]
            }
            
        except Exception as e:
            return {
                'status': 'Search Failed',
                'accessible': False,
                'error': str(e)
            }
    
    def _check_wpacfin_data(self) -> Dict:
        """Check Western Pacific Fishery Information Network for data"""
        try:
            response = requests.get("http://www.wpacfin.org/", timeout=15)
            
            if response.status_code == 200:
                content = response.text
                soup = BeautifulSoup(content, 'html.parser')
                
                # Look for data links
                data_links = []
                for link in soup.find_all('a', href=True):
                    href = link['href']
                    text = link.get_text().lower()
                    
                    if any(term in text for term in ['data', 'download', 'report', 'statistics']):
                        if any(term in text for term in ['hawaii', 'commercial', 'price']):
                            data_links.append({
                                'text': link.get_text().strip(),
                                'url': href if href.startswith('http') else f"http://www.wpacfin.org/{href}"
                            })
                
                return {
                    'status': 'Accessible',
                    'accessible': True,
                    'data_links': data_links,
                    'data_types': [
                        'Commercial purchase reports',
                        'Dealer data submissions',
                        'Price and volume summaries',
                        'Fleet statistics'
                    ],
                    'note': 'WPacFIN is a key source for Pacific commercial fishery data'
                }
            else:
                return {
                    'status': 'Connection Issue',
                    'accessible': False,
                    'status_code': response.status_code
                }
                
        except Exception as e:
            return {
                'status': 'Error',
                'accessible': False,
                'error': str(e)
            }
    
    def _check_generic_source(self, source_info: Dict) -> Dict:
        """Check a generic data source for accessibility"""
        try:
            response = requests.get(source_info['url'], timeout=15)
            
            if response.status_code == 200:
                content = response.text
                
                # Look for download patterns
                download_patterns = [
                    r'href="([^"]*\.csv[^"]*)"',
                    r'href="([^"]*\.xlsx?[^"]*)"',
                    r'href="([^"]*download[^"]*)"',
                    r'href="([^"]*data[^"]*)"'
                ]
                
                potential_files = []
                for pattern in download_patterns:
                    matches = re.findall(pattern, content, re.IGNORECASE)
                    potential_files.extend(matches)
                
                return {
                    'status': 'Accessible',
                    'accessible': True,
                    'url': source_info['url'],
                    'description': source_info['description'],
                    'potential_files': len(potential_files),
                    'file_samples': potential_files[:5]
                }
            else:
                return {
                    'status': 'HTTP Error',
                    'accessible': False,
                    'status_code': response.status_code,
                    'url': source_info['url']
                }
                
        except Exception as e:
            return {
                'status': 'Connection Failed',
                'accessible': False,
                'error': str(e),
                'url': source_info['url']
            }
    
    def download_available_datasets(self, max_files: int = 5) -> List[Dict]:
        """Attempt to download available datasets"""
        downloaded_files = []
        
        # Search for downloadable data first
        search_results = self.search_all_sources()
        
        print(f"\n📥 Attempting to download up to {max_files} datasets...")
        
        # Focus on Data.gov results first (most likely to have direct downloads)
        data_gov_results = search_results.get('data_gov_fisheries', {})
        datasets = data_gov_results.get('datasets', [])
        
        for i, dataset in enumerate(datasets[:max_files]):
            try:
                print(f"\n  Downloading dataset {i+1}: {dataset['title'][:50]}...")
                
                # Get dataset details
                dataset_url = f"https://catalog.data.gov/api/3/action/package_show?id={dataset['id']}"
                response = requests.get(dataset_url, timeout=30)
                
                if response.status_code == 200:
                    package_data = response.json()
                    resources = package_data.get('result', {}).get('resources', [])
                    
                    # Look for CSV resources
                    for resource in resources:
                        if resource.get('format', '').upper() in ['CSV', 'XLSX', 'XLS']:
                            file_url = resource.get('url')
                            if file_url:
                                downloaded_file = self._download_file(file_url, dataset['title'], resource.get('format'))
                                if downloaded_file:
                                    downloaded_files.append(downloaded_file)
                                    break
            
            except Exception as e:
                print(f"    ❌ Error downloading {dataset['title']}: {str(e)}")
                continue
        
        return downloaded_files
    
    def _download_file(self, url: str, dataset_title: str, file_format: str) -> Optional[Dict]:
        """Download a single data file"""
        try:
            response = requests.get(url, timeout=60)
            
            if response.status_code == 200:
                # Create filename
                safe_title = re.sub(r'[^a-zA-Z0-9_-]', '_', dataset_title[:30])
                filename = f"data/historical_{safe_title}_{datetime.now().strftime('%Y%m%d')}.{file_format.lower()}"
                
                # Save file
                import os
                os.makedirs("data", exist_ok=True)
                
                with open(filename, 'wb') as f:
                    f.write(response.content)
                
                # Try to preview the data
                preview = self._preview_data_file(filename, file_format)
                
                print(f"    ✅ Downloaded: {filename}")
                if preview:
                    print(f"    📊 Preview: {preview['rows']} rows, {preview['columns']} columns")
                    print(f"    🔤 Columns: {', '.join(preview['column_names'][:5])}...")
                
                return {
                    'filename': filename,
                    'title': dataset_title,
                    'url': url,
                    'format': file_format,
                    'preview': preview,
                    'download_date': datetime.now().isoformat()
                }
            else:
                print(f"    ❌ HTTP {response.status_code} downloading {url}")
                return None
                
        except Exception as e:
            print(f"    ❌ Error downloading {url}: {str(e)}")
            return None
    
    def _preview_data_file(self, filename: str, file_format: str) -> Optional[Dict]:
        """Preview a downloaded data file"""
        try:
            if file_format.upper() == 'CSV':
                df = pd.read_csv(filename, nrows=10)
            elif file_format.upper() in ['XLSX', 'XLS']:
                df = pd.read_excel(filename, nrows=10)
            else:
                return None
            
            return {
                'rows': len(df),
                'columns': len(df.columns),
                'column_names': list(df.columns),
                'sample_data': df.head(3).to_dict('records')
            }
            
        except Exception as e:
            return {'error': str(e)}
    
    def generate_data_source_report(self) -> str:
        """Generate comprehensive report of available data sources"""
        search_results = self.search_all_sources()
        
        report = [
            "🎯 HAWAII FISH AUCTION HISTORICAL DATA SOURCE REPORT",
            "=" * 65,
            "",
            f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            f"Purpose: Find real historical data to replace synthetic demo data",
            ""
        ]
        
        # Accessible sources
        accessible_sources = [(name, results) for name, results in search_results.items() 
                            if results.get('accessible', False)]
        
        if accessible_sources:
            report.extend([
                "✅ ACCESSIBLE DATA SOURCES:",
                ""
            ])
            
            for source_name, results in accessible_sources:
                report.append(f"🔗 {source_name.upper().replace('_', ' ')}")
                report.append(f"   Status: {results['status']}")
                
                if 'datasets_found' in results:
                    report.append(f"   Datasets: {results['datasets_found']} found")
                if 'potential_endpoints' in results:
                    report.append(f"   Endpoints: {len(results['potential_endpoints'])} potential")
                if 'data_links' in results:
                    report.append(f"   Data Links: {len(results['data_links'])} found")
                
                if 'priority' in results:
                    report.append(f"   Priority: {results['priority']}")
                
                report.append("")
        
        # Priority recommendations
        report.extend([
            "🎯 PRIORITY ACTIONS FOR HISTORICAL DATA:",
            "",
            "1. FEAT HAWAII TRENDS (HIGHEST PRIORITY):",
            "   - Navigate to https://apps-pifsc.fisheries.noaa.gov/FEAT/#/report/hawaii-trends",
            "   - Look for 'Export' or 'Download' buttons in the interface",
            "   - Check for CSV export options in the reports",
            "   - Most relevant to actual Hawaii fish auction data",
            "",
            "2. DATA.GOV NOAA DATASETS:",
            "   - Search results show multiple relevant datasets",
            "   - Direct CSV downloads available",
            "   - Focus on 'Honolulu' and 'Hawaii commercial' datasets",
            "",
            "3. WPACFIN NETWORK:",
            "   - Western Pacific fishery purchase reports",
            "   - Contact for dealer submission data",
            "   - May require registration for access",
            "",
            "4. NOAA INPORT ITEM 6349:",
            "   - Honolulu Retail Monitoring Fish Price Data (2016)",
            "   - Contains actual retail price monitoring data",
            "   - Requires manual navigation and download",
            ""
        ]
        
        # Data integration steps
        report.extend([
            "🔧 INTEGRATION STEPS AFTER DATA ACQUISITION:",
            "",
            "1. Data Cleaning and Standardization:",
            "   - Standardize species names across sources",
            "   - Convert price formats to consistent units",
            "   - Handle missing values and outliers",
            "",
            "2. Database Integration:",
            "   - Store in historical_market_data table",
            "   - Replace synthetic demo data",
            "   - Maintain data source attribution",
            "",
            "3. Model Retraining:",
            "   - Retrain prediction models with real data",
            "   - Validate improved accuracy",
            "   - Update confidence intervals",
            "",
            "4. System Validation:",
            "   - Test predictions against known outcomes",
            "   - Adjust feature importance based on real patterns",
            "   - Fine-tune buyer recommendations",
            ""
        ])
        
        # Technical implementation
        report.extend([
            "💻 TECHNICAL IMPLEMENTATION:",
            "",
            "To integrate downloaded data:",
            "1. Run: python historical_data_finder.py",
            "2. Review downloaded files in /data directory",
            "3. Use data_manager.py to process and load historical data",
            "4. Retrain models with: python prediction_model.py --retrain",
            "5. Verify system accuracy with real data patterns",
            ""
        ])
        
        return "\n".join(report)

def main():
    """Main function to run the historical data finder"""
    finder = HistoricalDataFinder()
    
    # Generate and display report
    report = finder.generate_data_source_report()
    print(report)
    
    # Save report to file
    with open('historical_data_source_report.txt', 'w') as f:
        f.write(report)
    
    print(f"\n📄 Full report saved to: historical_data_source_report.txt")
    
    # Attempt to download some datasets
    print(f"\n🚀 Attempting automatic downloads...")
    downloaded_files = finder.download_available_datasets(max_files=3)
    
    if downloaded_files:
        print(f"\n✅ Successfully downloaded {len(downloaded_files)} files:")
        for file_info in downloaded_files:
            print(f"   📁 {file_info['filename']}")
            print(f"      📊 {file_info['title'][:60]}...")
    else:
        print(f"\n⚠️  No files could be automatically downloaded.")
        print(f"   Manual navigation to data sources required.")

if __name__ == "__main__":
    main()
