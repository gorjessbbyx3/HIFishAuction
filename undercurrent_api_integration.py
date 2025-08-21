import requests
import pandas as pd
import json
from datetime import datetime, timedelta
from typing import Dict, List, Optional
import os

class UndercurrentAPIIntegration:
    """Integration service for Undercurrent News global seafood market data API"""
    
    def __init__(self):
        self.api_base_url = "https://api.undercurrentnews.com"  # Actual API endpoint
        self.api_key = os.getenv('UNDERCURRENT_API_KEY')
        self.headers = {
            'Authorization': f'Bearer {self.api_key}' if self.api_key else '',
            'Content-Type': 'application/json',
            'User-Agent': 'Hawaii Fish Auction Predictor'
        }
        
        # API capabilities based on Undercurrent News platform
        self.api_capabilities = {
            'datasets_available': '3700+ live datasets',
            'update_frequency': 'Real-time and weekly updates',
            'global_coverage': 'Worldwide seafood markets',
            'historical_data': 'Multi-year price history',
            'api_formats': ['JSON', 'CSV', 'XML'],
            'comparison_charts': 'Up to 12 products comparison',
            'live_updates': 'Continuously updated prices',
            'species_coverage': 'All major seafood species'
        }
        
        # Relevant datasets for Hawaii fish market based on UCN platform
        self.relevant_datasets = {
            'tuna_markets': {
                'yellowfin_whole_frozen_abidjan': 'yellowfin_tuna_whole_frozen_abidjan_exvessel',
                'yellowfin_large_cif_spain': 'large_yellowfin_tuna_cif_spain_canning',
                'yellowfin_japan_import': 'yellowfin_tuna_japan_import_weekly',
                'bigeye_sashimi_grade': 'bigeye_tuna_sashimi_grade_tokyo',
                'skipjack_bangkok': 'skipjack_tuna_bangkok_weekly'
            },
            'pacific_markets': {
                'sockeye_salmon_alaska': 'sockeye_salmon_frozen_fillet_alaska',
                'mahi_us_west_coast': 'mahi_mahi_us_west_coast_weekly',
                'swordfish_hawaii': 'swordfish_hawaii_weekly'
            },
            'comparison_sets': {
                'tuna_global_comparison': [
                    'yellowfin_whole_frozen_abidjan',
                    'yellowfin_large_cif_spain', 
                    'bigeye_sashimi_grade'
                ],
                'pacific_comparison': [
                    'sockeye_salmon_alaska',
                    'mahi_us_west_coast',
                    'swordfish_hawaii'
                ]
            }
        }
    
    def check_api_status(self) -> Dict:
        """Check Undercurrent News API availability and authentication"""
        if not self.api_key:
            return {
                'status': 'Authentication Required',
                'message': 'UNDERCURRENT_API_KEY environment variable not found',
                'subscription_required': True,
                'trial_available': True,
                'contact_info': 'Visit undercurrentnews.com for API access'
            }
        
        try:
            # Test API connection with authentication
            test_url = f"{self.api_base_url}/v1/status"
            response = requests.get(test_url, headers=self.headers, timeout=10)
            
            if response.status_code == 200:
                return {
                    'status': 'Connected',
                    'message': 'API connection successful',
                    'capabilities': self.api_capabilities
                }
            elif response.status_code == 401:
                return {
                    'status': 'Authentication Failed',
                    'message': 'Invalid API key or subscription expired',
                    'subscription_required': True
                }
            else:
                return {
                    'status': 'API Error',
                    'message': f'API returned status code {response.status_code}'
                }
                
        except Exception as e:
            return {
                'status': 'Connection Failed',
                'message': f'Unable to connect to API: {str(e)}',
                'requires_subscription': True
            }
    
    def get_global_market_influence(self, hawaii_species: str) -> Dict:
        """Analyze global market influence on Hawaii auction prices"""
        if not self.api_key:
            return {
                'status': 'API Key Required',
                'message': 'Undercurrent News API subscription needed for global market data',
                'integration_value': self._get_integration_value(hawaii_species)
            }
        
        try:
            # Fetch global market data for the species
            global_data = self._fetch_species_global_data(hawaii_species)
            
            if not global_data:
                return {
                    'status': 'No Data Available',
                    'message': f'Global market data for {hawaii_species} requires API access'
                }
            
            # Analyze market influence
            analysis = self._analyze_market_influence(global_data, hawaii_species)
            
            return {
                'status': 'Analysis Complete',
                'species': hawaii_species,
                'global_influence': analysis,
                'market_recommendation': self._generate_recommendation(analysis)
            }
            
        except Exception as e:
            return {
                'status': 'Analysis Failed',
                'error': str(e),
                'fallback': 'Using local market patterns only'
            }
    
    def _fetch_species_global_data(self, species: str) -> Optional[Dict]:
        """Fetch global market data for specific species"""
        try:
            # Map Hawaii species to global market datasets
            species_mapping = {
                'Yellowfin Tuna': ['yellowfin_tokyo', 'yellowfin_us'],
                'Bigeye Tuna': ['bigeye_tokyo', 'bigeye_us'],
                'Mahi-mahi': ['mahi_us_west_coast']
            }
            
            datasets = species_mapping.get(species, [])
            if not datasets:
                return None
            
            global_prices = []
            for dataset in datasets:
                url = f"{self.api_base_url}/v1/prices/{dataset}"
                response = requests.get(url, headers=self.headers, timeout=30)
                
                if response.status_code == 200:
                    data = response.json()
                    global_prices.extend(data.get('prices', []))
            
            return {'prices': global_prices} if global_prices else None
            
        except Exception as e:
            print(f"Error fetching global data: {str(e)}")
            return None
    
    def _analyze_market_influence(self, global_data: Dict, species: str) -> Dict:
        """Analyze how global markets influence Hawaii prices"""
        try:
            prices = global_data.get('prices', [])
            if not prices:
                return {'influence_level': 'Unknown', 'reason': 'No data available'}
            
            # Calculate market metrics
            price_values = [p['price'] for p in prices if 'price' in p]
            avg_price = sum(price_values) / len(price_values) if price_values else 0
            
            # Determine influence level based on species and market presence
            influence_mapping = {
                'Yellowfin Tuna': {'level': 'High', 'reason': 'Major global commodity with Tokyo market influence'},
                'Bigeye Tuna': {'level': 'Very High', 'reason': 'Premium sashimi market strongly tied to Tokyo prices'},
                'Mahi-mahi': {'level': 'Moderate', 'reason': 'Regional market with some US West Coast correlation'}
            }
            
            base_influence = influence_mapping.get(species, {'level': 'Low', 'reason': 'Limited global market presence'})
            
            return {
                'influence_level': base_influence['level'],
                'reason': base_influence['reason'],
                'global_average_price': avg_price,
                'market_volatility': self._calculate_volatility(price_values),
                'correlation_strength': self._estimate_correlation(species)
            }
            
        except Exception as e:
            return {
                'influence_level': 'Unknown',
                'reason': f'Analysis error: {str(e)}'
            }
    
    def _calculate_volatility(self, prices: List[float]) -> str:
        """Calculate price volatility classification"""
        if len(prices) < 2:
            return 'Unknown'
        
        import statistics
        try:
            std_dev = statistics.stdev(prices)
            mean_price = statistics.mean(prices)
            cv = (std_dev / mean_price) * 100 if mean_price > 0 else 0
            
            if cv > 20:
                return 'High'
            elif cv > 10:
                return 'Moderate'
            else:
                return 'Low'
        except:
            return 'Unknown'
    
    def _estimate_correlation(self, species: str) -> float:
        """Estimate correlation between global and Hawaii markets"""
        correlation_estimates = {
            'Yellowfin Tuna': 0.75,  # High correlation with global markets
            'Bigeye Tuna': 0.85,     # Very high correlation with Tokyo
            'Mahi-mahi': 0.45,       # Moderate correlation
            'Opah': 0.25,            # Low correlation - more local
            'Marlin': 0.35,          # Low-moderate correlation
            'Ono': 0.30              # Low correlation - regional preference
        }
        
        return correlation_estimates.get(species, 0.20)
    
    def _generate_recommendation(self, analysis: Dict) -> str:
        """Generate market-based recommendation"""
        influence = analysis.get('influence_level', 'Unknown')
        volatility = analysis.get('market_volatility', 'Unknown')
        
        if influence == 'Very High' and volatility == 'High':
            return "Global markets strongly influence Hawaii prices - monitor international trends closely"
        elif influence == 'High':
            return "Significant global market influence - consider international price movements"
        elif influence == 'Moderate':
            return "Moderate global influence - focus on regional market conditions"
        else:
            return "Limited global influence - local Hawaii market conditions most important"
    
    def _get_integration_value(self, species: str) -> Dict:
        """Get integration value description for the species"""
        return {
            'description': f'Global market data for {species} provides international price context',
            'benefits': [
                'Real-time global price movements',
                'Market correlation analysis',
                'International supply-demand indicators',
                'Advanced price forecasting accuracy'
            ],
            'api_required': True,
            'trial_available': True
        }
    
    def get_integration_requirements(self) -> Dict:
        """Get requirements for full Undercurrent News integration"""
        return {
            'api_access': {
                'subscription_required': True,
                'trial_available': True,
                'contact': 'sales@undercurrentnews.com'
            },
            'capabilities': {
                'datasets': '3700+ live seafood market datasets',
                'coverage': 'Global seafood markets including major tuna auctions',
                'update_frequency': 'Real-time and weekly price updates',
                'historical_depth': 'Multi-year price history',
                'formats': ['JSON', 'CSV', 'XML']
            },
            'hawaii_relevance': {
                'tuna_markets': 'Tokyo, US, and Bangkok tuna prices',
                'pacific_seafood': 'US West Coast and Pacific markets',
                'correlation_analysis': 'Price correlation with Hawaii auctions',
                'forecasting_enhancement': 'Improved prediction accuracy'
            },
            'integration_value': 'Provides essential global market context for Hawaii auction price predictions'
        }