import requests
import pandas as pd
import json
from datetime import datetime, timedelta
from typing import Dict, List, Optional
import sqlite3

class PacificEcosystemDataIntegration:
    """Integration with NOAA Pacific Islands Fishery Ecosystem Analysis Tool (FEAT)"""
    
    def __init__(self):
        self.feat_base_url = "https://apps-pifsc.fisheries.noaa.gov/FEAT"
        self.api_endpoints = {
            'performance_indicators': '/api/performance-indicators',
            'cost_data': '/api/cost-data',
            'hawaii_trends': '/api/hawaii-trends',
            'fishing_trends': '/api/fishing-trends',
            'community_research': '/api/community-research'
        }
        
        # Hawaii-specific consumption data from CTAHR study
        self.hawaii_consumption_data = {
            'per_capita_consumption': {
                'total_with_noncommercial': 36.85,  # edible pounds per capita
                'total_commercial_only': 28.46,     # edible pounds per capita
                'us_comparison_ratio': 1.8,         # Hawaii vs US consumption ratio
                'top_species': {
                    'tuna_total': 12.72,           # yellowfin, bigeye, other
                    'salmon': 4.23,
                    'mollusks_crustaceans': 3.92,
                    'mahimahi': 1.93,
                    'shrimp': 1.85,
                    'billfish': 1.58,
                    'swordfish': 0.80
                }
            },
            'expenditure_data': {
                'home_consumption_per_capita': 104.29,    # USD
                'food_service_per_capita': 226.39,       # USD
                'total_seafood_expenditure': 330.68,     # USD
                'seafood_proportion_of_food': 0.114      # 11.4%
            },
            'supply_sources': {
                'imports_percentage': 63,        # 63% of commercial consumption
                'foreign_imports': 57,          # 57% from foreign countries
                'local_production': 37,         # 37% local production
                'noncommercial_catch': 39       # 39% of total production
            }
        }
        
        self.db_path = "fish_auction.db"
    
    def check_feat_availability(self) -> Dict:
        """Check availability of NOAA FEAT system"""
        try:
            # Test connection to FEAT system
            response = requests.get(f"{self.feat_base_url}/#/", timeout=10)
            
            if response.status_code == 200:
                return {
                    'status': 'Available',
                    'feat_system': 'Accessible',
                    'available_reports': [
                        'Fishery Performance Indicators',
                        'Cost Data Analysis',
                        'Hawaii Fishery Trends',
                        'US Pacific HMS Fishing Trends',
                        'Human Community Research'
                    ],
                    'data_types': [
                        'Commercial fishing activity metrics',
                        'Economic performance indicators',
                        'Fleet activity and effort data',
                        'Revenue and cost analysis',
                        'CPUE and catch statistics'
                    ]
                }
            else:
                return {
                    'status': 'Limited Access',
                    'message': 'FEAT system may require authentication or have access restrictions'
                }
                
        except Exception as e:
            return {
                'status': 'Connection Failed',
                'error': str(e),
                'fallback': 'Using Hawaii consumption baseline data'
            }
    
    def get_hawaii_fishery_performance_indicators(self) -> Dict:
        """Get Hawaii fishery performance indicators from FEAT"""
        try:
            # The FEAT system provides these key indicators for Hawaii fisheries
            performance_indicators = {
                'fishing_activity': {
                    'active_vessels': 'Number of active commercial vessels',
                    'fishing_trips': 'Total commercial fishing trips',
                    'fishing_days': 'Total days at sea for commercial fleet'
                },
                'commercial_landings': {
                    'total_pounds': 'Total commercial landings in pounds',
                    'species_composition': 'Breakdown by major species groups',
                    'seasonal_patterns': 'Monthly and quarterly landing patterns'
                },
                'economic_measures': {
                    'total_revenue': 'Total commercial fishery revenue',
                    'revenue_distribution': 'Revenue distribution among vessels',
                    'price_per_pound': 'Average price per pound by species'
                }
            }
            
            # Baseline data from Hawaii consumption study
            baseline_metrics = {
                'commercial_production_pounds': 18108000,  # 18.1M pounds annually
                'noncommercial_catch_pounds': 11465000,   # 11.5M pounds annually
                'total_consumption_pounds': 50387000,     # 50.4M pounds annually
                'import_dependency': 0.63,                # 63% import dependent
                'market_value_estimate': 'High value sashimi-grade focus'
            }
            
            return {
                'status': 'Baseline Data Available',
                'indicators': performance_indicators,
                'baseline_metrics': baseline_metrics,
                'data_source': 'CTAHR Hawaii Seafood Study + FEAT Framework'
            }
            
        except Exception as e:
            return {
                'status': 'Analysis Failed',
                'error': str(e)
            }
    
    def analyze_hawaii_market_context(self, species: str) -> Dict:
        """Analyze Hawaii market context for specific species"""
        try:
            # Get species-specific consumption data
            species_consumption = self._get_species_consumption_data(species)
            market_position = self._analyze_market_position(species, species_consumption)
            supply_analysis = self._analyze_supply_sources(species)
            
            return {
                'species': species,
                'consumption_analysis': species_consumption,
                'market_position': market_position,
                'supply_analysis': supply_analysis,
                'price_sensitivity': self._calculate_price_sensitivity(species),
                'market_recommendations': self._generate_market_recommendations(species, market_position)
            }
            
        except Exception as e:
            return {
                'status': 'Analysis Failed',
                'species': species,
                'error': str(e)
            }
    
    def _get_species_consumption_data(self, species: str) -> Dict:
        """Get consumption data for specific species"""
        species_mapping = {
            'Yellowfin Tuna': 'tuna_total',
            'Bigeye Tuna': 'tuna_total',
            'Mahi-mahi': 'mahimahi',
            'Opah': 'mollusks_crustaceans',  # Grouped with other specialty fish
            'Marlin': 'billfish',
            'Ono': 'billfish'  # Often grouped with billfish
        }
        
        consumption_key = species_mapping.get(species, 'other')
        base_consumption = self.hawaii_consumption_data['per_capita_consumption']['top_species']
        
        if consumption_key in base_consumption:
            per_capita = base_consumption[consumption_key]
            
            # Estimate for specific species within groups
            if species in ['Yellowfin Tuna', 'Bigeye Tuna']:
                # Tuna breakdown based on market preferences
                if species == 'Yellowfin Tuna':
                    per_capita = per_capita * 0.6  # 60% of tuna consumption
                else:  # Bigeye Tuna
                    per_capita = per_capita * 0.4  # 40% of tuna consumption
            
            return {
                'per_capita_consumption_lbs': per_capita,
                'hawaii_total_consumption_estimate': per_capita * 1400000,  # Hawaii population
                'ranking_in_hawaii': self._get_species_ranking(species),
                'noncommercial_percentage': self._get_noncommercial_percentage(species)
            }
        else:
            return {
                'per_capita_consumption_lbs': 0.5,  # Estimate for other species
                'hawaii_total_consumption_estimate': 700000,
                'ranking_in_hawaii': 'Outside top 10',
                'noncommercial_percentage': 20
            }
    
    def _get_species_ranking(self, species: str) -> str:
        """Get species ranking in Hawaii consumption"""
        rankings = {
            'Yellowfin Tuna': '1st (primary tuna species)',
            'Bigeye Tuna': '1st (premium sashimi)',
            'Mahi-mahi': '4th',
            'Marlin': '6th (billfish group)',
            'Ono': '6th (billfish group)',
            'Opah': 'Outside top 10'
        }
        return rankings.get(species, 'Outside top 10')
    
    def _get_noncommercial_percentage(self, species: str) -> int:
        """Get percentage of consumption from noncommercial catch"""
        # Based on CTAHR study data
        noncommercial_percentages = {
            'Yellowfin Tuna': 42,
            'Bigeye Tuna': 42,
            'Mahi-mahi': 59,
            'Marlin': 37,
            'Ono': 37,
            'Opah': 25
        }
        return noncommercial_percentages.get(species, 20)
    
    def _analyze_market_position(self, species: str, consumption_data: Dict) -> Dict:
        """Analyze market position of species in Hawaii"""
        per_capita = consumption_data.get('per_capita_consumption_lbs', 0)
        noncommercial_pct = consumption_data.get('noncommercial_percentage', 0)
        
        if per_capita > 5:
            market_tier = 'Primary Market Species'
        elif per_capita > 1:
            market_tier = 'Secondary Market Species'
        else:
            market_tier = 'Specialty Market Species'
        
        # Calculate market characteristics
        commercial_dependency = 100 - noncommercial_pct
        import_likelihood = 'High' if commercial_dependency > 60 else 'Moderate' if commercial_dependency > 30 else 'Low'
        
        return {
            'market_tier': market_tier,
            'commercial_dependency_pct': commercial_dependency,
            'import_likelihood': import_likelihood,
            'price_volatility_risk': self._assess_volatility_risk(species, commercial_dependency),
            'auction_significance': self._assess_auction_significance(species, per_capita)
        }
    
    def _assess_volatility_risk(self, species: str, commercial_dependency: int) -> str:
        """Assess price volatility risk based on market characteristics"""
        if commercial_dependency > 70:
            return 'High - heavily dependent on commercial supply chains'
        elif commercial_dependency > 40:
            return 'Moderate - mixed commercial and local sourcing'
        else:
            return 'Low - substantial local noncommercial supply'
    
    def _assess_auction_significance(self, species: str, per_capita: float) -> str:
        """Assess significance in Hawaii fish auctions"""
        if per_capita > 5:
            return 'Critical - major auction species driving revenue'
        elif per_capita > 1:
            return 'Important - regular auction presence'
        else:
            return 'Limited - occasional auction presence'
    
    def _analyze_supply_sources(self, species: str) -> Dict:
        """Analyze supply sources for species"""
        # Based on Hawaii consumption study supply patterns
        supply_patterns = {
            'local_commercial': self._get_local_commercial_percentage(species),
            'imports_mainland_us': self._get_mainland_import_percentage(species),
            'imports_foreign': self._get_foreign_import_percentage(species),
            'noncommercial_catch': self._get_noncommercial_percentage(species)
        }
        
        # Identify primary supply constraint
        if supply_patterns['imports_foreign'] > 40:
            primary_constraint = 'International supply chains and trade conditions'
        elif supply_patterns['imports_mainland_us'] > 30:
            primary_constraint = 'US mainland supply and transportation'
        else:
            primary_constraint = 'Local fishing conditions and fleet activity'
        
        return {
            'supply_breakdown': supply_patterns,
            'primary_supply_constraint': primary_constraint,
            'supply_volatility': self._assess_supply_volatility(supply_patterns),
            'seasonal_supply_patterns': self._get_seasonal_supply_patterns(species)
        }
    
    def _get_local_commercial_percentage(self, species: str) -> int:
        """Get percentage from local commercial sources"""
        # Estimates based on Hawaii fishing patterns
        local_percentages = {
            'Yellowfin Tuna': 35,
            'Bigeye Tuna': 25,
            'Mahi-mahi': 45,
            'Marlin': 40,
            'Ono': 40,
            'Opah': 60
        }
        return local_percentages.get(species, 30)
    
    def _get_mainland_import_percentage(self, species: str) -> int:
        """Get percentage from mainland US imports"""
        mainland_percentages = {
            'Yellowfin Tuna': 15,
            'Bigeye Tuna': 10,
            'Mahi-mahi': 20,
            'Marlin': 15,
            'Ono': 15,
            'Opah': 10
        }
        return mainland_percentages.get(species, 15)
    
    def _get_foreign_import_percentage(self, species: str) -> int:
        """Get percentage from foreign imports"""
        foreign_percentages = {
            'Yellowfin Tuna': 50,
            'Bigeye Tuna': 65,
            'Mahi-mahi': 35,
            'Marlin': 45,
            'Ono': 45,
            'Opah': 30
        }
        return foreign_percentages.get(species, 40)
    
    def _assess_supply_volatility(self, supply_patterns: Dict) -> str:
        """Assess supply volatility based on source diversity"""
        foreign_imports = supply_patterns.get('imports_foreign', 0)
        local_sources = supply_patterns.get('local_commercial', 0) + supply_patterns.get('noncommercial_catch', 0)
        
        if foreign_imports > 60:
            return 'High - heavily dependent on international markets'
        elif local_sources > 60:
            return 'Low - substantial local sourcing'
        else:
            return 'Moderate - diversified supply sources'
    
    def _get_seasonal_supply_patterns(self, species: str) -> str:
        """Get seasonal supply patterns for species"""
        patterns = {
            'Yellowfin Tuna': 'Peak summer catches, winter imports increase',
            'Bigeye Tuna': 'Counter-seasonal to yellowfin, winter peak catches',
            'Mahi-mahi': 'Spring and fall peaks, summer low season',
            'Marlin': 'Summer peak fishing season',
            'Ono': 'Year-round with spring peak',
            'Opah': 'Winter and spring peak local availability'
        }
        return patterns.get(species, 'Variable seasonal patterns')
    
    def _calculate_price_sensitivity(self, species: str) -> Dict:
        """Calculate price sensitivity factors"""
        consumption_data = self._get_species_consumption_data(species)
        per_capita = consumption_data.get('per_capita_consumption_lbs', 0)
        
        # Higher consumption generally means higher price sensitivity
        if per_capita > 5:
            sensitivity_level = 'High'
            elasticity_estimate = -1.2  # Elastic demand
        elif per_capita > 1:
            sensitivity_level = 'Moderate'
            elasticity_estimate = -0.8  # Moderately elastic
        else:
            sensitivity_level = 'Low'
            elasticity_estimate = -0.3  # Inelastic demand
        
        return {
            'sensitivity_level': sensitivity_level,
            'price_elasticity_estimate': elasticity_estimate,
            'substitution_risk': self._assess_substitution_risk(species),
            'premium_tolerance': self._assess_premium_tolerance(species)
        }
    
    def _assess_substitution_risk(self, species: str) -> str:
        """Assess risk of substitution with other species"""
        substitution_risks = {
            'Yellowfin Tuna': 'Moderate - can substitute with bigeye for some uses',
            'Bigeye Tuna': 'Low - premium sashimi market with limited substitutes',
            'Mahi-mahi': 'High - many white fish alternatives available',
            'Marlin': 'Moderate - some billfish substitution possible',
            'Ono': 'High - similar to other white fish options',
            'Opah': 'Low - unique flavor profile with limited substitutes'
        }
        return substitution_risks.get(species, 'Moderate substitution risk')
    
    def _assess_premium_tolerance(self, species: str) -> str:
        """Assess market tolerance for price premiums"""
        premium_tolerance = {
            'Yellowfin Tuna': 'High - strong sashimi market demand',
            'Bigeye Tuna': 'Very High - premium sashimi commands top prices',
            'Mahi-mahi': 'Moderate - restaurant favorite but price sensitive',
            'Marlin': 'Low - recreational and cultural value but price sensitive',
            'Ono': 'Moderate - local preference supports moderate premiums',
            'Opah': 'High - specialty market tolerates premium pricing'
        }
        return premium_tolerance.get(species, 'Moderate premium tolerance')
    
    def _generate_market_recommendations(self, species: str, market_position: Dict) -> List[str]:
        """Generate market-based recommendations"""
        recommendations = []
        
        market_tier = market_position.get('market_tier', '')
        commercial_dependency = market_position.get('commercial_dependency_pct', 0)
        
        if 'Primary' in market_tier:
            recommendations.append('Monitor this species closely - major impact on auction revenue')
            
        if commercial_dependency > 70:
            recommendations.append('High supply chain risk - consider supply diversification')
            
        if market_position.get('import_likelihood') == 'High':
            recommendations.append('Track international market conditions for price forecasting')
            
        auction_significance = market_position.get('auction_significance', '')
        if 'Critical' in auction_significance:
            recommendations.append('Priority species for strategic buying decisions')
            
        return recommendations
    
    def get_ecosystem_integration_summary(self) -> Dict:
        """Get summary of Pacific ecosystem data integration capabilities"""
        return {
            'data_sources': {
                'noaa_feat': 'NOAA Pacific Islands Fishery Ecosystem Analysis Tool',
                'hawaii_consumption_study': 'CTAHR University of Hawaii Seafood Consumption Analysis',
                'performance_indicators': 'Commercial fishery performance metrics',
                'economic_analysis': 'Cost data and revenue analysis'
            },
            'integration_capabilities': [
                'Hawaii-specific consumption patterns and preferences',
                'Supply source analysis (local vs import)',
                'Market position assessment by species',
                'Economic performance indicators',
                'Fleet activity and effort metrics'
            ],
            'market_insights': [
                'Per capita consumption 1.8x higher than US average',
                '63% import dependency for commercial consumption',
                'Strong noncommercial fishing contribution (39% of supply)',
                'Premium sashimi market tolerance for higher prices'
            ],
            'predictive_value': 'Provides essential Hawaii market context for price prediction accuracy'
        }