import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional
import sqlite3

class GlobalMarketAnalyzer:
    """Analyze global seafood market impacts on Hawaii fish auction prices"""
    
    def __init__(self, db_path="fish_auction.db"):
        self.db_path = db_path
        
        # Global market influence factors based on species and markets
        self.market_influence_matrix = {
            'Yellowfin Tuna': {
                'tokyo_market': 0.75,      # High influence from Tokyo tuna market
                'spanish_canning': 0.45,   # Moderate influence from European canning
                'abidjan_export': 0.35,    # Some influence from West African exports
                'correlation_strength': 0.80
            },
            'Bigeye Tuna': {
                'tokyo_sashimi': 0.90,     # Very high influence from Tokyo sashimi market
                'japan_import': 0.85,      # High influence from Japan imports
                'us_market': 0.40,         # Moderate US market influence
                'correlation_strength': 0.85
            },
            'Mahi-mahi': {
                'us_west_coast': 0.60,     # Moderate influence from US West Coast
                'pacific_regional': 0.35,  # Some Pacific regional influence
                'correlation_strength': 0.45
            },
            'Opah': {
                'hawaii_local': 0.95,      # Primarily local Hawaii market
                'us_specialty': 0.25,      # Limited US specialty market influence
                'correlation_strength': 0.20
            },
            'Marlin': {
                'pacific_regional': 0.40,  # Some Pacific regional influence
                'japan_market': 0.30,      # Limited Japanese market influence
                'correlation_strength': 0.35
            },
            'Ono': {
                'hawaii_local': 0.85,      # Primarily local Hawaii preference
                'us_specialty': 0.30,      # Some US specialty market influence
                'correlation_strength': 0.25
            }
        }
    
    def analyze_global_price_pressure(self, species: str, current_hawaii_price: float) -> Dict:
        """Analyze global price pressure on Hawaii species"""
        try:
            # Get market influence factors for the species
            influence_data = self.market_influence_matrix.get(species, {})
            
            if not influence_data:
                return {
                    'status': 'Limited Global Influence',
                    'species': species,
                    'influence_level': 'Low',
                    'pressure_analysis': 'Species primarily influenced by local Hawaii market conditions'
                }
            
            # Calculate weighted global influence
            correlation_strength = influence_data.get('correlation_strength', 0.20)
            
            # Simulate global market conditions based on real market patterns
            global_conditions = self._simulate_global_market_conditions(species)
            
            # Calculate price pressure
            pressure_analysis = self._calculate_price_pressure(
                species, current_hawaii_price, global_conditions, correlation_strength
            )
            
            return {
                'status': 'Analysis Complete',
                'species': species,
                'correlation_strength': correlation_strength,
                'global_conditions': global_conditions,
                'pressure_analysis': pressure_analysis,
                'recommendation': self._generate_global_recommendation(pressure_analysis)
            }
            
        except Exception as e:
            return {
                'status': 'Analysis Failed',
                'error': str(e),
                'fallback': 'Using local market analysis only'
            }
    
    def _simulate_global_market_conditions(self, species: str) -> Dict:
        """Simulate global market conditions based on real market patterns"""
        
        # Base market conditions following real market trends
        market_conditions = {
            'Yellowfin Tuna': {
                'tokyo_price_trend': 'stable_to_rising',
                'european_demand': 'steady',
                'global_supply': 'seasonal_variation',
                'price_volatility': 'moderate',
                'seasonal_factor': self._get_seasonal_factor('yellowfin', datetime.now().month)
            },
            'Bigeye Tuna': {
                'tokyo_sashimi_premium': 'high_demand',
                'japan_import_volume': 'stable',
                'sashimi_grade_supply': 'limited',
                'price_volatility': 'high',
                'seasonal_factor': self._get_seasonal_factor('bigeye', datetime.now().month)
            },
            'Mahi-mahi': {
                'us_west_coast_demand': 'restaurant_recovery',
                'pacific_supply': 'weather_dependent',
                'price_volatility': 'moderate_to_high',
                'seasonal_factor': self._get_seasonal_factor('mahi', datetime.now().month)
            }
        }
        
        return market_conditions.get(species, {
            'global_influence': 'limited',
            'price_volatility': 'low',
            'seasonal_factor': 1.0
        })
    
    def _get_seasonal_factor(self, species_type: str, month: int) -> float:
        """Get seasonal factor based on global market patterns"""
        seasonal_patterns = {
            'yellowfin': {
                1: 0.95, 2: 0.95, 3: 1.0, 4: 1.05, 5: 1.15, 6: 1.20,
                7: 1.25, 8: 1.20, 9: 1.10, 10: 1.05, 11: 0.95, 12: 0.90
            },
            'bigeye': {
                1: 1.15, 2: 1.10, 3: 1.05, 4: 0.95, 5: 0.85, 6: 0.80,
                7: 0.85, 8: 0.95, 9: 1.05, 10: 1.15, 11: 1.20, 12: 1.15
            },
            'mahi': {
                1: 1.0, 2: 1.0, 3: 1.10, 4: 1.20, 5: 1.15, 6: 0.85,
                7: 0.80, 8: 0.90, 9: 1.10, 10: 1.15, 11: 1.05, 12: 0.95
            }
        }
        
        return seasonal_patterns.get(species_type, {}).get(month, 1.0)
    
    def _calculate_price_pressure(self, species: str, current_price: float, 
                                global_conditions: Dict, correlation: float) -> Dict:
        """Calculate price pressure from global markets"""
        try:
            seasonal_factor = global_conditions.get('seasonal_factor', 1.0)
            volatility = global_conditions.get('price_volatility', 'moderate')
            
            # Calculate pressure indicators
            seasonal_pressure = (seasonal_factor - 1.0) * 100  # Percentage change
            
            # Volatility impact on price uncertainty
            volatility_impact = {
                'low': 0.05,
                'moderate': 0.10,
                'moderate_to_high': 0.15,
                'high': 0.20
            }.get(volatility, 0.10)
            
            # Global price pressure calculation
            global_pressure = seasonal_pressure * correlation
            price_uncertainty = current_price * volatility_impact
            
            # Market pressure classification
            if abs(global_pressure) > 10:
                pressure_level = 'High'
            elif abs(global_pressure) > 5:
                pressure_level = 'Moderate'
            else:
                pressure_level = 'Low'
            
            direction = 'Upward' if global_pressure > 0 else 'Downward' if global_pressure < 0 else 'Neutral'
            
            return {
                'pressure_level': pressure_level,
                'direction': direction,
                'global_pressure_percent': round(global_pressure, 2),
                'price_uncertainty_range': round(price_uncertainty, 2),
                'correlation_strength': correlation,
                'volatility_rating': volatility,
                'seasonal_impact': round(seasonal_pressure, 2)
            }
            
        except Exception as e:
            return {
                'pressure_level': 'Unknown',
                'direction': 'Unknown',
                'error': str(e)
            }
    
    def _generate_global_recommendation(self, pressure_analysis: Dict) -> str:
        """Generate recommendation based on global market pressure"""
        pressure_level = pressure_analysis.get('pressure_level', 'Unknown')
        direction = pressure_analysis.get('direction', 'Unknown')
        global_pressure = pressure_analysis.get('global_pressure_percent', 0)
        
        if pressure_level == 'High' and direction == 'Upward':
            return f"Strong global market pressure (+{abs(global_pressure):.1f}%) - Consider prepaying to avoid price increases"
        elif pressure_level == 'High' and direction == 'Downward':
            return f"Global markets declining (-{abs(global_pressure):.1f}%) - Wait for better prices"
        elif pressure_level == 'Moderate' and direction == 'Upward':
            return f"Moderate upward pressure (+{abs(global_pressure):.1f}%) - Monitor global trends closely"
        elif pressure_level == 'Moderate' and direction == 'Downward':
            return f"Moderate downward pressure (-{abs(global_pressure):.1f}%) - Potential buying opportunity"
        else:
            return "Global markets stable - Focus on local Hawaii market conditions"
    
    def get_market_correlation_analysis(self, species: str) -> Dict:
        """Get detailed market correlation analysis for species"""
        influence_data = self.market_influence_matrix.get(species, {})
        
        if not influence_data:
            return {
                'species': species,
                'correlation_type': 'Local Market Dependent',
                'primary_influences': ['Hawaii local demand', 'Regional supply conditions'],
                'global_sensitivity': 'Low'
            }
        
        correlation = influence_data.get('correlation_strength', 0.20)
        
        # Identify primary global market influences
        market_influences = {k: v for k, v in influence_data.items() 
                           if k != 'correlation_strength' and v > 0.3}
        
        # Sort by influence strength
        primary_markets = sorted(market_influences.items(), 
                               key=lambda x: x[1], reverse=True)[:3]
        
        sensitivity_level = 'High' if correlation > 0.7 else 'Moderate' if correlation > 0.4 else 'Low'
        
        return {
            'species': species,
            'correlation_strength': correlation,
            'global_sensitivity': sensitivity_level,
            'primary_market_influences': [market for market, _ in primary_markets],
            'correlation_type': self._get_correlation_type(correlation),
            'market_integration': self._get_market_integration_level(species, correlation)
        }
    
    def _get_correlation_type(self, correlation: float) -> str:
        """Classify correlation type"""
        if correlation > 0.7:
            return 'Globally Integrated'
        elif correlation > 0.4:
            return 'Regionally Influenced'
        else:
            return 'Locally Driven'
    
    def _get_market_integration_level(self, species: str, correlation: float) -> str:
        """Describe market integration level"""
        if correlation > 0.8:
            return f"{species} prices closely follow international market movements"
        elif correlation > 0.5:
            return f"{species} prices moderately influenced by global market trends"
        else:
            return f"{species} prices primarily driven by local Hawaii market conditions"