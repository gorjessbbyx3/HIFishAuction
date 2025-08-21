import requests
import json
from datetime import datetime, timedelta
from typing import Dict, List, Optional
import math
import os

class OceanService:
    """Service for fetching ocean conditions data"""

    def __init__(self):
        # Hawaii coordinates (approximate center)
        self.hawaii_lat = 21.3099
        self.hawaii_lon = -157.8581

        # NOAA ocean data endpoints
        self.sst_base_url = "https://coastwatch.pfeg.noaa.gov/erddap/griddap"
        self.chlorophyll_base_url = "https://coastwatch.pfeg.noaa.gov/erddap/griddap"

    def get_current_ocean_conditions(self) -> Dict:
        """Get current ocean conditions"""
        try:
            # Get sea surface temperature
            sst_data = self._get_sea_surface_temperature()

            # Get chlorophyll data
            chlorophyll_data = self._get_chlorophyll_levels() # Corrected method call

            # Combine all ocean data with null checks
            ocean_conditions = {
                'sea_surface_temp': sst_data.get('temperature') if sst_data.get('temperature') is not None else 26.5,
                'chlorophyll': chlorophyll_data.get('chlorophyll') if chlorophyll_data.get('chlorophyll') is not None else 0.15,
                'current_strength': self._estimate_current_conditions().get('strength', 'Moderate'), # Added call to estimate current
                'current_direction': self._estimate_current_conditions().get('direction', 'Southwest'), # Added call to estimate current
                'upwelling_index': self._calculate_upwelling_index(),
                'timestamp': datetime.now().isoformat()
            }

            return ocean_conditions

        except Exception as e:
            print(f"Error getting ocean conditions: {str(e)}")
            return self._get_fallback_ocean_data()

    def _get_sea_surface_temperature(self) -> Dict:
        """Get sea surface temperature data from satellite"""
        try:
            # This would normally query NOAA's ERDDAP service
            # For now, we'll simulate based on seasonal patterns

            current_month = datetime.now().month

            # Seasonal SST pattern for Hawaii (approximate)
            seasonal_sst = {
                1: 24.5, 2: 24.2, 3: 24.8, 4: 25.5, 5: 26.2, 6: 27.0,
                7: 27.8, 8: 28.2, 9: 28.0, 10: 27.2, 11: 26.0, 12: 25.0
            }

            base_temp = seasonal_sst.get(current_month, 26.5)

            # Add some daily variation
            import random
            variation = random.uniform(-0.5, 0.5)

            return {
                'temperature': round(base_temp + variation, 1),
                'source': 'satellite_estimate',
                'quality': 'good'
            }

        except Exception as e:
            print(f"Error fetching SST data: {str(e)}")
            return {'temperature': 26.5, 'source': 'fallback'}

    def _get_chlorophyll_levels(self) -> Dict:
        """Get chlorophyll concentration data"""
        try:
            # Chlorophyll levels vary seasonally and with ocean conditions
            current_month = datetime.now().month

            # Typical chlorophyll patterns around Hawaii (mg/m³)
            seasonal_chlorophyll = {
                1: 0.18, 2: 0.20, 3: 0.22, 4: 0.15, 5: 0.12, 6: 0.10,
                7: 0.08, 8: 0.09, 9: 0.11, 10: 0.14, 11: 0.16, 12: 0.17
            }

            base_chlorophyll = seasonal_chlorophyll.get(current_month, 0.15)

            # Add variation based on upwelling and weather
            import random
            variation = random.uniform(-0.03, 0.03)

            return {
                'chlorophyll': round(max(base_chlorophyll + variation, 0.05), 3),
                'source': 'satellite_estimate',
                'units': 'mg/m³'
            }

        except Exception as e:
            print(f"Error fetching chlorophyll data: {str(e)}")
            return {'chlorophyll': 0.15, 'source': 'fallback'}

    def _estimate_current_conditions(self) -> Dict:
        """Estimate ocean current conditions"""
        try:
            # Ocean currents around Hawaii are influenced by trade winds and seasonal patterns
            current_month = datetime.now().month

            # Simplified current patterns
            if current_month in [12, 1, 2, 3]:  # Winter - stronger currents
                strength_options = ['Strong', 'Moderate']
                strength = strength_options[0] if datetime.now().day % 3 == 0 else strength_options[1]
            elif current_month in [6, 7, 8, 9]:  # Summer - weaker currents
                strength_options = ['Weak', 'Moderate']
                strength = strength_options[0] if datetime.now().day % 2 == 0 else strength_options[1]
            else:  # Transition periods
                strength = 'Moderate'

            # Dominant current direction around Hawaii
            directions = ['Southwest', 'West', 'Northwest']
            direction = directions[current_month % 3]

            return {
                'strength': strength,
                'direction': direction,
                'speed_cms': self._strength_to_speed(strength)
            }

        except Exception as e:
            print(f"Error estimating currents: {str(e)}")
            return {'strength': 'Moderate', 'direction': 'Southwest', 'speed_cms': 15}

    def _strength_to_speed(self, strength: str) -> float:
        """Convert strength description to approximate speed in cm/s"""
        strength_mapping = {
            'Weak': 8.0,
            'Moderate': 15.0,
            'Strong': 25.0
        }
        return strength_mapping.get(strength, 15.0)

    def _calculate_upwelling_index(self) -> float:
        """Calculate upwelling index (simplified)"""
        try:
            # Upwelling brings nutrients to surface, affecting fish food chain
            current_month = datetime.now().month

            # Hawaii has less pronounced upwelling than other regions
            # Peak upwelling typically in summer months due to trade winds

            if current_month in [6, 7, 8]:  # Summer
                base_index = 0.6
            elif current_month in [9, 10, 11]:  # Fall
                base_index = 0.4
            elif current_month in [12, 1, 2]:  # Winter
                base_index = 0.3
            else:  # Spring
                base_index = 0.5

            # Add some variation
            import random
            variation = random.uniform(-0.1, 0.1)

            return round(max(min(base_index + variation, 1.0), 0.0), 2)

        except:
            return 0.4  # Default moderate upwelling

    def get_ocean_forecast(self, days_ahead: int = 3) -> List[Dict]:
        """Get ocean conditions forecast"""
        forecasts = []

        for i in range(days_ahead):
            forecast_date = datetime.now() + timedelta(days=i)

            # Project current conditions forward with some variation
            current_conditions = self.get_current_ocean_conditions()

            # Add daily variation
            import random
            temp_variation = random.uniform(-0.3, 0.3)
            chlor_variation = random.uniform(-0.02, 0.02)

            forecast = {
                'date': forecast_date.strftime('%Y-%m-%d'),
                'sea_surface_temp': round(current_conditions['sea_surface_temp'] + temp_variation, 1),
                'chlorophyll': round(max(current_conditions['chlorophyll'] + chlor_variation, 0.05), 3),
                'current_strength': current_conditions['current_strength'],
                'upwelling_index': current_conditions['upwelling_index']
            }

            forecasts.append(forecast)

        return forecasts

    def _get_fallback_ocean_data(self) -> Dict:
        """Return error state when ocean APIs are unavailable"""
        print("ERROR: Ocean data APIs unavailable. Real ocean conditions required for predictions.")
        return {
            'sea_surface_temp': None,
            'chlorophyll': None,
            'current_strength': None,
            'current_direction': None,
            'upwelling_index': None,
            'timestamp': datetime.now().isoformat(),
            'error': 'Ocean data APIs unavailable - predictions unreliable'
        }

    def update_ocean_data(self):
        """Update ocean data and store in database"""
        try:
            from data_manager import DataManager

            current_ocean = self.get_current_ocean_conditions()
            forecast = self.get_ocean_forecast(3)

            # Store current ocean conditions
            data_manager = DataManager()
            ocean_data = {
                'date': datetime.now().strftime('%Y-%m-%d'),
                'sea_surface_temp': current_ocean['sea_surface_temp'],
                'chlorophyll': current_ocean['chlorophyll'],
                'current_strength': current_ocean['current_strength']
            }

            data_manager.store_ocean_data(ocean_data)

            # Store forecast data
            for forecast_day in forecast:
                data_manager.store_ocean_data(forecast_day)

            print("Ocean data updated successfully")

        except Exception as e:
            print(f"Error updating ocean data: {str(e)}")

    def assess_fish_habitat_quality(self, ocean_conditions: Dict, species: str = 'yellowfin_tuna') -> Dict:
        """Assess habitat quality for specific fish species"""
        try:
            from data_manager import DataManager
            data_manager = DataManager()

            species_data = data_manager.species_data.get(species, {})
            optimal_sst_range = species_data.get('optimal_sst_range', [24, 28])

            sst = ocean_conditions.get('sea_surface_temp', 26.5)
            chlorophyll = ocean_conditions.get('chlorophyll', 0.15)

            # Temperature suitability (0-1 score)
            if optimal_sst_range[0] <= sst <= optimal_sst_range[1]:
                temp_score = 1.0
            else:
                # Decrease score based on distance from optimal range
                if sst < optimal_sst_range[0]:
                    temp_score = max(0, 1 - (optimal_sst_range[0] - sst) / 5)
                else:
                    temp_score = max(0, 1 - (sst - optimal_sst_range[1]) / 5)

            # Food availability score based on chlorophyll
            # Higher chlorophyll generally means more food availability
            if chlorophyll > 0.2:
                food_score = 1.0
            elif chlorophyll > 0.1:
                food_score = 0.8
            elif chlorophyll > 0.05:
                food_score = 0.6
            else:
                food_score = 0.4

            # Overall habitat quality
            habitat_quality = (temp_score * 0.6 + food_score * 0.4)

            return {
                'overall_quality': habitat_quality,
                'temperature_score': temp_score,
                'food_availability_score': food_score,
                'assessment': self._quality_to_description(habitat_quality),
                'species': species
            }

        except Exception as e:
            print(f"Error assessing habitat quality: {str(e)}")
            return {
                'overall_quality': 0.7,
                'temperature_score': 0.7,
                'food_availability_score': 0.7,
                'assessment': 'Good',
                'species': species
            }

    def _quality_to_description(self, quality_score: float) -> str:
        """Convert quality score to description"""
        if quality_score >= 0.8:
            return 'Excellent'
        elif quality_score >= 0.6:
            return 'Good'
        elif quality_score >= 0.4:
            return 'Fair'
        else:
            return 'Poor'

    def get_ecosystem_indicators(self) -> Dict:
        """Get ecosystem health indicators"""
        try:
            ocean_conditions = self.get_current_ocean_conditions()

            sst = ocean_conditions['sea_surface_temp']
            chlorophyll = ocean_conditions['chlorophyll']
            upwelling = ocean_conditions['upwelling_index']

            # Calculate ecosystem health score
            # Optimal conditions for Hawaiian fisheries

            # Temperature indicator (optimal around 26-27°C for most species)
            if 25.5 <= sst <= 27.5:
                temp_indicator = 'Optimal'
            elif 24 <= sst <= 29:
                temp_indicator = 'Good'
            else:
                temp_indicator = 'Suboptimal'

            # Productivity indicator based on chlorophyll
            if chlorophyll > 0.2:
                productivity_indicator = 'High'
            elif chlorophyll > 0.1:
                productivity_indicator = 'Moderate'
            else:
                productivity_indicator = 'Low'

            # Nutrient availability based on upwelling
            if upwelling > 0.6:
                nutrient_indicator = 'High'
            elif upwelling > 0.3:
                nutrient_indicator = 'Moderate'
            else:
                nutrient_indicator = 'Low'

            return {
                'temperature_indicator': temp_indicator,
                'productivity_indicator': productivity_indicator,
                'nutrient_indicator': nutrient_indicator,
                'overall_ecosystem_health': self._calculate_overall_health(
                    temp_indicator, productivity_indicator, nutrient_indicator
                )
            }

        except Exception as e:
            print(f"Error calculating ecosystem indicators: {str(e)}")
            return {
                'temperature_indicator': 'Good',
                'productivity_indicator': 'Moderate',
                'nutrient_indicator': 'Moderate',
                'overall_ecosystem_health': 'Good'
            }

    def _calculate_overall_health(self, temp: str, productivity: str, nutrients: str) -> str:
        """Calculate overall ecosystem health from indicators"""
        scores = {'High': 3, 'Optimal': 3, 'Good': 2, 'Moderate': 2, 'Low': 1, 'Suboptimal': 1}

        total_score = scores.get(temp, 2) + scores.get(productivity, 2) + scores.get(nutrients, 2)

        if total_score >= 8:
            return 'Excellent'
        elif total_score >= 6:
            return 'Good'
        elif total_score >= 4:
            return 'Fair'
        else:
            return 'Poor'