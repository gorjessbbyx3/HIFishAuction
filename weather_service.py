import requests
import json
from datetime import datetime, timedelta
from typing import Dict, List, Optional
import os
import re
import random

class WeatherService:
    """Service for fetching weather data from NOAA API"""

    def __init__(self):
        self.base_url = "https://api.weather.gov"
        self.honolulu_grid = {"office": "HFO", "gridX": 66, "gridY": 86}  # Honolulu grid point
        self.headers = {
            'User-Agent': 'Hawaii Fish Auction Predictor (your-email@domain.com)'
        }

    def get_current_weather(self) -> Dict:
        """Get current weather conditions for Honolulu area"""
        try:
            # Get current observations from Honolulu airport
            obs_url = f"{self.base_url}/stations/PHNL/observations/latest"
            response = requests.get(obs_url, headers=self.headers, timeout=10)

            if response.status_code == 200:
                data = response.json()
                properties = data.get('properties', {})

                # Extract weather data
                weather_data = {
                    'wind_speed': self._extract_wind_speed(properties),
                    'wave_height': self._estimate_wave_height(properties),
                    'storm_warnings': self._get_storm_warnings(),
                    'temperature': self._extract_temperature(properties),
                    'timestamp': datetime.now().isoformat()
                }

                return weather_data
            else:
                print(f"Weather API error: {response.status_code}")
                return self._get_fallback_weather_data()

        except requests.exceptions.RequestException as e:
            print(f"NOAA API connection failed: {str(e)}")
            return self._get_fallback_weather_data()
        except Exception as e:
            print(f"Error fetching weather data: {str(e)}")
            return self._get_fallback_weather_data()

    def _extract_wind_speed(self, properties: Dict) -> float:
        """Extract wind speed from NOAA data"""
        try:
            wind_speed_ms = properties.get('windSpeed', {}).get('value')
            if wind_speed_ms is not None:
                # Convert m/s to knots (1 m/s = 1.944 knots)
                return round(wind_speed_ms * 1.944, 1)
            return 10.0  # Default fallback
        except:
            return 10.0

    def _extract_temperature(self, properties: Dict) -> float:
        """Extract temperature from NOAA data"""
        try:
            temp_celsius = properties.get('temperature', {}).get('value')
            if temp_celsius is not None:
                return round(temp_celsius, 1)
            return 26.5  # Default fallback for Hawaii
        except:
            return 26.5

    def _estimate_wave_height(self, properties: Dict) -> float:
        """Estimate wave height based on wind conditions"""
        try:
            wind_speed_ms = properties.get('windSpeed', {}).get('value', 5)
            wind_speed_knots = wind_speed_ms * 1.944

            # Rough estimation: wave height correlates with wind speed
            # This is a simplified model - real implementation would use marine forecasts
            if wind_speed_knots < 10:
                return 2.0
            elif wind_speed_knots < 15:
                return 3.5
            elif wind_speed_knots < 20:
                return 5.0
            elif wind_speed_knots < 25:
                return 7.0
            else:
                return 10.0

        except:
            return 3.0  # Default moderate wave height

    def _get_storm_warnings(self) -> List[str]:
        """Get active storm warnings for Hawaii"""
        try:
            # Get active alerts for Hawaii
            alerts_url = f"{self.base_url}/alerts/active?area=HI"
            response = requests.get(alerts_url, headers=self.headers, timeout=10)

            if response.status_code == 200:
                data = response.json()
                warnings = []

                for alert in data.get('features', []):
                    properties = alert.get('properties', {})
                    event = properties.get('event', '')

                    # Filter for storm-related warnings
                    storm_events = [
                        'Hurricane Warning', 'Hurricane Watch', 'Tropical Storm Warning',
                        'Tropical Storm Watch', 'High Wind Warning', 'Gale Warning',
                        'Storm Warning', 'Small Craft Advisory'
                    ]

                    if any(storm_event.lower() in event.lower() for storm_event in storm_events):
                        warnings.append(event)

                return warnings[:5]  # Limit to 5 most recent warnings

        except requests.exceptions.RequestException as e:
            print(f"Error fetching storm warnings: {str(e)}")
        except Exception as e:
            print(f"Error fetching storm warnings: {str(e)}")

        return []

    def get_forecast(self, days_ahead: int = 3) -> List[Dict]:
        """Get weather forecast for specified days ahead"""
        try:
            # Get forecast for Honolulu grid
            forecast_url = f"{self.base_url}/gridpoints/{self.honolulu_grid['office']}/{self.honolulu_grid['gridX']},{self.honolulu_grid['gridY']}/forecast"
            response = requests.get(forecast_url, headers=self.headers, timeout=10)

            if response.status_code == 200:
                data = response.json()
                periods = data.get('properties', {}).get('periods', [])

                forecasts = []
                for i, period in enumerate(periods[:days_ahead * 2]):  # 2 periods per day (day/night)
                    if i % 2 == 0:  # Only take day forecasts
                        forecast = {
                            'date': (datetime.now() + timedelta(days=i//2)).strftime('%Y-%m-%d'),
                            'temperature': self._extract_forecast_temp(period),
                            'wind_speed': self._extract_forecast_wind(period),
                            'conditions': period.get('shortForecast', 'Unknown'),
                            'storm_probability': self._estimate_storm_probability(period)
                        }
                        forecasts.append(forecast)

                return forecasts

        except requests.exceptions.RequestException as e:
            print(f"Error fetching forecast: {str(e)}")
            return self._get_fallback_forecast(days_ahead)
        except Exception as e:
            print(f"Error fetching forecast: {str(e)}")
            return self._get_fallback_forecast(days_ahead)

    def _extract_forecast_temp(self, period: Dict) -> float:
        """Extract temperature from forecast period"""
        try:
            temp_f = period.get('temperature')
            if temp_f is not None:
                # Convert Fahrenheit to Celsius
                return round((temp_f - 32) * 5/9, 1)
            return 26.5
        except:
            return 26.5

    def _extract_forecast_wind(self, period: Dict) -> float:
        """Extract wind speed from forecast period"""
        try:
            wind_desc = period.get('windSpeed', '10 mph')
            # Parse wind speed from description like "10 to 15 mph"
            numbers = re.findall(r'\d+', wind_desc)
            if numbers:
                mph = int(numbers[0])
                # Convert mph to knots (1 mph = 0.868976 knots)
                return round(mph * 0.868976, 1)
            return 10.0
        except:
            return 10.0

    def _estimate_storm_probability(self, period: Dict) -> float:
        """Estimate storm probability from forecast"""
        try:
            forecast = period.get('shortForecast', '').lower()
            detailed = period.get('detailedForecast', '').lower()

            storm_keywords = ['storm', 'thunder', 'severe', 'heavy rain', 'high wind']

            combined_text = f"{forecast} {detailed}"
            storm_indicators = sum(1 for keyword in storm_keywords if keyword in combined_text)

            return min(storm_indicators * 0.3, 1.0)  # Cap at 100%

        except:
            return 0.1

    def _get_fallback_weather_data(self) -> Dict:
        """Return error state when NOAA API is unavailable"""
        print("ERROR: NOAA Weather API connection failed. Real weather data required for predictions.")
        return {
            'wind_speed': None,
            'wave_height': None,
            'storm_warnings': ['API_ERROR: NOAA Weather API unavailable'],
            'temperature': None,
            'timestamp': datetime.now().isoformat(),
            'error': 'NOAA Weather API connection failed - predictions unavailable'
        }

    def _get_fallback_forecast(self, days_ahead: int) -> List[Dict]:
        """Return error state when forecast API fails"""
        print("ERROR: NOAA Forecast API unavailable")
        forecasts = []
        for i in range(days_ahead):
            forecast = {
                'date': (datetime.now() + timedelta(days=i)).strftime('%Y-%m-%d'),
                'temperature': None,
                'wind_speed': None,
                'conditions': 'API_ERROR',
                'storm_probability': None,
                'error': 'NOAA Forecast API unavailable'
            }
            forecasts.append(forecast)
        return forecasts

    def update_weather_data(self):
        """Update weather data and store in database"""
        try:
            from data_manager import DataManager

            current_weather = self.get_current_weather()
            forecast = self.get_forecast(3)

            # Store current weather
            data_manager = DataManager()
            weather_data = {
                'date': datetime.now().strftime('%Y-%m-%d'),
                'wind_speed': current_weather.get('wind_speed'),
                'wave_height': current_weather.get('wave_height'),
                'storm_warnings': current_weather.get('storm_warnings', []),
                'temperature': current_weather.get('temperature')
            }

            data_manager.store_weather_data(weather_data)

            # Store forecast data
            for forecast_day in forecast:
                # Provide fallback wave height if wind speed is None from forecast
                wind_speed_for_wave = forecast_day.get('wind_speed')
                if wind_speed_for_wave is None:
                    wind_speed_for_wave = 10.0 # Default to a moderate wind speed

                forecast_data = {
                    'date': forecast_day['date'],
                    'wind_speed': forecast_day.get('wind_speed'),
                    'wave_height': self._estimate_wave_height({'windSpeed': {'value': wind_speed_for_wave / 1.944}}), # Pass value in m/s
                    'storm_warnings': [] if forecast_day.get('storm_probability', 0) < 0.5 else ['Storm Risk'],
                    'temperature': forecast_day.get('temperature')
                }
                data_manager.store_weather_data(forecast_data)

            print("Weather data updated successfully")

        except ImportError:
            print("Error: DataManager not found. Cannot update weather data.")
        except Exception as e:
            print(f"Error updating weather data: {str(e)}")

    def is_fishing_disrupted(self, weather_data: Dict) -> bool:
        """Determine if fishing would be disrupted by weather conditions"""
        wind_speed = weather_data.get('wind_speed', 0)
        wave_height = weather_data.get('wave_height', 0)
        storm_warnings = weather_data.get('storm_warnings', [])

        # Fishing disruption criteria
        if storm_warnings and 'API_ERROR' not in storm_warnings[0]: # Check if it's a real warning
            return True
        if wind_speed is not None and wind_speed > 20:  # Knots
            return True
        if wave_height is not None and wave_height > 8:  # Feet
            return True

        return False

    def get_fishing_impact_score(self, weather_data: Dict) -> float:
        """Calculate fishing impact score (0-1, where 1 is maximum disruption)"""
        wind_speed = weather_data.get('wind_speed', 0)
        wave_height = weather_data.get('wave_height', 0)
        storm_warnings = weather_data.get('storm_warnings', [])

        # Handle cases where weather data might be None due to API errors
        if wind_speed is None or wave_height is None:
            return 1.0 # Assume maximum disruption if data is missing

        # Base impact from storm warnings
        storm_impact = 1.0 if storm_warnings and 'API_ERROR' not in storm_warnings[0] else 0.0

        # Wind impact (normalized)
        wind_impact = min(max((wind_speed - 10) / 20, 0), 1)  # 0 at 10 knots, 1 at 30 knots

        # Wave impact (normalized)
        wave_impact = min(max((wave_height - 2) / 8, 0), 1)  # 0 at 2 feet, 1 at 10 feet

        # Combined impact (weighted average)
        total_impact = (storm_impact * 0.5 + wind_impact * 0.3 + wave_impact * 0.2)

        return min(total_impact, 1.0)