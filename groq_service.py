import os
import requests
import json
from datetime import datetime
from typing import Dict, List, Optional

class GroqService:
    """Enhanced AI service for sophisticated fish market analysis using Groq"""

    def __init__(self):
        self.api_key = os.getenv('GROQ_API_KEY')
        self.base_url = "https://api.groq.com/openai/v1/chat/completions"
        self.model = "llama3-70b-8192"

        # Enhanced knowledge base for fish markets
        self.fish_market_knowledge = self._load_enhanced_knowledge_base()
        self.analysis_cache = {}  # Cache for recent analyses

        if not self.api_key:
            print("Warning: GROQ_API_KEY not found in environment variables")

    def _load_enhanced_knowledge_base(self) -> Dict:
        """Load comprehensive fish market knowledge base"""
        return {
            "species_profiles": {
                "yellowfin_tuna": {
                    "optimal_sst_range": [24, 28],
                    "behavior": "Prefers warm water currents, follows baitfish schools",
                    "seasonal_patterns": "Peak summer catches, migrates with water temperature",
                    "market_factors": "Premium sashimi grade, export demand from Japan",
                    "fishing_methods": "Longline, trolling in deep water (100-400m)",
                    "environmental_sensitivity": "High - very responsive to SST changes"
                },
                "bigeye_tuna": {
                    "optimal_sst_range": [18, 24],
                    "behavior": "Deep water species, less surface dependent",
                    "seasonal_patterns": "Counter-seasonal to yellowfin",
                    "market_factors": "Premium sashimi, highest value",
                    "fishing_methods": "Deep longline fishing",
                    "environmental_sensitivity": "Moderate - deeper habitat"
                }
            }
        }

    def analyze_market_conditions(self, weather_data: Dict, ocean_data: Dict, 
                                species: str, trend: str) -> Dict:
        """Analyze market conditions using AI"""
        try:
            if not self.api_key:
                return self._get_fallback_analysis()

            # Create analysis prompt
            prompt = self._create_analysis_prompt(weather_data, ocean_data, species, trend)

            # Make API call
            headers = {
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json"
            }

            data = {
                "model": self.model,
                "messages": [
                    {"role": "system", "content": self._get_expert_system_prompt()},
                    {"role": "user", "content": prompt}
                ],
                "max_tokens": 1000,
                "temperature": 0.3
            }

            response = requests.post(self.base_url, headers=headers, json=data, timeout=30)

            if response.status_code == 200:
                result = response.json()
                analysis_text = result['choices'][0]['message']['content']
                return {"analysis": analysis_text}
            else:
                return self._get_fallback_analysis()

        except Exception as e:
            print(f"Error calling Groq API: {str(e)}")
            return self._get_fallback_analysis()

    def _create_analysis_prompt(self, weather_data: Dict, ocean_data: Dict, 
                              species: str, trend: str) -> str:
        """Create analysis prompt for AI"""
        return f"""
        Analyze the Hawaii fish auction market conditions for {species}:

        Weather: Wind {weather_data.get('wind_speed', 0)} knots, 
                Waves {weather_data.get('wave_height', 0)} ft
        Ocean: SST {ocean_data.get('sea_surface_temp', 0)}°C, 
               Chlorophyll {ocean_data.get('chlorophyll', 0)} mg/m³
        Current trend: {trend}

        Provide a brief market analysis focusing on how these conditions 
        affect {species} availability and pricing.
        """

    def _get_expert_system_prompt(self) -> str:
        """Get expert system prompt"""
        return """You are a Hawaii fish market expert analyzing auction conditions. 
        Focus on how environmental factors affect fish availability and pricing."""

    def _get_fallback_analysis(self) -> Dict:
        """Fallback analysis when API unavailable"""
        return {
            "analysis": "Market analysis unavailable. Using historical patterns for prediction."
        }