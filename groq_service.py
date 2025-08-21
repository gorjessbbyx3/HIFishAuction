import os
import json
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
import requests
import numpy as np
import pandas as pd

class GroqService:
    """Enhanced AI service for sophisticated fish market analysis using Groq"""
    
    def __init__(self):
        self.api_key = os.getenv('GROQ_API_KEY')
        self.base_url = "https://api.groq.com/openai/v1/chat/completions"
        self.model = "llama3-70b-8192"  # Upgraded to larger model for better analysis
        
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
                    "behavior": "Deep water species, night feeding, follows thermoclines",
                    "seasonal_patterns": "Counter-seasonal to yellowfin, winter abundance",
                    "market_factors": "High-fat content, premium pricing, limited supply",
                    "fishing_methods": "Deep-set longline, nighttime fishing",
                    "environmental_sensitivity": "Medium - depth refugia from surface conditions"
                },
                "mahi_mahi": {
                    "optimal_sst_range": [26, 30],
                    "behavior": "Surface feeder, follows floating debris and FADs",
                    "seasonal_patterns": "Spring/fall peaks, follows warm water masses",
                    "market_factors": "Restaurant favorite, consistent demand, moderate pricing",
                    "fishing_methods": "Trolling, FAD fishing, surface gear",
                    "environmental_sensitivity": "Very high - surface conditions critical"
                }
            },
            "market_dynamics": {
                "demand_factors": [
                    "Restaurant reopenings and tourism levels",
                    "Japanese sashimi market demand",
                    "Local poke shop consumption",
                    "Mainland US restaurant chains",
                    "Seasonal holiday demand spikes"
                ],
                "supply_factors": [
                    "Fleet size and vessel availability",
                    "Fuel costs and operational expenses",
                    "Weather-related fishing disruptions",
                    "Regulatory restrictions and quotas",
                    "Fish aggregating device (FAD) effectiveness"
                ],
                "price_elasticity": {
                    "high_grade_tuna": "Low elasticity - premium market",
                    "mid_grade_fish": "Medium elasticity - restaurant market",
                    "local_consumption": "High elasticity - price sensitive"
                }
            },
            "environmental_correlations": {
                "el_nino_effects": "Reduced yellowfin, increased mahi-mahi in eastern areas",
                "la_nina_effects": "Enhanced yellowfin abundance, cooler water species",
                "trade_wind_patterns": "Affect upwelling and baitfish distribution",
                "storm_impacts": "3-7 day supply disruption, 15-30% price increases",
                "moon_phases": "New moon periods show higher catch rates for tunas"
            }
        }
    
    def analyze_market_conditions(self, weather_data: Dict, ocean_data: Dict, 
                                species: str, price_trend: str) -> Dict:
        """Generate AI-powered market analysis"""
        if not self.api_key:
            return {
                'analysis': 'Groq AI analysis unavailable - API key required',
                'confidence': 0.0,
                'key_insights': ['API key required for AI analysis'],
                'recommendations': ['Provide GROQ_API_KEY for market insights']
            }
        
        try:
            # Enhanced analysis with knowledge base integration
            enhanced_analysis = self._perform_enhanced_analysis(weather_data, ocean_data, species, price_trend)
            
            # Prepare sophisticated analysis prompt
            prompt = self._create_enhanced_analysis_prompt(
                weather_data, ocean_data, species, price_trend, enhanced_analysis
            )
            
            headers = {
                'Authorization': f'Bearer {self.api_key}',
                'Content-Type': 'application/json'
            }
            
            payload = {
                'model': self.model,
                'messages': [
                    {
                        'role': 'system',
                        'content': self._get_expert_system_prompt()
                    },
                    {
                        'role': 'user',
                        'content': prompt
                    }
                ],
                'max_tokens': 800,
                'temperature': 0.2
            }
            
            response = requests.post(self.base_url, headers=headers, json=payload, timeout=30)
            
            if response.status_code == 200:
                result = response.json()
                analysis_text = result['choices'][0]['message']['content']
                parsed_analysis = self._parse_analysis_response(analysis_text)
                
                # Enhance with quantitative analytics
                parsed_analysis.update(enhanced_analysis)
                
                # Cache the analysis
                cache_key = f"{species}_{datetime.now().strftime('%Y%m%d_%H')}"
                self.analysis_cache[cache_key] = parsed_analysis
                
                return parsed_analysis
            else:
                print(f"Groq API error: {response.status_code}")
                return self._get_fallback_analysis()
                
        except Exception as e:
            print(f"Error calling Groq API: {str(e)}")
            return self._get_fallback_analysis()
    
    def _get_expert_system_prompt(self) -> str:
        """Enhanced system prompt with deep market expertise"""
        return """You are Dr. Marina Ichthys, a world-renowned expert in Pacific pelagic fisheries economics with 25 years of experience analyzing Hawaii's fish auction markets. You have:

- PhD in Marine Resource Economics from University of Hawaii
- Published 50+ papers on Pacific tuna market dynamics
- Former advisor to NOAA Pacific Islands Fisheries Science Center
- Current consultant to major Hawaii fishing operations
- Deep expertise in environmental-economic modeling of fish markets

Your analysis integrates:
1. Oceanographic conditions and fish behavior patterns
2. Market microstructure and auction dynamics
3. Supply chain logistics and quality factors
4. International trade flows and demand patterns
5. Environmental regulatory impacts
6. Seasonal migration and recruitment patterns

Provide sophisticated, quantitative analysis with specific actionable recommendations for fish buyers at Hawaii auctions. Focus on risk-adjusted profit optimization and strategic timing decisions."""

    def _perform_enhanced_analysis(self, weather_data: Dict, ocean_data: Dict, 
                                 species: str, price_trend: str) -> Dict:
        """Perform quantitative analysis using knowledge base"""
        species_key = species.lower().replace(' ', '_').replace('(', '').replace(')', '')
        species_profile = self.fish_market_knowledge["species_profiles"].get(species_key, {})
        
        # Environmental suitability scoring
        environmental_score = self._calculate_environmental_suitability(
            weather_data, ocean_data, species_profile
        )
        
        # Market pressure analysis
        market_pressure = self._analyze_market_pressure(weather_data, ocean_data, species_key)
        
        # Risk assessment
        risk_factors = self._assess_risk_factors(weather_data, ocean_data, species_profile)
        
        # Price volatility prediction
        volatility_forecast = self._forecast_price_volatility(
            environmental_score, market_pressure, species_key
        )
        
        return {
            'environmental_suitability': environmental_score,
            'market_pressure_index': market_pressure,
            'risk_assessment': risk_factors,
            'volatility_forecast': volatility_forecast,
            'species_expertise': species_profile
        }
    
    def _calculate_environmental_suitability(self, weather_data: Dict, 
                                           ocean_data: Dict, species_profile: Dict) -> Dict:
        """Calculate environmental suitability for target species"""
        sst = ocean_data.get('sea_surface_temp')
        wind_speed = weather_data.get('wind_speed')
        storm_warnings = weather_data.get('storm_warnings', [])
        chlorophyll = ocean_data.get('chlorophyll')
        
        if not all([sst, wind_speed, chlorophyll]):
            return {'score': 0.0, 'factors': ['Environmental data unavailable']}
        
        optimal_range = species_profile.get('optimal_sst_range', [24, 28])
        sensitivity = species_profile.get('environmental_sensitivity', 'Medium')
        
        # Temperature suitability (0-1 scale)
        temp_optimal = (optimal_range[0] + optimal_range[1]) / 2
        temp_range = optimal_range[1] - optimal_range[0]
        temp_score = max(0, 1 - abs(sst - temp_optimal) / (temp_range * 1.5))
        
        # Fishing condition suitability
        fishing_score = 1.0
        if storm_warnings:
            fishing_score *= 0.1  # Severe penalty for storms
        elif wind_speed > 25:
            fishing_score *= 0.3  # High winds
        elif wind_speed > 20:
            fishing_score *= 0.6  # Moderate winds
        
        # Food availability (chlorophyll proxy)
        food_score = min(1.0, chlorophyll / 0.2)  # Normalize around 0.2 mg/m³
        
        # Weighted composite score
        if sensitivity == 'Very high':
            weights = [0.5, 0.3, 0.2]  # Temp, fishing, food
        elif sensitivity == 'High':
            weights = [0.4, 0.35, 0.25]
        elif sensitivity == 'Medium':
            weights = [0.3, 0.4, 0.3]
        else:
            weights = [0.25, 0.5, 0.25]  # Low sensitivity
        
        composite_score = (temp_score * weights[0] + 
                          fishing_score * weights[1] + 
                          food_score * weights[2])
        
        return {
            'score': round(composite_score, 3),
            'temperature_suitability': round(temp_score, 3),
            'fishing_conditions': round(fishing_score, 3),
            'food_availability': round(food_score, 3),
            'optimal_temp_range': optimal_range,
            'current_temp': sst
        }
    
    def _analyze_market_pressure(self, weather_data: Dict, ocean_data: Dict, species: str) -> Dict:
        """Analyze supply-demand pressure indicators"""
        storm_warnings = weather_data.get('storm_warnings', [])
        wind_speed = weather_data.get('wind_speed', 0)
        
        # Supply disruption index
        supply_disruption = 0.0
        if storm_warnings:
            supply_disruption += len(storm_warnings) * 0.3
        if wind_speed > 20:
            supply_disruption += (wind_speed - 20) * 0.02
        
        supply_disruption = min(1.0, supply_disruption)
        
        # Demand seasonality (simplified - would use real seasonal data)
        current_month = datetime.now().month
        seasonal_demand = {
            'yellowfin_tuna': [0.8, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.2, 1.1, 1.0, 0.9, 0.8],
            'bigeye_tuna': [1.2, 1.1, 1.0, 0.9, 0.8, 0.8, 0.9, 1.0, 1.1, 1.2, 1.2, 1.1],
            'mahi_mahi': [0.9, 0.9, 1.1, 1.2, 1.1, 0.8, 0.8, 0.9, 1.1, 1.2, 1.0, 0.9]
        }
        
        demand_multiplier = seasonal_demand.get(species, [1.0] * 12)[current_month - 1]
        
        # Market pressure index
        pressure_index = (supply_disruption * 0.6 + (demand_multiplier - 1.0) * 0.4)
        
        return {
            'pressure_index': round(pressure_index, 3),
            'supply_disruption': round(supply_disruption, 3),
            'seasonal_demand_factor': round(demand_multiplier, 3),
            'interpretation': self._interpret_market_pressure(pressure_index)
        }
    
    def _assess_risk_factors(self, weather_data: Dict, ocean_data: Dict, species_profile: Dict) -> Dict:
        """Comprehensive risk assessment"""
        risks = []
        risk_score = 0.0
        
        # Weather risks
        storm_warnings = weather_data.get('storm_warnings', [])
        wind_speed = weather_data.get('wind_speed', 0)
        
        if storm_warnings:
            risks.append({
                'type': 'Weather',
                'severity': 'High',
                'description': f"{len(storm_warnings)} active storm warning(s)",
                'impact': 'Supply disruption 3-7 days'
            })
            risk_score += 0.4
        elif wind_speed > 25:
            risks.append({
                'type': 'Weather',
                'severity': 'Medium',
                'description': f"High winds ({wind_speed} knots)",
                'impact': 'Reduced fishing activity'
            })
            risk_score += 0.2
        
        # Environmental risks
        sst = ocean_data.get('sea_surface_temp')
        optimal_range = species_profile.get('optimal_sst_range', [24, 28])
        
        if sst and (sst < optimal_range[0] - 2 or sst > optimal_range[1] + 2):
            risks.append({
                'type': 'Environmental',
                'severity': 'Medium',
                'description': f"Suboptimal water temperature ({sst}°C)",
                'impact': 'Reduced catch efficiency'
            })
            risk_score += 0.15
        
        # Market risks (would integrate real market data)
        current_month = datetime.now().month
        if current_month in [12, 1, 2]:  # Winter months typically higher volatility
            risks.append({
                'type': 'Market',
                'severity': 'Low',
                'description': "Seasonal market volatility period",
                'impact': 'Increased price uncertainty'
            })
            risk_score += 0.1
        
        return {
            'overall_risk_score': min(1.0, risk_score),
            'risk_level': self._categorize_risk_level(min(1.0, risk_score)),
            'risk_factors': risks,
            'mitigation_strategies': self._suggest_risk_mitigation(risks)
        }
    
    def _forecast_price_volatility(self, env_score: Dict, market_pressure: Dict, species: str) -> Dict:
        """Forecast price volatility based on multiple factors"""
        base_volatility = {
            'yellowfin_tuna': 0.15,
            'bigeye_tuna': 0.18,
            'mahi_mahi': 0.22
        }.get(species, 0.18)
        
        # Adjust volatility based on conditions
        volatility_multiplier = 1.0
        
        # Environmental instability increases volatility
        if env_score['score'] < 0.3:
            volatility_multiplier *= 1.5
        elif env_score['score'] < 0.6:
            volatility_multiplier *= 1.2
        
        # Market pressure affects volatility
        pressure = market_pressure['pressure_index']
        if abs(pressure) > 0.3:
            volatility_multiplier *= 1.3
        elif abs(pressure) > 0.1:
            volatility_multiplier *= 1.1
        
        forecasted_volatility = base_volatility * volatility_multiplier
        
        return {
            'forecasted_volatility': round(forecasted_volatility, 3),
            'base_volatility': base_volatility,
            'volatility_multiplier': round(volatility_multiplier, 2),
            'volatility_drivers': self._identify_volatility_drivers(env_score, market_pressure)
        }
    
    def _create_enhanced_analysis_prompt(self, weather_data: Dict, ocean_data: Dict, 
                                       species: str, price_trend: str, enhanced_analysis: Dict) -> str:
        """Create sophisticated analysis prompt with enhanced context"""
        
        # Handle API error cases
        if weather_data.get('error') or ocean_data.get('error'):
            return f"""
            CRITICAL: Environmental data APIs are currently unavailable. 
            
            Without real-time NOAA weather and ocean conditions, accurate {species.replace('_', ' ')} auction price predictions cannot be generated.
            
            Analyze the market implications of this data unavailability and provide specific recommendations for:
            1. Alternative data sources or manual observation methods
            2. Risk management strategies during data outages
            3. Timeline for restoring predictive capabilities
            
            Focus on actionable steps for fish buyers during this data limitation period.
            """
        
        env_analysis = enhanced_analysis.get('environmental_suitability', {})
        market_analysis = enhanced_analysis.get('market_pressure_index', {})
        risk_analysis = enhanced_analysis.get('risk_assessment', {})
        volatility_analysis = enhanced_analysis.get('volatility_forecast', {})
        species_expertise = enhanced_analysis.get('species_expertise', {})
        
        prompt = f"""
        HAWAII FISH AUCTION MARKET ANALYSIS - {species.replace('_', ' ').title()}
        Current Price Trend: {price_trend}
        
        ENVIRONMENTAL CONDITIONS & ANALYSIS:
        - Sea Surface Temperature: {ocean_data.get('sea_surface_temp', 'N/A')}°C
        - Environmental Suitability Score: {env_analysis.get('score', 'N/A')}/1.0
        - Optimal Temperature Range: {env_analysis.get('optimal_temp_range', 'N/A')}°C
        - Food Availability (Chlorophyll): {ocean_data.get('chlorophyll', 'N/A')} mg/m³
        - Current Strength: {ocean_data.get('current_strength', 'N/A')}
        
        FISHING CONDITIONS:
        - Wind Speed: {weather_data.get('wind_speed', 'N/A')} knots
        - Wave Height: {weather_data.get('wave_height', 'N/A')} feet
        - Storm Warnings: {len(weather_data.get('storm_warnings', []))} active
        - Fishing Condition Score: {env_analysis.get('fishing_conditions', 'N/A')}/1.0
        
        MARKET DYNAMICS:
        - Market Pressure Index: {market_analysis.get('pressure_index', 'N/A')}
        - Supply Disruption Factor: {market_analysis.get('supply_disruption', 'N/A')}
        - Seasonal Demand Factor: {market_analysis.get('seasonal_demand_factor', 'N/A')}
        - Forecasted Volatility: {volatility_analysis.get('forecasted_volatility', 'N/A')}
        
        RISK ASSESSMENT:
        - Overall Risk Score: {risk_analysis.get('overall_risk_score', 'N/A')}/1.0
        - Risk Level: {risk_analysis.get('risk_level', 'N/A')}
        - Key Risk Factors: {len(risk_analysis.get('risk_factors', []))} identified
        
        SPECIES-SPECIFIC INTELLIGENCE:
        - Behavioral Patterns: {species_expertise.get('behavior', 'Standard pelagic patterns')}
        - Environmental Sensitivity: {species_expertise.get('environmental_sensitivity', 'Medium')}
        - Market Positioning: {species_expertise.get('market_factors', 'Standard commodity')}
        
        ANALYTICAL REQUIREMENTS:
        Based on this comprehensive data, provide a sophisticated market analysis including:
        
        1. MARKET OUTLOOK (200 words):
        - Integrate environmental suitability with market pressure dynamics
        - Quantify expected price movement with confidence intervals
        - Explain the primary drivers using oceanographic and economic principles
        
        2. STRATEGIC RECOMMENDATIONS (150 words):
        - Specific buying/selling timing recommendations
        - Risk-adjusted position sizing guidance
        - Hedging strategies for identified risk factors
        
        3. QUANTITATIVE INSIGHTS (100 words):
        - Price volatility expectations with ranges
        - Probability assessments for different scenarios
        - ROI optimization strategies
        
        4. MONITORING PRIORITIES (50 words):
        - Key indicators to track for early signal detection
        - Critical thresholds for strategy adjustments
        
        Format as detailed JSON with: market_outlook, strategic_recommendations, quantitative_insights, monitoring_priorities, confidence_level (0-100), risk_adjusted_rating (A-F)
        """
        
        return prompt
    
    def _parse_analysis_response(self, analysis_text: str) -> Dict:
        """Parse Groq AI response into structured format"""
        try:
            # Try to extract JSON from response
            if '{' in analysis_text and '}' in analysis_text:
                start = analysis_text.find('{')
                end = analysis_text.rfind('}') + 1
                json_str = analysis_text[start:end]
                parsed = json.loads(json_str)
                
                return {
                    'analysis': parsed.get('analysis', analysis_text[:200]),
                    'confidence': float(parsed.get('confidence', 50.0)) / 100.0,
                    'key_insights': parsed.get('key_insights', [analysis_text[:100]]),
                    'recommendations': parsed.get('recommendations', ['Monitor conditions closely'])
                }
            else:
                # Fallback to plain text parsing
                return {
                    'analysis': analysis_text[:200],
                    'confidence': 0.5,
                    'key_insights': [analysis_text[:100]],
                    'recommendations': ['Monitor market conditions']
                }
                
        except json.JSONDecodeError:
            return {
                'analysis': analysis_text[:200],
                'confidence': 0.5,
                'key_insights': ['AI analysis provided'],
                'recommendations': ['Review market conditions']
            }
    
    def _interpret_market_pressure(self, pressure_index: float) -> str:
        """Interpret market pressure index"""
        if pressure_index > 0.3:
            return "High bullish pressure - supply constraints likely"
        elif pressure_index > 0.1:
            return "Moderate bullish pressure - slight price support"
        elif pressure_index < -0.3:
            return "High bearish pressure - oversupply conditions"
        elif pressure_index < -0.1:
            return "Moderate bearish pressure - price weakness expected"
        else:
            return "Neutral market pressure - balanced supply/demand"
    
    def _categorize_risk_level(self, risk_score: float) -> str:
        """Categorize overall risk level"""
        if risk_score >= 0.7:
            return "High"
        elif risk_score >= 0.4:
            return "Medium"
        elif risk_score >= 0.2:
            return "Low"
        else:
            return "Minimal"
    
    def _suggest_risk_mitigation(self, risks: List[Dict]) -> List[str]:
        """Suggest risk mitigation strategies"""
        strategies = []
        
        for risk in risks:
            if risk['type'] == 'Weather':
                if risk['severity'] == 'High':
                    strategies.append("Consider pre-purchasing before storm impact")
                    strategies.append("Diversify supplier base across multiple fishing areas")
                else:
                    strategies.append("Monitor weather updates for timing adjustments")
            elif risk['type'] == 'Environmental':
                strategies.append("Track fish aggregating device (FAD) reports")
                strategies.append("Consider alternative species with better environmental match")
            elif risk['type'] == 'Market':
                strategies.append("Implement dynamic pricing strategies")
                strategies.append("Maintain flexible inventory levels")
        
        return list(set(strategies))  # Remove duplicates
    
    def _identify_volatility_drivers(self, env_score: Dict, market_pressure: Dict) -> List[str]:
        """Identify primary drivers of price volatility"""
        drivers = []
        
        if env_score['score'] < 0.5:
            drivers.append("Environmental instability")
        if env_score.get('fishing_conditions', 1.0) < 0.7:
            drivers.append("Poor fishing conditions")
        if abs(market_pressure['pressure_index']) > 0.2:
            drivers.append("Market supply-demand imbalance")
        if market_pressure['supply_disruption'] > 0.3:
            drivers.append("Supply chain disruptions")
        
        return drivers if drivers else ["Normal market fluctuations"]
    
    def _get_fallback_analysis(self) -> Dict:
        """Return sophisticated fallback when Groq API fails"""
        return {
            'analysis': 'Advanced AI market analysis temporarily unavailable - Groq API connection required',
            'confidence': 0.0,
            'key_insights': [
                'Enhanced market analytics require Groq API connectivity',
                'Sophisticated risk modeling unavailable in offline mode',
                'Quantitative analysis capabilities limited without AI'
            ],
            'recommendations': [
                'Restore Groq API connection for full analytical capabilities',
                'Implement manual market monitoring protocols',
                'Use simplified decision frameworks until AI restoration'
            ],
            'market_outlook': 'Comprehensive market analysis unavailable',
            'strategic_recommendations': 'AI-powered strategies require API connectivity',
            'risk_adjusted_rating': 'N/A'
        }
    
    def generate_investment_insights(self, predictions: List[Dict], 
                                   investment_amount: float) -> Dict:
        """Generate investment strategy insights using AI"""
        if not self.api_key:
            return {
                'strategy': 'Investment analysis unavailable - Groq API key required',
                'risk_assessment': 'high',
                'potential_savings': 0.0,
                'recommendations': ['Provide GROQ_API_KEY for investment insights']
            }
        
        try:
            # Create investment analysis prompt
            prompt = f"""
            Based on the following fish auction price predictions, analyze investment opportunities:
            
            Investment Amount: ${investment_amount:,.2f}
            Predictions: {json.dumps(predictions, indent=2)}
            
            Provide investment strategy recommendations focusing on:
            1. Risk level (low/medium/high)
            2. Optimal timing for prepayments
            3. Expected return/savings potential
            4. Key risk factors to monitor
            
            Format as JSON with fields: strategy, risk_assessment, potential_savings, recommendations
            """
            
            headers = {
                'Authorization': f'Bearer {self.api_key}',
                'Content-Type': 'application/json'
            }
            
            payload = {
                'model': self.model,
                'messages': [
                    {
                        'role': 'system',
                        'content': 'You are a financial analyst specializing in commodity trading and fish market investments.'
                    },
                    {
                        'role': 'user',
                        'content': prompt
                    }
                ],
                'max_tokens': 400,
                'temperature': 0.2
            }
            
            response = requests.post(self.base_url, headers=headers, json=payload, timeout=30)
            
            if response.status_code == 200:
                result = response.json()
                analysis_text = result['choices'][0]['message']['content']
                return self._parse_investment_response(analysis_text, investment_amount)
            else:
                return self._get_fallback_investment_analysis()
                
        except Exception as e:
            print(f"Error generating investment insights: {str(e)}")
            return self._get_fallback_investment_analysis()
    
    def _parse_investment_response(self, analysis_text: str, investment_amount: float) -> Dict:
        """Parse investment analysis response"""
        try:
            if '{' in analysis_text and '}' in analysis_text:
                start = analysis_text.find('{')
                end = analysis_text.rfind('}') + 1
                json_str = analysis_text[start:end]
                parsed = json.loads(json_str)
                
                return {
                    'strategy': parsed.get('strategy', analysis_text[:150]),
                    'risk_assessment': parsed.get('risk_assessment', 'medium'),
                    'potential_savings': float(parsed.get('potential_savings', investment_amount * 0.05)),
                    'recommendations': parsed.get('recommendations', ['Monitor market closely'])
                }
            else:
                return {
                    'strategy': analysis_text[:150],
                    'risk_assessment': 'medium',
                    'potential_savings': investment_amount * 0.05,
                    'recommendations': ['Review investment strategy']
                }
                
        except (json.JSONDecodeError, ValueError):
            return {
                'strategy': analysis_text[:150],
                'risk_assessment': 'medium',
                'potential_savings': investment_amount * 0.03,
                'recommendations': ['Consult investment analysis']
            }
    
    def _get_fallback_investment_analysis(self) -> Dict:
        """Return fallback investment analysis"""
        return {
            'strategy': 'Investment analysis unavailable - Groq API connection failed',
            'risk_assessment': 'high',
            'potential_savings': 0.0,
            'recommendations': ['Check Groq API connectivity for investment insights']
        }