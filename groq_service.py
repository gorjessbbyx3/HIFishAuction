import os
import json
from datetime import datetime
from typing import Dict, List, Optional
import requests

class GroqService:
    """Service for AI-powered market analysis using Groq"""
    
    def __init__(self):
        self.api_key = os.getenv('GROQ_API_KEY')
        self.base_url = "https://api.groq.com/openai/v1/chat/completions"
        self.model = "llama3-8b-8192"
        
        if not self.api_key:
            print("Warning: GROQ_API_KEY not found in environment variables")
    
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
            # Prepare analysis prompt
            prompt = self._create_analysis_prompt(weather_data, ocean_data, species, price_trend)
            
            headers = {
                'Authorization': f'Bearer {self.api_key}',
                'Content-Type': 'application/json'
            }
            
            payload = {
                'model': self.model,
                'messages': [
                    {
                        'role': 'system',
                        'content': 'You are an expert fish market analyst specializing in Hawaii fish auctions. Provide concise, actionable insights about fish price trends based on environmental conditions.'
                    },
                    {
                        'role': 'user',
                        'content': prompt
                    }
                ],
                'max_tokens': 500,
                'temperature': 0.3
            }
            
            response = requests.post(self.base_url, headers=headers, json=payload, timeout=30)
            
            if response.status_code == 200:
                result = response.json()
                analysis_text = result['choices'][0]['message']['content']
                return self._parse_analysis_response(analysis_text)
            else:
                print(f"Groq API error: {response.status_code}")
                return self._get_fallback_analysis()
                
        except Exception as e:
            print(f"Error calling Groq API: {str(e)}")
            return self._get_fallback_analysis()
    
    def _create_analysis_prompt(self, weather_data: Dict, ocean_data: Dict, 
                              species: str, price_trend: str) -> str:
        """Create analysis prompt for Groq AI"""
        
        # Handle API error cases
        if weather_data.get('error') or ocean_data.get('error'):
            return f"""
            Environmental data is currently unavailable due to API connection issues.
            Without real-time weather and ocean conditions, accurate fish auction price predictions cannot be made.
            
            Please analyze the impact of missing environmental data on {species.replace('_', ' ')} price predictions
            and recommend next steps for obtaining reliable data sources.
            """
        
        prompt = f"""
        Analyze the Hawaii fish auction market conditions for {species.replace('_', ' ')} with current price trend: {price_trend}
        
        WEATHER CONDITIONS:
        - Wind Speed: {weather_data.get('wind_speed', 'N/A')} knots
        - Wave Height: {weather_data.get('wave_height', 'N/A')} feet
        - Temperature: {weather_data.get('temperature', 'N/A')}°C
        - Storm Warnings: {len(weather_data.get('storm_warnings', []))} active
        
        OCEAN CONDITIONS:
        - Sea Surface Temperature: {ocean_data.get('sea_surface_temp', 'N/A')}°C
        - Chlorophyll Level: {ocean_data.get('chlorophyll', 'N/A')} mg/m³
        - Current Strength: {ocean_data.get('current_strength', 'N/A')}
        - Upwelling Index: {ocean_data.get('upwelling_index', 'N/A')}
        
        Provide a brief analysis (2-3 sentences) focusing on:
        1. How these conditions affect fishing operations
        2. Expected impact on fish supply and auction prices
        3. Confidence level in the prediction (0-100%)
        4. One key recommendation for buyers
        
        Format as JSON with fields: analysis, confidence, key_insights (array), recommendations (array)
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
    
    def _get_fallback_analysis(self) -> Dict:
        """Return fallback when Groq API fails"""
        return {
            'analysis': 'AI market analysis unavailable - Groq API connection failed',
            'confidence': 0.0,
            'key_insights': ['Groq API connection required for AI insights'],
            'recommendations': ['Check Groq API connectivity and credentials']
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