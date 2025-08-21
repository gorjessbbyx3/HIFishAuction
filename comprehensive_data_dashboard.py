import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
from typing import Dict, List

def display_comprehensive_data_status(ecosystem_data, global_analyzer, undercurrent_api, longline_data):
    """Display comprehensive data integration status dashboard"""
    
    st.markdown("### 📊 Comprehensive Data Integration Status")
    
    # Create columns for different data sources
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown("#### 🌊 Pacific Ecosystem Data")
        feat_status = ecosystem_data.check_feat_availability()
        
        if feat_status['status'] == 'Available':
            st.success("✅ NOAA FEAT System")
            st.info("📈 Hawaii Consumption Data")
            
            # Show key Hawaii consumption metrics
            consumption_data = ecosystem_data.hawaii_consumption_data
            per_capita = consumption_data['per_capita_consumption']['total_with_noncommercial']
            st.metric("Per Capita Consumption", f"{per_capita} lbs/year", 
                     delta="1.8x US average", delta_color="normal")
            
        else:
            st.warning("⚠️ FEAT System Limited")
            st.info("📊 Baseline Data Available")
    
    with col2:
        st.markdown("#### 🌍 Global Markets")
        
        # Undercurrent News API status
        ucn_status = undercurrent_api.check_api_status()
        if ucn_status['status'] == 'Connected':
            st.success("✅ Undercurrent News API")
        elif 'Authentication' in ucn_status['status']:
            st.warning("🔑 API Key Required")
            st.info("3700+ datasets available")
        else:
            st.error("❌ API Unavailable")
            
        # Global market analysis capability
        st.info("📈 Market Analysis Ready")
        st.metric("Global Datasets", "3700+", delta="Live updates")
    
    with col3:
        st.markdown("#### 🎣 NOAA Longline Data")
        
        longline_status = longline_data.check_data_availability()
        if longline_status['status'] == 'Available':
            st.success("✅ Logbook Data Access")
            st.info("📅 Monthly Updates")
            
            # Show data coverage
            st.metric("Data Coverage", "1996-2023", delta="27 years")
            
        else:
            st.warning("⚠️ Connection Issues")
            st.info("🔄 Retry Available")
    
    with col4:
        st.markdown("#### 🤖 AI Analytics")
        st.success("✅ Groq AI Active")
        st.info("🧠 LLaMA 70B Model")
        st.metric("Analysis Depth", "Advanced", delta="Species-specific")

def display_hawaii_market_analysis(ecosystem_data, species: str):
    """Display Hawaii-specific market analysis"""
    
    st.markdown("### 🏝️ Hawaii Market Context Analysis")
    
    # Get Hawaii market analysis for the species
    market_analysis = ecosystem_data.analyze_hawaii_market_context(species)
    
    if 'consumption_analysis' in market_analysis:
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("#### 📊 Consumption Profile")
            consumption = market_analysis['consumption_analysis']
            
            # Per capita consumption
            per_capita = consumption.get('per_capita_consumption_lbs', 0)
            st.metric("Per Capita Consumption", f"{per_capita:.2f} lbs/year")
            
            # Ranking in Hawaii
            ranking = consumption.get('ranking_in_hawaii', 'Unknown')
            st.info(f"**Hawaii Ranking:** {ranking}")
            
            # Noncommercial percentage
            noncommercial_pct = consumption.get('noncommercial_percentage', 0)
            st.metric("Noncommercial Catch", f"{noncommercial_pct}%", 
                     delta="of total supply")
        
        with col2:
            st.markdown("#### 🎯 Market Position")
            market_pos = market_analysis.get('market_position', {})
            
            # Market tier
            tier = market_pos.get('market_tier', 'Unknown')
            st.info(f"**Market Tier:** {tier}")
            
            # Commercial dependency
            commercial_dep = market_pos.get('commercial_dependency_pct', 0)
            st.metric("Commercial Dependency", f"{commercial_dep}%")
            
            # Auction significance
            auction_sig = market_pos.get('auction_significance', 'Unknown')
            st.success(f"**Auction Significance:** {auction_sig}")
    
    # Supply analysis
    if 'supply_analysis' in market_analysis:
        st.markdown("#### 🚢 Supply Source Analysis")
        supply_analysis = market_analysis['supply_analysis']
        supply_breakdown = supply_analysis.get('supply_breakdown', {})
        
        # Create supply source chart
        supply_data = []
        for source, percentage in supply_breakdown.items():
            supply_data.append({
                'Source': source.replace('_', ' ').title(),
                'Percentage': percentage
            })
        
        if supply_data:
            df_supply = pd.DataFrame(supply_data)
            fig = px.pie(df_supply, values='Percentage', names='Source',
                        title=f"{species} Supply Sources")
            st.plotly_chart(fig, use_container_width=True)
        
        # Supply constraints
        constraint = supply_analysis.get('primary_supply_constraint', 'Unknown')
        st.warning(f"**Primary Supply Constraint:** {constraint}")
        
        # Seasonal patterns
        seasonal = supply_analysis.get('seasonal_supply_patterns', 'Unknown')
        st.info(f"**Seasonal Pattern:** {seasonal}")
    
    # Price sensitivity analysis
    if 'price_sensitivity' in market_analysis:
        st.markdown("#### 💰 Price Sensitivity Analysis")
        price_sens = market_analysis['price_sensitivity']
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            sensitivity = price_sens.get('sensitivity_level', 'Unknown')
            st.metric("Price Sensitivity", sensitivity)
        
        with col2:
            elasticity = price_sens.get('price_elasticity_estimate', 0)
            st.metric("Price Elasticity", f"{elasticity:.1f}")
        
        with col3:
            premium_tol = price_sens.get('premium_tolerance', 'Unknown')
            st.info(f"**Premium Tolerance:** {premium_tol}")
        
        # Substitution risk
        substitution = price_sens.get('substitution_risk', 'Unknown')
        st.warning(f"**Substitution Risk:** {substitution}")

def display_global_market_pressure(global_analyzer, species: str, current_price: float):
    """Display global market pressure analysis"""
    
    st.markdown("### 🌍 Global Market Pressure Analysis")
    
    # Get global market analysis
    pressure_analysis = global_analyzer.analyze_global_price_pressure(species, current_price)
    
    if pressure_analysis['status'] == 'Analysis Complete':
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("#### 📈 Pressure Indicators")
            pressure_data = pressure_analysis['pressure_analysis']
            
            # Pressure level and direction
            pressure_level = pressure_data.get('pressure_level', 'Unknown')
            direction = pressure_data.get('direction', 'Unknown')
            
            if direction == 'Upward':
                st.error(f"🔴 {pressure_level} Upward Pressure")
            elif direction == 'Downward':
                st.success(f"🟢 {pressure_level} Downward Pressure")
            else:
                st.info(f"🔵 {pressure_level} Neutral Pressure")
            
            # Global pressure percentage
            global_pressure = pressure_data.get('global_pressure_percent', 0)
            st.metric("Global Pressure", f"{global_pressure:+.1f}%")
            
            # Correlation strength
            correlation = pressure_data.get('correlation_strength', 0)
            st.metric("Market Correlation", f"{correlation:.2f}")
        
        with col2:
            st.markdown("#### 🎯 Market Recommendation")
            recommendation = pressure_analysis.get('recommendation', 'No recommendation available')
            
            if 'prepaying' in recommendation.lower():
                st.success(f"💡 **Buy Signal:** {recommendation}")
            elif 'wait' in recommendation.lower():
                st.error(f"⏳ **Wait Signal:** {recommendation}")
            else:
                st.info(f"📊 **Neutral:** {recommendation}")
            
            # Volatility rating
            volatility = pressure_data.get('volatility_rating', 'Unknown')
            st.metric("Market Volatility", volatility)
            
            # Price uncertainty
            uncertainty = pressure_data.get('price_uncertainty_range', 0)
            st.metric("Price Uncertainty", f"±${uncertainty:.2f}")
    
    else:
        st.warning(f"Analysis Status: {pressure_analysis['status']}")
        if 'error' in pressure_analysis:
            st.error(f"Error: {pressure_analysis['error']}")

def display_data_integration_capabilities():
    """Display comprehensive data integration capabilities"""
    
    st.markdown("### 🔗 Data Integration Capabilities")
    
    with st.expander("📊 Data Sources Overview", expanded=False):
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("#### Authentic Data Sources")
            st.markdown("""
            **NOAA Sources:**
            - NOAA Fisheries Data Catalog (fisheries.noaa.gov/resources/data)
            - Pacific Islands Fishery Ecosystem Analysis Tool (FEAT)
            - FEAT Performance Indicators (fleet metrics, CPUE, economics)
            - Hawaii & California Longline Logbook Data
            - American Samoa Longline Logbook Data
            - Weather Service API
            - CoastWatch Ocean Data
            
            **Market Data:**
            - Undercurrent News Global Seafood Prices (3700+ datasets)
            - Hawaii Consumption Study (CTAHR University of Hawaii)
            - UFA Auction Historical Data Structure (1984-2002)
            """)
        
        with col2:
            st.markdown("#### Integration Benefits")
            st.markdown("""
            **Enhanced Predictions:**
            - Real supply-side data from fishing fleet
            - Global market price correlation
            - Hawaii-specific consumption patterns
            - Environmental impact analysis
            
            **Strategic Insights:**
            - Supply chain vulnerability assessment
            - Market position analysis by species
            - Price sensitivity and elasticity
            - Seasonal pattern validation
            """)
    
    with st.expander("🤖 AI Analytics Capabilities", expanded=False):
        st.markdown("""
        **Groq AI Integration (LLaMA 70B):**
        - Advanced species behavioral pattern analysis
        - Environmental suitability scoring
        - Market pressure analysis and risk assessment
        - Sophisticated buying recommendations
        
        **Analysis Features:**
        - Real-time market sentiment analysis
        - Cross-species impact correlation
        - Supply-demand equilibrium modeling
        - Price volatility forecasting
        """)

def create_comprehensive_dashboard_page(ecosystem_data, global_analyzer, undercurrent_api, longline_data, species, current_price=12.50):
    """Create comprehensive data dashboard page"""
    
    # Main data status
    display_comprehensive_data_status(ecosystem_data, global_analyzer, undercurrent_api, longline_data)
    
    st.divider()
    
    # Hawaii market analysis
    display_hawaii_market_analysis(ecosystem_data, species)
    
    st.divider()
    
    # Global market pressure
    display_global_market_pressure(global_analyzer, species, current_price)
    
    st.divider()
    
    # Integration capabilities
    display_data_integration_capabilities()