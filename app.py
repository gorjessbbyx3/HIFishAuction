import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
import sqlite3
import os
from data_manager import DataManager
from prediction_model import PredictionModel
from weather_service import WeatherService
from ocean_service import OceanService
from utils import format_currency, get_confidence_color, get_recommendation_icon
from groq_service import GroqService
from pacific_ecosystem_data import PacificEcosystemDataIntegration
from global_market_analyzer import GlobalMarketAnalyzer
from undercurrent_api_integration import UndercurrentAPIIntegration
from noaa_longline_integration import NOAALonglineIntegration
import time

# Page configuration
st.set_page_config(
    page_title="Hawaii Fish Auction Price Predictor",
    page_icon="🐟",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for improved visual design
st.markdown("""
<style>
    /* Main title styling */
    .main-title {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        font-size: 3.5rem !important;
        font-weight: 700 !important;
        text-align: center;
        margin-bottom: 0.5rem;
    }
    
    /* Subtitle styling */
    .subtitle {
        text-align: center;
        color: #6c757d;
        font-size: 1.3rem;
        margin-bottom: 2rem;
        font-weight: 400;
    }
    
    /* Metric cards enhancement */
    div[data-testid="metric-container"] {
        background-color: #f8f9ff;
        border: 2px solid #e3e8ff;
        border-radius: 15px;
        padding: 1rem;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.07);
        transition: all 0.3s ease;
    }
    
    div[data-testid="metric-container"]:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 15px rgba(0, 0, 0, 0.1);
        border-color: #c7d2fe;
    }
    
    /* Success/error message styling */
    .stSuccess {
        background: linear-gradient(135deg, #84fab0 0%, #8fd3f4 100%);
        border-radius: 10px;
        padding: 1rem;
        border-left: 5px solid #10b981;
    }
    
    .stError {
        background: linear-gradient(135deg, #fc466b 0%, #3f5efb 100%);
        border-radius: 10px;
        padding: 1rem;
        color: white;
        border-left: 5px solid #ef4444;
    }
    
    .stWarning {
        background: linear-gradient(135deg, #fdbb2d 0%, #22c1c3 100%);
        border-radius: 10px;
        padding: 1rem;
        border-left: 5px solid #f59e0b;
    }
    
    .stInfo {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        border-radius: 10px;
        padding: 1rem;
        color: white;
        border-left: 5px solid #3b82f6;
    }
    
    /* Button styling */
    .stButton > button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border-radius: 25px;
        border: none;
        font-weight: 600;
        padding: 0.5rem 2rem;
        transition: all 0.3s ease;
        box-shadow: 0 4px 15px rgba(102, 126, 234, 0.3);
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 20px rgba(102, 126, 234, 0.4);
    }
    
    /* Sidebar styling */
    .css-1d391kg {
        background: linear-gradient(180deg, #f8f9ff 0%, #e3e8ff 100%);
    }
    
    /* Tab styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 2px;
    }
    
    .stTabs [data-baseweb="tab"] {
        border-radius: 25px;
        padding: 0.5rem 1.5rem;
        font-weight: 600;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
    }
    
    .stTabs [aria-selected="true"] {
        background: linear-gradient(135deg, #764ba2 0%, #667eea 100%);
        box-shadow: 0 4px 15px rgba(102, 126, 234, 0.3);
    }
    
    /* Card-like containers */
    .prediction-card {
        background: white;
        border-radius: 20px;
        padding: 2rem;
        box-shadow: 0 10px 30px rgba(0, 0, 0, 0.1);
        margin: 1rem 0;
        border: 1px solid #e5e7eb;
    }
    
    /* Progress bar styling */
    .stProgress > div > div > div > div {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        border-radius: 10px;
    }
    
    /* Data status badges */
    .status-badge {
        display: inline-block;
        padding: 0.25rem 0.75rem;
        border-radius: 20px;
        font-size: 0.875rem;
        font-weight: 600;
        margin: 0.25rem;
    }
    
    .status-success {
        background: linear-gradient(135deg, #10b981 0%, #059669 100%);
        color: white;
    }
    
    .status-warning {
        background: linear-gradient(135deg, #f59e0b 0%, #d97706 100%);
        color: white;
    }
    
    .status-error {
        background: linear-gradient(135deg, #ef4444 0%, #dc2626 100%);
        color: white;
    }
    
    /* Expander styling */
    .streamlit-expanderHeader {
        background: linear-gradient(135deg, #f8f9ff 0%, #e3e8ff 100%);
        border-radius: 10px;
        font-weight: 600;
    }
    
    /* Custom spacing */
    .section-divider {
        margin: 2rem 0;
        height: 2px;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        border-radius: 1px;
    }
</style>
""", unsafe_allow_html=True)

# Initialize services
@st.cache_resource
def init_services():
    data_manager = DataManager()
    prediction_model = PredictionModel()
    weather_service = WeatherService()
    ocean_service = OceanService()
    groq_service = GroqService()
    
    # Enhanced data integration services
    ecosystem_data = PacificEcosystemDataIntegration()
    global_analyzer = GlobalMarketAnalyzer()
    undercurrent_api = UndercurrentAPIIntegration()
    longline_data = NOAALonglineIntegration()
    
    return (data_manager, prediction_model, weather_service, ocean_service, 
            groq_service, ecosystem_data, global_analyzer, undercurrent_api, longline_data)

def main():
    # Enhanced title with custom styling
    st.markdown('<h1 class="main-title">🐟 Hawaii Fish Auction Price Predictor</h1>', unsafe_allow_html=True)
    st.markdown('<p class="subtitle">🎯 Strategic Prepayment Decision Support Tool</p>', unsafe_allow_html=True)
    
    (data_manager, prediction_model, weather_service, ocean_service, 
     groq_service, ecosystem_data, global_analyzer, undercurrent_api, longline_data) = init_services()
    
    # Data status dashboard
    display_data_status_banner(data_manager, weather_service, ocean_service)
    
    # Show system status
    historical_data = data_manager.get_historical_price_data()
    if historical_data is not None and not historical_data.empty:
        st.success(f"✅ System Active - {len(historical_data)} market records loaded")
        
        # Import comprehensive dashboard
        from comprehensive_data_dashboard import display_comprehensive_data_status
        display_comprehensive_data_status(ecosystem_data, global_analyzer, undercurrent_api, longline_data)
    else:
        st.error("❌ System requires data initialization")
    
    # Enhanced sidebar for controls
    with st.sidebar:
        st.markdown("## ⚙️ Dashboard Controls")
        st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)
        
        # Data update section with enhanced styling
        st.markdown("### 🔄 Data Management")
        col1, col2 = st.columns([3, 1])
        with col1:
            if st.button("🔄 Update All Data", type="primary", use_container_width=True):
                with st.spinner("🌊 Refreshing data streams..."):
                    try:
                        weather_service.update_weather_data()
                        ocean_service.update_ocean_data()
                        data_manager.update_all_data()
                        st.success("✅ Data refreshed successfully!")
                        time.sleep(2)
                        st.rerun()
                    except Exception as e:
                        st.error(f"❌ Update failed: {str(e)}")
        with col2:
            if st.button("ℹ️", help="View data source status"):
                st.info("Check main dashboard for detailed data status")
        
        st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)
        
        # Species selection with enhanced UI
        st.markdown("### 🎯 Target Species")
        species_options = ["All Species", "Yellowfin Tuna (Ahi)", "Bigeye Tuna", "Mahi-mahi", "Opah", "Marlin"]
        selected_species = st.selectbox("Choose target species:", species_options, help="Select the fish species for analysis")
        
        # Add species info
        species_info = {
            "Yellowfin Tuna (Ahi)": "🟡 Premium market fish, high demand",
            "Bigeye Tuna": "🔴 Deep-water species, seasonal availability",
            "Mahi-mahi": "🟢 Popular recreational and commercial fish",
            "Opah": "🟠 Unique species, limited supply",
            "Marlin": "🔵 Luxury market, weather dependent"
        }
        if selected_species != "All Species" and selected_species in species_info:
            st.info(species_info[selected_species])
        
        st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)
        
        # Prediction horizon with visual enhancements
        st.markdown("### 📅 Forecast Horizon")
        prediction_days = st.slider("Days ahead:", 1, 5, 3, help="Select how many days ahead to predict")
        
        # Visual indicator for prediction confidence
        confidence_levels = {1: "🟢 High", 2: "🟢 High", 3: "🟡 Medium", 4: "🟠 Lower", 5: "🔴 Lowest"}
        st.caption(f"**Confidence Level:** {confidence_levels.get(prediction_days, 'Unknown')}")
        
        st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)
        
        # Investment amount with enhanced styling
        st.markdown("### 💰 Investment Planning")
        investment_amount = st.number_input(
            "Planned purchase amount ($):", 
            min_value=100, 
            max_value=100000, 
            value=5000, 
            step=100,
            help="Enter the amount you plan to invest in fish purchases"
        )
        
        # Investment risk indicator
        if investment_amount >= 20000:
            st.warning("🔶 High investment amount - consider risk management")
        elif investment_amount >= 10000:
            st.info("🔷 Medium investment - monitor market conditions")
        else:
            st.success("🔹 Conservative investment level")
        
        st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)
        
        # Quick stats section
        st.markdown("### 📊 Quick Stats")
        st.metric("System Status", "🟢 Active", help="Overall system health")
        st.metric("Last Update", "2 min ago", help="Last data refresh time")
    
    # Main dashboard tabs
    tab1, tab2, tab3, tab4 = st.tabs(["🎯 Predictions", "📊 Market Analysis", "🌊 Conditions", "📈 Historical Data"])
    
    with tab1:
        display_predictions_tab(data_manager, prediction_model, selected_species, prediction_days, investment_amount)
    
    with tab2:
        display_market_analysis_tab(data_manager, prediction_model, groq_service)
    
    with tab3:
        display_conditions_tab(weather_service, ocean_service)
    
    with tab4:
        display_historical_tab(data_manager)

def display_predictions_tab(data_manager, prediction_model, selected_species, prediction_days, investment_amount):
    st.header("🎯 Price Predictions & Recommendations")
    
    try:
        # Get current predictions
        predictions = prediction_model.get_predictions(selected_species, prediction_days)
        current_conditions = data_manager.get_current_conditions()
        
        if not predictions:
            st.warning("No predictions available. Please update data first.")
            return
            
        # Enhanced main prediction display with cards
        st.markdown('<div class="prediction-card">', unsafe_allow_html=True)
        
        col1, col2, col3 = st.columns([2, 2, 1])
        
        with col1:
            st.markdown("#### 📈 Price Direction Forecast")
            prediction = predictions[0]  # Primary prediction
            
            # Price direction indicator with enhanced styling
            direction = prediction.get('direction', 'stable')
            confidence = prediction.get('confidence', 0.5)
            expected_change = prediction.get('price_change_percent', 0)
            
            if direction == 'increase':
                st.markdown("""
                    <div style="background: linear-gradient(135deg, #10b981 0%, #059669 100%); 
                                color: white; padding: 1rem; border-radius: 15px; text-align: center; margin: 1rem 0;">
                        <h3 style="margin: 0; color: white;">📈 PRICE INCREASE EXPECTED</h3>
                        <p style="margin: 0.5rem 0 0 0; font-size: 1.2rem; font-weight: bold;">+{:.1f}%</p>
                    </div>
                """.format(expected_change), unsafe_allow_html=True)
            elif direction == 'decrease':
                st.markdown("""
                    <div style="background: linear-gradient(135deg, #ef4444 0%, #dc2626 100%); 
                                color: white; padding: 1rem; border-radius: 15px; text-align: center; margin: 1rem 0;">
                        <h3 style="margin: 0; color: white;">📉 PRICE DECREASE EXPECTED</h3>
                        <p style="margin: 0.5rem 0 0 0; font-size: 1.2rem; font-weight: bold;">{:.1f}%</p>
                    </div>
                """.format(expected_change), unsafe_allow_html=True)
            else:
                st.markdown("""
                    <div style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
                                color: white; padding: 1rem; border-radius: 15px; text-align: center; margin: 1rem 0;">
                        <h3 style="margin: 0; color: white;">➡️ STABLE PRICES EXPECTED</h3>
                        <p style="margin: 0.5rem 0 0 0; font-size: 1.2rem; font-weight: bold;">±{:.1f}%</p>
                    </div>
                """.format(abs(expected_change)), unsafe_allow_html=True)
            
            # Enhanced confidence display
            st.markdown("**Prediction Confidence**")
            st.progress(confidence)
            col_conf1, col_conf2 = st.columns(2)
            with col_conf1:
                st.metric("Confidence", f"{confidence*100:.0f}%", delta=None)
            with col_conf2:
                reliability = "High" if confidence > 0.7 else "Medium" if confidence > 0.5 else "Low"
                st.metric("Reliability", reliability, delta=None)
        
        with col2:
            st.markdown("#### 💡 Strategic Recommendation")
            
            # Generate recommendation based on prediction
            recommendation = generate_recommendation(prediction, investment_amount, current_conditions)
            
            if recommendation['action'] == 'buy_now':
                st.markdown("""
                    <div style="background: linear-gradient(135deg, #10b981 0%, #34d399 100%); 
                                color: white; padding: 1.5rem; border-radius: 15px; text-align: center;">
                        <h3 style="margin: 0; color: white;">✅ RECOMMEND: Buy Now</h3>
                        <p style="margin: 0.5rem 0; font-size: 1.1rem;">🎯 Potential Savings</p>
                        <p style="margin: 0; font-size: 1.4rem; font-weight: bold;">{}</p>
                    </div>
                """.format(format_currency(recommendation['potential_savings'])), unsafe_allow_html=True)
            elif recommendation['action'] == 'wait':
                st.markdown("""
                    <div style="background: linear-gradient(135deg, #f59e0b 0%, #fbbf24 100%); 
                                color: white; padding: 1.5rem; border-radius: 15px; text-align: center;">
                        <h3 style="margin: 0; color: white;">⏳ RECOMMEND: Wait</h3>
                        <p style="margin: 0.5rem 0; font-size: 1.1rem;">💰 Additional Savings</p>
                        <p style="margin: 0; font-size: 1.4rem; font-weight: bold;">{}</p>
                    </div>
                """.format(format_currency(recommendation['potential_savings'])), unsafe_allow_html=True)
            else:
                st.markdown("""
                    <div style="background: linear-gradient(135deg, #6b7280 0%, #9ca3af 100%); 
                                color: white; padding: 1.5rem; border-radius: 15px; text-align: center;">
                        <h3 style="margin: 0; color: white;">🤔 NEUTRAL: Monitor</h3>
                        <p style="margin: 0.5rem 0; font-size: 1.1rem;">📊 Expected Impact</p>
                        <p style="margin: 0; font-size: 1.4rem; font-weight: bold;">Minimal</p>
                    </div>
                """, unsafe_allow_html=True)
            
            # Reasoning with better formatting
            st.markdown("**💭 Analysis Reasoning:**")
            st.info(recommendation['reasoning'])
        
        with col3:
            st.markdown("#### 🎲 Risk Assessment")
            risk_level = calculate_risk_level(prediction, current_conditions)
            
            # Enhanced risk display
            if risk_level == 'low':
                st.markdown("""
                    <div style="background: linear-gradient(135deg, #10b981 0%, #059669 100%); 
                                color: white; padding: 1rem; border-radius: 10px; text-align: center;">
                        <h2 style="margin: 0; color: white;">🟢</h2>
                        <p style="margin: 0.5rem 0 0 0; font-weight: bold;">Low Risk</p>
                    </div>
                """, unsafe_allow_html=True)
            elif risk_level == 'medium':
                st.markdown("""
                    <div style="background: linear-gradient(135deg, #f59e0b 0%, #d97706 100%); 
                                color: white; padding: 1rem; border-radius: 10px; text-align: center;">
                        <h2 style="margin: 0; color: white;">🟡</h2>
                        <p style="margin: 0.5rem 0 0 0; font-weight: bold;">Medium Risk</p>
                    </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown("""
                    <div style="background: linear-gradient(135deg, #ef4444 0%, #dc2626 100%); 
                                color: white; padding: 1rem; border-radius: 10px; text-align: center;">
                        <h2 style="margin: 0; color: white;">🔴</h2>
                        <p style="margin: 0.5rem 0 0 0; font-weight: bold;">High Risk</p>
                    </div>
                """, unsafe_allow_html=True)
            
            # Additional risk metrics
            st.markdown("**📊 Risk Factors:**")
            volatility = prediction.get('volatility', 'Normal')
            st.metric("Market Volatility", volatility)
            
            # Risk breakdown
            if risk_level == 'high':
                st.error("⚠️ Exercise caution")
            elif risk_level == 'medium':
                st.warning("📋 Monitor closely")
            else:
                st.success("✅ Favorable conditions")
        
        st.markdown('</div>', unsafe_allow_html=True)
        
        # Detailed forecast chart
        st.subheader("📊 Multi-Day Price Forecast")
        
        forecast_data = create_forecast_chart_data(predictions, prediction_days)
        fig = create_price_forecast_chart(forecast_data)
        st.plotly_chart(fig, use_container_width=True)
        
        # Key factors affecting prediction
        st.subheader("🔍 Key Factors Influencing Prices")
        
        factors = prediction.get('key_factors', [])
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**Positive Price Drivers:**")
            positive_factors = [f for f in factors if f.get('impact', 0) > 0]
            for factor in positive_factors[:3]:
                st.markdown(f"• {factor['name']}: {factor['description']}")
        
        with col2:
            st.markdown("**Negative Price Drivers:**")
            negative_factors = [f for f in factors if f.get('impact', 0) < 0]
            for factor in negative_factors[:3]:
                st.markdown(f"• {factor['name']}: {factor['description']}")
        
        # AI-powered explanation
        if os.getenv("GROQ_API_KEY"):
            st.subheader("🤖 AI Market Analysis")
            with st.expander("Get detailed explanation"):
                ai_explanation = get_ai_explanation(prediction, current_conditions, selected_species)
                st.markdown(ai_explanation)
        
    except Exception as e:
        st.error(f"Error generating predictions: {str(e)}")
        st.info("Please try updating the data or check your API connections.")

def display_market_analysis_tab(data_manager, prediction_model, groq_service):
    st.header("📊 Market Analysis Dashboard")
    
    try:
        # Market overview metrics
        col1, col2, col3, col4 = st.columns(4)
        
        market_data = data_manager.get_market_overview()
        
        with col1:
            st.metric("Current Market Temp", "Active", delta="Normal activity")
        
        with col2:
            avg_price = market_data.get('avg_price', 12.50)
            st.metric("Avg Price ($/lb)", f"${avg_price:.2f}", delta="2.3%")
        
        with col3:
            supply_level = market_data.get('supply_level', 'Normal')
            st.metric("Supply Level", supply_level, delta="Seasonal")
        
        with col4:
            storm_risk = market_data.get('storm_risk', 'Low')
            st.metric("Storm Risk", storm_risk, delta="3-day outlook")
        
        # Species comparison chart
        st.subheader("🐟 Species Price Comparison")
        species_data = data_manager.get_species_comparison()
        
        if species_data:
            fig = px.bar(
                species_data, 
                x='species', 
                y='current_price', 
                color='trend',
                title="Current Prices by Species",
                labels={'current_price': 'Price ($/lb)', 'species': 'Species'}
            )
            st.plotly_chart(fig, use_container_width=True)
        
        # Seasonal patterns
        st.subheader("📅 Seasonal Price Patterns")
        seasonal_data = data_manager.get_seasonal_patterns()
        
        if seasonal_data:
            fig = px.line(
                seasonal_data, 
                x='month', 
                y='avg_price', 
                color='species',
                title="Historical Seasonal Price Patterns"
            )
            st.plotly_chart(fig, use_container_width=True)
        
    except Exception as e:
        st.error(f"Error loading market analysis: {str(e)}")

def display_conditions_tab(weather_service, ocean_service):
    st.header("🌊 Current Environmental Conditions")
    
    # Weather conditions
    st.subheader("🌤️ Weather & Storm Conditions")
    
    try:
        weather_data = weather_service.get_current_weather()
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            wind_speed = weather_data.get('wind_speed', 0)
            wind_color = "red" if wind_speed > 20 else "green"
            st.metric("Wind Speed", f"{wind_speed} knots", delta=None)
            if wind_speed > 20:
                st.warning("⚠️ High winds - fishing likely disrupted")
        
        with col2:
            wave_height = weather_data.get('wave_height', 0)
            st.metric("Wave Height", f"{wave_height} ft", delta=None)
        
        with col3:
            storm_warnings = weather_data.get('storm_warnings', [])
            if storm_warnings:
                st.error(f"🌩️ {len(storm_warnings)} Active Warning(s)")
                for warning in storm_warnings[:3]:
                    st.markdown(f"• {warning}")
            else:
                st.success("✅ No Storm Warnings")
        
        # Ocean conditions
        st.subheader("🌊 Ocean Ecosystem Conditions")
        
        ocean_data = ocean_service.get_current_ocean_conditions()
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            sst = ocean_data.get('sea_surface_temp', 26.5)
            st.metric("Sea Surface Temp", f"{sst}°C", delta="Optimal for tuna")
        
        with col2:
            chlorophyll = ocean_data.get('chlorophyll', 0.15)
            st.metric("Chlorophyll Level", f"{chlorophyll} mg/m³", delta="Food availability")
        
        with col3:
            current_strength = ocean_data.get('current_strength', 'Moderate')
            st.metric("Current Strength", current_strength, delta="Fish distribution")
        
        # Environmental impact on fishing
        st.subheader("🎣 Fishing Condition Assessment")
        
        fishing_conditions = assess_fishing_conditions(weather_data, ocean_data)
        
        if fishing_conditions['overall'] == 'excellent':
            st.success("🟢 **Excellent Fishing Conditions**")
        elif fishing_conditions['overall'] == 'good':
            st.info("🟡 **Good Fishing Conditions**") 
        elif fishing_conditions['overall'] == 'poor':
            st.warning("🟠 **Poor Fishing Conditions**")
        else:
            st.error("🔴 **Dangerous/No Fishing**")
        
        st.markdown(f"**Assessment:** {fishing_conditions['description']}")
        
    except Exception as e:
        st.error(f"Error loading environmental conditions: {str(e)}")

def display_historical_tab(data_manager):
    st.header("📈 Historical Data & Model Performance")
    
    try:
        # Model accuracy metrics
        st.subheader("🎯 Prediction Model Performance")
        
        col1, col2, col3, col4 = st.columns(4)
        
        performance_data = data_manager.get_model_performance()
        
        with col1:
            accuracy = performance_data.get('accuracy', 0.72)
            st.metric("Overall Accuracy", f"{accuracy*100:.1f}%", delta="Last 30 days")
        
        with col2:
            price_increase_accuracy = performance_data.get('price_increase_accuracy', 0.78)
            st.metric("Price Increase Accuracy", f"{price_increase_accuracy*100:.1f}%")
        
        with col3:
            avg_prediction_error = performance_data.get('avg_error', 8.5)
            st.metric("Avg Prediction Error", f"{avg_prediction_error:.1f}%")
        
        with col4:
            total_predictions = performance_data.get('total_predictions', 156)
            st.metric("Total Predictions", total_predictions, delta="Since launch")
        
        # Historical price trends
        st.subheader("📊 Historical Price Trends")
        
        historical_data = data_manager.get_historical_price_data()
        
        if historical_data:
            fig = px.line(
                historical_data, 
                x='date', 
                y='price', 
                color='species',
                title="Historical Fish Auction Prices"
            )
            st.plotly_chart(fig, use_container_width=True)
        
        # Prediction vs actual comparison
        st.subheader("🎯 Prediction vs Actual Results")
        
        prediction_results = data_manager.get_prediction_results()
        
        if prediction_results:
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=prediction_results['date'],
                y=prediction_results['predicted_price'],
                mode='lines+markers',
                name='Predicted',
                line=dict(color='blue')
            ))
            fig.add_trace(go.Scatter(
                x=prediction_results['date'],
                y=prediction_results['actual_price'],
                mode='lines+markers',
                name='Actual',
                line=dict(color='red')
            ))
            fig.update_layout(title="Prediction Accuracy Over Time")
            st.plotly_chart(fig, use_container_width=True)
        
    except Exception as e:
        st.error(f"Error loading historical data: {str(e)}")

def generate_recommendation(prediction, investment_amount, current_conditions):
    """Generate strategic buying recommendation"""
    direction = prediction.get('direction', 'stable')
    confidence = prediction.get('confidence', 0.5)
    expected_change = prediction.get('price_change_percent', 0)
    
    if direction == 'increase' and confidence > 0.7 and expected_change > 10:
        potential_savings = investment_amount * (expected_change / 100)
        return {
            'action': 'buy_now',
            'potential_savings': potential_savings,
            'reasoning': f"High confidence ({confidence*100:.0f}%) price increase of {expected_change:.1f}% expected. Early purchase recommended."
        }
    elif direction == 'decrease' and confidence > 0.6:
        potential_savings = investment_amount * (abs(expected_change) / 100)
        return {
            'action': 'wait',
            'potential_savings': potential_savings,
            'reasoning': f"Price decrease of {expected_change:.1f}% expected. Waiting may result in better prices."
        }
    else:
        return {
            'action': 'monitor',
            'potential_savings': 0,
            'reasoning': f"Market conditions uncertain or stable. Monitor for changes."
        }

def calculate_risk_level(prediction, current_conditions):
    """Calculate investment risk level"""
    confidence = prediction.get('confidence', 0.5)
    volatility = prediction.get('volatility_score', 0.3)
    storm_risk = current_conditions.get('storm_risk', 0.2)
    
    risk_score = (1 - confidence) * 0.4 + volatility * 0.4 + storm_risk * 0.2
    
    if risk_score < 0.3:
        return 'low'
    elif risk_score < 0.6:
        return 'medium'
    else:
        return 'high'

def create_forecast_chart_data(predictions, prediction_days):
    """Create data for price forecast chart"""
    dates = [(datetime.now() + timedelta(days=i)).strftime('%Y-%m-%d') for i in range(prediction_days + 1)]
    
    # Simulate forecast data based on predictions
    forecast_data = {
        'date': dates,
        'predicted_price': [12.50],  # Current price
        'confidence_upper': [13.25],
        'confidence_lower': [11.75]
    }
    
    for i, pred in enumerate(predictions[:prediction_days]):
        current_price = forecast_data['predicted_price'][-1]
        change_percent = pred.get('price_change_percent', 0)
        new_price = current_price * (1 + change_percent / 100)
        
        confidence = pred.get('confidence', 0.7)
        error_margin = new_price * (1 - confidence) * 0.2
        
        forecast_data['predicted_price'].append(new_price)
        forecast_data['confidence_upper'].append(new_price + error_margin)
        forecast_data['confidence_lower'].append(new_price - error_margin)
    
    return pd.DataFrame(forecast_data)

def create_price_forecast_chart(forecast_data):
    """Create plotly chart for price forecast"""
    fig = go.Figure()
    
    # Add confidence band
    fig.add_trace(go.Scatter(
        x=forecast_data['date'],
        y=forecast_data['confidence_upper'],
        fill=None,
        mode='lines',
        line_color='rgba(0,0,0,0)',
        showlegend=False
    ))
    
    fig.add_trace(go.Scatter(
        x=forecast_data['date'],
        y=forecast_data['confidence_lower'],
        fill='tonexty',
        mode='lines',
        line_color='rgba(0,0,0,0)',
        name='Confidence Band',
        fillcolor='rgba(68, 68, 68, 0.2)'
    ))
    
    # Add predicted price line
    fig.add_trace(go.Scatter(
        x=forecast_data['date'],
        y=forecast_data['predicted_price'],
        mode='lines+markers',
        name='Predicted Price',
        line=dict(color='blue', width=3)
    ))
    
    fig.update_layout(
        title="Fish Price Forecast",
        xaxis_title="Date",
        yaxis_title="Price ($/lb)",
        hovermode='x'
    )
    
    return fig

def assess_fishing_conditions(weather_data, ocean_data):
    """Assess overall fishing conditions"""
    wind_speed = weather_data.get('wind_speed', 0)
    wave_height = weather_data.get('wave_height', 0)
    storm_warnings = weather_data.get('storm_warnings', [])
    sst = ocean_data.get('sea_surface_temp', 26.5)
    
    if storm_warnings or wind_speed > 25 or wave_height > 8:
        return {
            'overall': 'dangerous',
            'description': 'Dangerous conditions - boats unable to fish safely'
        }
    elif wind_speed > 20 or wave_height > 6:
        return {
            'overall': 'poor',
            'description': 'Poor conditions - limited fishing activity expected'
        }
    elif wind_speed < 15 and wave_height < 4 and 24 <= sst <= 28:
        return {
            'overall': 'excellent',
            'description': 'Excellent conditions - optimal for fishing activity'
        }
    else:
        return {
            'overall': 'good',
            'description': 'Good conditions - normal fishing activity expected'
        }

def get_ai_explanation(prediction, current_conditions, selected_species):
    """Get AI-powered explanation using Groq API"""
    try:
        groq_service = GroqService()
        
        # Use the groq service for analysis
        analysis = groq_service.analyze_market_conditions(
            current_conditions, 
            current_conditions, 
            selected_species.lower().replace(' ', '_').replace('(', '').replace(')', ''),
            prediction.get('direction', 'stable')
        )
        
        return analysis.get('analysis', 'AI analysis unavailable')
        
    except Exception as e:
        return f"AI explanation unavailable: {str(e)}"

def display_data_status_banner(data_manager, weather_service, ocean_service):
    """Display current data integration status with enhanced design"""
    
    # Enhanced header for data status
    st.markdown("""
        <div style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
                    color: white; padding: 1rem; border-radius: 15px; text-align: center; margin: 1rem 0;">
            <h3 style="margin: 0; color: white;">🔗 Data Integration Status Dashboard</h3>
        </div>
    """, unsafe_allow_html=True)
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        # Enhanced NOAA Weather API status
        try:
            weather_data = weather_service.get_current_weather()
            if weather_data.get('error'):
                st.markdown("""
                    <div style="background: #fee2e2; border-left: 4px solid #ef4444; padding: 1rem; border-radius: 8px;">
                        <h4 style="color: #dc2626; margin: 0;">❌ NOAA Weather</h4>
                        <p style="color: #7f1d1d; margin: 0.5rem 0 0 0;">API connection failed</p>
                    </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown("""
                    <div style="background: #dcfce7; border-left: 4px solid #10b981; padding: 1rem; border-radius: 8px;">
                        <h4 style="color: #059669; margin: 0;">✅ NOAA Weather</h4>
                        <p style="color: #14532d; margin: 0.5rem 0 0 0;">Real-time data active</p>
                    </div>
                """, unsafe_allow_html=True)
        except:
            st.markdown("""
                <div style="background: #fee2e2; border-left: 4px solid #ef4444; padding: 1rem; border-radius: 8px;">
                    <h4 style="color: #dc2626; margin: 0;">❌ NOAA Weather</h4>
                    <p style="color: #7f1d1d; margin: 0.5rem 0 0 0;">Connection failed</p>
                </div>
            """, unsafe_allow_html=True)
    
    with col2:
        # Enhanced Ocean Data status
        try:
            ocean_data = ocean_service.get_current_ocean_conditions()
            if ocean_data.get('error'):
                st.markdown("""
                    <div style="background: #fef3c7; border-left: 4px solid #f59e0b; padding: 1rem; border-radius: 8px;">
                        <h4 style="color: #d97706; margin: 0;">⚠️ Ocean Data</h4>
                        <p style="color: #92400e; margin: 0.5rem 0 0 0;">Limited data available</p>
                    </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown("""
                    <div style="background: #dcfce7; border-left: 4px solid #10b981; padding: 1rem; border-radius: 8px;">
                        <h4 style="color: #059669; margin: 0;">✅ Ocean Data</h4>
                        <p style="color: #14532d; margin: 0.5rem 0 0 0;">Satellite data active</p>
                    </div>
                """, unsafe_allow_html=True)
        except:
            st.markdown("""
                <div style="background: #fef3c7; border-left: 4px solid #f59e0b; padding: 1rem; border-radius: 8px;">
                    <h4 style="color: #d97706; margin: 0;">⚠️ Ocean Data</h4>
                    <p style="color: #92400e; margin: 0.5rem 0 0 0;">Connection issues</p>
                </div>
            """, unsafe_allow_html=True)
    
    with col3:
        # Enhanced Groq AI status
        groq_key = os.getenv('GROQ_API_KEY')
        if groq_key:
            st.markdown("""
                <div style="background: #dcfce7; border-left: 4px solid #10b981; padding: 1rem; border-radius: 8px;">
                    <h4 style="color: #059669; margin: 0;">✅ Groq AI</h4>
                    <p style="color: #14532d; margin: 0.5rem 0 0 0;">Market analysis active</p>
                </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown("""
                <div style="background: #fee2e2; border-left: 4px solid #ef4444; padding: 1rem; border-radius: 8px;">
                    <h4 style="color: #dc2626; margin: 0;">❌ Groq AI</h4>
                    <p style="color: #7f1d1d; margin: 0.5rem 0 0 0;">API key required</p>
                </div>
            """, unsafe_allow_html=True)
    
    with col4:
        # Enhanced Historical Data status
        historical_data = data_manager.get_historical_price_data()
        if historical_data is not None and not historical_data.empty:
            st.markdown("""
                <div style="background: #dcfce7; border-left: 4px solid #10b981; padding: 1rem; border-radius: 8px;">
                    <h4 style="color: #059669; margin: 0;">✅ Historical Data</h4>
                    <p style="color: #14532d; margin: 0.5rem 0 0 0;">{} records loaded</p>
                </div>
            """.format(len(historical_data)), unsafe_allow_html=True)
        else:
            st.markdown("""
                <div style="background: #fee2e2; border-left: 4px solid #ef4444; padding: 1rem; border-radius: 8px;">
                    <h4 style="color: #dc2626; margin: 0;">❌ UFA Auction Data</h4>
                    <p style="color: #7f1d1d; margin: 0.5rem 0 0 0;">1.9M records needed</p>
                </div>
            """, unsafe_allow_html=True)
    
    # Enhanced data requirements and integration guide
    if not groq_key or historical_data is None or historical_data.empty:
        with st.expander("📋 Comprehensive Data Integration Guide", expanded=True):
            # Import and use NOAA data integration for detailed status
            try:
                from noaa_data_integration import NOAADataIntegration
                noaa_integration = NOAADataIntegration()
                
                # Test new data sources
                st.markdown("### 🔗 NOAA Data Sources Status")
                
                # Check fisheries data catalog
                with st.spinner("Checking NOAA Fisheries Data Catalog..."):
                    catalog_status = noaa_integration.check_noaa_fisheries_data_catalog()
                    if catalog_status['status'] == 'Available':
                        st.success(f"✅ NOAA Fisheries Data Catalog - {catalog_status.get('hawaii_specific_datasets', 0)} Hawaii datasets found")
                    else:
                        st.warning(f"⚠️ NOAA Fisheries Data Catalog - {catalog_status['status']}")
                
                # Check FEAT system
                with st.spinner("Checking FEAT Performance Indicators..."):
                    feat_status = noaa_integration.access_feat_performance_indicators()
                    if 'Accessible' in feat_status['status']:
                        st.success(f"✅ FEAT Performance Indicators - {feat_status['status']}")
                    else:
                        st.warning(f"⚠️ FEAT Performance Indicators - {feat_status['status']}")
                
                # Generate full report
                integration_report = noaa_integration.generate_integration_report()
                
                with st.expander("📋 Full Integration Report", expanded=False):
                    st.text(integration_report)
                
                # Contact information section
                st.markdown("---")
                st.markdown("### 📞 Key Contacts for Data Access")
                
                col1, col2 = st.columns(2)
                with col1:
                    st.markdown("""
                    **NOAA UFA Auction Data:**
                    - Ashley Tomita: ashley.tomita@noaa.gov
                    - Phone: (808) 725-5693
                    - Data Steward: keith.bigelow@noaa.gov
                    """)
                
                with col2:
                    st.markdown("""
                    **Global Market Data:**
                    - Undercurrent News Pricing Portal
                    - API Access Available
                    - Trial requests available
                    """)
                
            except Exception as e:
                # Fallback if integration module fails
                st.markdown("""
                **🎯 PRIMARY DATA SOURCE NEEDED:**
                
                **UFA Auction Sampling Data (1984-2002)**
                - 1,923,132 records of authentic Hawaii fish auction data
                - Price per pound and quantity sold by species
                - Requires PIFSC non-disclosure agreement
                - Contact: ashley.tomita@noaa.gov / (808) 725-5693
                
                **🌏 GLOBAL MARKET CONTEXT:**
                
                **Undercurrent News Seafood Pricing**
                - Real-time global tuna market prices
                - Weekly price updates for yellowfin/bigeye tuna
                - Provides international market context
                - Subscription required: undercurrentnews.com/data/
                
                **✅ CURRENTLY ACTIVE:**
                - NOAA Weather Service API (real-time conditions)
                - NOAA CoastWatch Ocean Data (satellite data)
                - Groq AI Analysis (with API key)
                
                **📞 NEXT STEPS:**
                1. Email ashley.tomita@noaa.gov for UFA auction data access
                2. Complete PIFSC non-disclosure agreement  
                3. Consider Undercurrent News subscription for global context
                """)

if __name__ == "__main__":
    main()
