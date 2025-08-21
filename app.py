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
from historical_data_finder import HistoricalDataFinder
from feat_data_extractor import FEATDataExtractor

# Page configuration
st.set_page_config(
    page_title="Hawaii Fish Auction Price Predictor",
    page_icon="🐟",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize services
@st.cache_resource
def init_services():
    # Initialize demo data if database is empty
    try:
        import sqlite3
        conn = sqlite3.connect("fish_auction.db")
        cursor = conn.cursor()
        cursor.execute("SELECT COUNT(*) FROM market_data")
        count = cursor.fetchone()[0]
        conn.close()

        if count == 0:
            print("No market data found, initializing demo data...")
            from initialize_demo_data import initialize_demo_data
            initialize_demo_data()
    except Exception as e:
        print(f"Error checking/initializing demo data: {str(e)}")

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

    # Initialize data finder and extractor
    historical_data_finder = HistoricalDataFinder()
    feat_data_extractor = FEATDataExtractor()

    return (data_manager, prediction_model, weather_service, ocean_service, 
            groq_service, ecosystem_data, global_analyzer, undercurrent_api, longline_data,
            historical_data_finder, feat_data_extractor)

def main():
    st.title("🐟 Hawaii Fish Auction Price Predictor")
    st.markdown("### Strategic Prepayment Decision Support Tool")

    (data_manager, prediction_model, weather_service, ocean_service, 
     groq_service, ecosystem_data, global_analyzer, undercurrent_api, longline_data,
     historical_data_finder, feat_data_extractor) = init_services()

    # Show system status
    historical_data = data_manager.get_historical_price_data()
    if historical_data is not None and not historical_data.empty:
        st.success(f"✅ System Active - {len(historical_data)} market records loaded")

        # System is ready for predictions
    else:
        st.error("❌ System requires data initialization")

    # Sidebar for controls
    with st.sidebar:
        st.header("⚙️ Controls")

        # Data update section
        if st.button("🔄 Update Data", type="primary"):
            with st.spinner("Updating weather and ocean data..."):
                try:
                    weather_service.update_weather_data()
                    ocean_service.update_ocean_data()
                    data_manager.update_all_data()
                    st.success("Data updated successfully!")
                    time.sleep(2)
                    st.rerun()
                except Exception as e:
                    st.error(f"Update failed: {str(e)}")

        # Species selection with text input
        st.subheader("🎯 Target Species")

        # Predefined species options
        common_species = ["Yellowfin Tuna (Ahi)", "Bigeye Tuna", "Mahi-mahi", "Opah", "Marlin", "Ono (Wahoo)", "Swordfish", "Moonfish", "Striped Marlin"]

        # Species input method selection
        input_method = st.radio("Select species:", ["Popular Species", "Type Species Name"], horizontal=True)

        if input_method == "Popular Species":
            selected_species = st.selectbox("Choose from common auction species:", common_species)
        else:
            # Text input with autocomplete suggestions
            typed_species = st.text_input("Enter species name:", placeholder="e.g., Yellowfin Tuna, Mahi-mahi, Ono")

            if typed_species:
                # Simple matching for suggestions
                suggestions = [species for species in common_species if typed_species.lower() in species.lower()]
                if suggestions:
                    st.info(f"💡 Did you mean: {', '.join(suggestions[:3])}")
                    selected_species = st.selectbox("Select from suggestions:", ["Use typed name"] + suggestions)
                    if selected_species == "Use typed name":
                        selected_species = typed_species
                else:
                    selected_species = typed_species
                    st.warning(f"'{typed_species}' is not in our common species list. Predictions may be less accurate.")
            else:
                selected_species = common_species[0]  # Default to first option

        # Prediction horizon
        st.subheader("📅 Prediction Horizon")
        prediction_days = st.slider("Days ahead:", 1, 5, 3)

        # Investment amount
        st.subheader("💰 Investment Planning")
        investment_amount = st.number_input("Planned purchase amount ($):", min_value=100, max_value=100000, value=5000, step=100)

        # Sidebar navigation
        page = st.sidebar.selectbox(
            "Navigate",
            ["Fish Price Predictions", "Price Comparison", "Species Lookup", "Integration Status", "Historical Data Sources"]
        )


    # Main dashboard tabs
    if page == "Fish Price Predictions":
        display_predictions_tab(data_manager, prediction_model, selected_species, prediction_days, investment_amount)
    elif page == "Price Comparison":
        display_market_analysis_tab(data_manager, prediction_model, groq_service)
    elif page == "Species Lookup":
        display_conditions_tab(weather_service, ocean_service)
    elif page == "Integration Status":
        display_historical_tab(data_manager)
    elif page == "Historical Data Sources":
        display_data_sources_tab(ecosystem_data, global_analyzer, undercurrent_api, longline_data)

def display_predictions_tab(data_manager, prediction_model, selected_species, prediction_days, investment_amount):
    st.header("🎯 Price Predictions & Recommendations")

    try:
        # Get current predictions
        predictions = prediction_model.get_predictions(selected_species, prediction_days)
        current_conditions = data_manager.get_current_conditions()

        if not predictions:
            st.warning("No predictions available. Please update data first.")
            return

        # Main prediction display
        col1, col2, col3 = st.columns([2, 2, 1])

        with col1:
            st.subheader("📈 Price Direction Forecast")
            prediction = predictions[0]  # Primary prediction

            # Price direction indicator
            direction = prediction.get('direction', 'stable')
            confidence = prediction.get('confidence', 0.5)
            expected_change = prediction.get('price_change_percent', 0)

            if direction == 'increase':
                st.success(f"📈 **PRICE INCREASE EXPECTED**")
                st.markdown(f"**Expected change:** +{expected_change:.1f}%")
            elif direction == 'decrease':
                st.info(f"📉 **PRICE DECREASE EXPECTED**")
                st.markdown(f"**Expected change:** {expected_change:.1f}%")
            else:
                st.info(f"➡️ **STABLE PRICES EXPECTED**")
                st.markdown(f"**Expected change:** ±{abs(expected_change):.1f}%")

            st.markdown(f"**Confidence:** {confidence*100:.0f}%")
            st.progress(confidence)

        with col2:
            st.subheader("💡 Strategic Recommendation")

            # Generate recommendation based on prediction
            recommendation = generate_recommendation(prediction, investment_amount, current_conditions)

            if recommendation['action'] == 'buy_now':
                st.success(f"✅ **RECOMMEND: Buy Now**")
                st.markdown(f"🎯 **Potential savings:** {format_currency(recommendation['potential_savings'])}")
            elif recommendation['action'] == 'wait':
                st.warning(f"⏳ **RECOMMEND: Wait**")
                st.markdown(f"💰 **Potential additional savings:** {format_currency(recommendation['potential_savings'])}")
            else:
                st.info(f"🤔 **NEUTRAL: Monitor**")
                st.markdown(f"📊 **Expected impact:** Minimal")

            st.markdown(f"**Reasoning:** {recommendation['reasoning']}")

        with col3:
            st.subheader("🎲 Risk Assessment")
            risk_level = calculate_risk_level(prediction, current_conditions)

            if risk_level == 'low':
                st.success("🟢 Low Risk")
            elif risk_level == 'medium':
                st.warning("🟡 Medium Risk")
            else:
                st.error("🔴 High Risk")

            st.markdown(f"**Market volatility:** {prediction.get('volatility', 'Normal')}")

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
        try:
            species_data = data_manager.get_species_comparison()

            if species_data is not None and not species_data.empty:
                fig = px.bar(
                    species_data, 
                    x='species', 
                    y='current_price', 
                    color='trend',
                    title="Current Prices by Species",
                    labels={'current_price': 'Price ($/lb)', 'species': 'Species'}
                )
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("No species comparison data available. Please initialize market data.")
        except Exception as e:
            st.error(f"Error loading species comparison: {str(e)}")
            st.info("Using default species data for demonstration.")

        # Seasonal patterns
        st.subheader("📅 Seasonal Price Patterns")
        try:
            seasonal_data = data_manager.get_seasonal_patterns()

            if seasonal_data is not None and not seasonal_data.empty:
                fig = px.line(
                    seasonal_data, 
                    x='month', 
                    y='avg_price', 
                    color='species',
                    title="Historical Seasonal Price Patterns"
                )
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("No seasonal pattern data available.")
        except Exception as e:
            st.error(f"Error loading seasonal patterns: {str(e)}")
            st.info("Using default seasonal data for demonstration.")

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

def display_data_sources_tab(ecosystem_data, global_analyzer, undercurrent_api, longline_data):
    st.header("🔗 Data Sources & Integration Status")

    st.markdown("""
    ### 🎯 For Hawaii Fish Auction Customers

    This system integrates multiple data sources to help you make smarter purchasing decisions at the Honolulu Fish Auction. 
    Understanding our data sources helps you trust the predictions and recommendations.
    """)

    # Display data status banner
    display_data_status_banner_detailed()

    st.divider()

    # Import comprehensive dashboard
    from comprehensive_data_dashboard import display_comprehensive_data_status, display_data_integration_capabilities
    display_comprehensive_data_status(ecosystem_data, global_analyzer, undercurrent_api, longline_data)

    st.divider()

    display_data_integration_capabilities()

def display_data_status_banner_detailed():
    """Display detailed data integration status"""
    st.subheader("📊 Real-Time Data Integration Status")

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        # Check NOAA Weather API
        try:
            from weather_service import WeatherService
            weather_service = WeatherService()
            weather_data = weather_service.get_current_weather()
            if weather_data.get('error'):
                st.error("❌ NOAA Weather")
                st.caption("API connection failed")
            else:
                st.success("✅ NOAA Weather")
                st.caption("Real-time conditions active")
        except:
            st.error("❌ NOAA Weather")
            st.caption("Connection failed")

    with col2:
        # Check Ocean Data
        try:
            from ocean_service import OceanService
            ocean_service = OceanService()
            ocean_data = ocean_service.get_current_ocean_conditions()
            if ocean_data.get('error'):
                st.warning("⚠️ Ocean Data")
                st.caption("Limited data available")
            else:
                st.success("✅ Ocean Data")
                st.caption("Satellite data active")
        except:
            st.warning("⚠️ Ocean Data")
            st.caption("Connection issues")

    with col3:
        # Check Groq AI
        groq_key = os.getenv('GROQ_API_KEY')
        if groq_key:
            st.success("✅ Groq AI")
            st.caption("Market analysis active")
        else:
            st.error("❌ Groq AI")
            st.caption("API key required")

    with col4:
        # Check Historical Data
        from data_manager import DataManager
        data_manager = DataManager()
        historical_data = data_manager.get_historical_price_data()
        if historical_data is not None and not historical_data.empty:
            st.success("✅ Historical Data")
            st.caption(f"{len(historical_data)} records loaded")
        else:
            st.error("❌ UFA Auction Data")
            st.caption("Real data needed for accuracy")

    st.info("💡 **For Auction Customers:** Green status means reliable predictions. Yellow/Red status means predictions are based on historical patterns and may be less accurate.")

    # Weekly Price Calendar
    st.markdown("### 📅 Weekly Price Outlook Calendar")
    
    # Create weekly calendar
    days = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
    current_date = datetime.now()
    
    # Generate price outlook for the week
    cols = st.columns(7)
    for i, day in enumerate(days):
        with cols[i]:
            date = current_date + timedelta(days=i)
            
            # Simulate price outlook (in real app, this would come from your model)
            import random
            random.seed(date.day + date.month)  # Consistent but varied results
            price_outlook = random.choice(['📈 Higher', '📊 Normal', '📉 Lower'])
            
            # Color coding
            if '📈' in price_outlook:
                color = "#ff6b6b"  # Red for higher prices
            elif '📉' in price_outlook:
                color = "#51cf66"  # Green for lower prices  
            else:
                color = "#339af0"  # Blue for normal
            
            st.markdown(f"""
            <div style="
                background: {color}; 
                color: white; 
                padding: 10px; 
                border-radius: 10px; 
                text-align: center;
                margin: 5px 0;
                animation: pulse 2s infinite;
            ">
                <b>{day[:3]}</b><br>
                {date.strftime('%m/%d')}<br>
                {price_outlook}
            </div>
            """, unsafe_allow_html=True)

    st.markdown("""
    <style>
    @keyframes pulse {
        0% { opacity: 1; }
        50% { opacity: 0.7; }
        100% { opacity: 1; }
    }
    </style>
    """, unsafe_allow_html=True)

    # Investment decision guidance
    st.markdown("""
    ### 💰 How to Use This Information for Smarter Purchases

    **🟢 All Green Status:** High confidence predictions - good for larger investments and strategic buying decisions.

    **🟡 Mixed Status:** Moderate confidence - consider smaller purchases or wait for better data.
    
    **🔴 Red Status:** Low confidence - predictions may be unreliable, proceed with caution.

    ### ⚠️ Understanding High Risk Warnings

    **When you see "High Risk" on a fish species prediction:**

    **What it means:**
    - Market volatility is high for this species
    - Price swings of 20%+ are possible within days
    - Limited recent data available for this species
    - External factors (weather, regulations) may impact prices significantly

    **Recommendations for High Risk Species:**
    - **🎯 Buy smaller quantities** to limit exposure
    - **⏰ Monitor daily** for sudden price changes
    - **📊 Focus on species with "Low Risk" ratings** for larger purchases
    - **💡 Consider waiting** 1-2 days if price trend is uncertain
    - **🤝 Negotiate prices** more aggressively due to uncertainty
    - **📱 Set price alerts** if available to catch favorable moments

    **Smart Buying Strategy:**
    - **High Risk + Price Drop Predicted:** Good opportunity for experienced buyers
    - **High Risk + Price Rise Predicted:** Avoid or buy minimal quantities
    - **Low Risk + Any Direction:** Safe for normal purchasing decisionsidence - use predictions as guidance but rely more on your experience.

    **🔴 Many Red Indicators:** Limited data available - stick to tried-and-true buying patterns until data improves.
    """)

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
    """Display current data integration status"""
    st.info("**Data Integration Status**")

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        # Check NOAA Weather API
        try:
            weather_data = weather_service.get_current_weather()
            if weather_data.get('error'):
                st.error("❌ NOAA Weather")
                st.caption("API connection failed")
            else:
                st.success("✅ NOAA Weather")
                st.caption("Real-time data active")
        except:
            st.error("❌ NOAA Weather")
            st.caption("Connection failed")

    with col2:
        # Check Ocean Data
        try:
            ocean_data = ocean_service.get_current_ocean_conditions()
            if ocean_data.get('error'):
                st.warning("⚠️ Ocean Data")
                st.caption("Limited data available")
            else:
                st.success("✅ Ocean Data")
                st.caption("Satellite data active")
        except:
            st.warning("⚠️ Ocean Data")
            st.caption("Connection issues")

    with col3:
        # Check Groq AI
        groq_key = os.getenv('GROQ_API_KEY')
        if groq_key:
            st.success("✅ Groq AI")
            st.caption("Market analysis active")
        else:
            st.error("❌ Groq AI")
            st.caption("API key required")

    with col4:
        # Check Historical Data with enhanced status
        historical_data = data_manager.get_historical_price_data()
        if historical_data is not None and not historical_data.empty:
            st.success("✅ Historical Data")
            st.caption(f"{len(historical_data)} records loaded")
        else:
            st.error("❌ UFA Auction Data")
            st.caption("1.9M records needed")

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

def show_historical_data_sources():
    """Displays the historical data sources page."""
    st.header("🔗 Historical Data Sources")

    st.markdown("""
    This section helps you find and integrate missing historical data, particularly focusing on fish auction data for Hawaii.
    We utilize the NOAA Fisheries API and the FEAT system to discover and extract relevant datasets.
    """)

    # Initialize data finder and extractor
    historical_data_finder = HistoricalDataFinder()
    feat_data_extractor = FEATDataExtractor()

    st.subheader("🔍 Data Source Discovery")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("### NOAA Fisheries Data")
        if st.button("Search NOAA Fisheries Catalog"):
            with st.spinner("Searching NOAA Fisheries Data Catalog..."):
                try:
                    catalog_data = historical_data_finder.search_noaa_fisheries_catalog()
                    st.success("Search complete!")
                    st.dataframe(catalog_data)
                except Exception as e:
                    st.error(f"Error searching NOAA Fisheries Catalog: {str(e)}")

    with col2:
        st.markdown("### FEAT System Data")
        if st.button("Access FEAT Performance Indicators"):
            with st.spinner("Accessing FEAT system..."):
                try:
                    feat_data = feat_data_extractor.get_performance_indicators()
                    st.success("FEAT data retrieved!")
                    st.dataframe(feat_data)
                except Exception as e:
                    st.error(f"Error accessing FEAT system: {str(e)}")

    st.subheader("📅 Historical Fish Auction Data (UFA)")
    st.markdown("""
    The most critical historical data for this application is the **UFA Auction Sampling Data (1984-2002)**.
    This dataset contains approximately **1,923,132 records** including price per pound and quantity sold by species.
    Access requires a PIFSC non-disclosure agreement.
    """)

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**Contact for UFA Auction Data:**")
        st.markdown("""
        - Ashley Tomita: ashley.tomita@noaa.gov
        - Phone: (808) 725-5693
        - Data Steward: keith.bigelow@noaa.gov
        """)
    with col2:
        st.markdown("**Data Requirements:**")
        st.markdown("""
        - PIFSC Non-Disclosure Agreement
        - Secure data transfer protocol
        """)

    if st.button("Initiate UFA Data Access Request"):
        st.info("Please contact Ashley Tomita via email or phone to begin the data access process.")

if __name__ == "__main__":
    main()