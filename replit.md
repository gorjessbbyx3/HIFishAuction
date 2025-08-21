# Hawaii Fish Auction Price Prediction System

## Project Overview
An advanced fish auction price prediction system designed to help buyers at Hawaii fish auctions make strategic purchasing decisions. The system predicts whether fish prices will increase, decrease, or remain stable 2-3 days in advance using weather forecasts, ocean ecosystem data, seasonal patterns, and Groq AI insights.

## Target Users
- Fish buyers at Hawaii auctions who need to decide whether to prepay for fish
- Businesses that rely on predictable fish supply and pricing

## Project Architecture

### Core Components
1. **Data Collection Module** (`data/collectors/`)
   - Weather data from NOAA API
   - Simulated fish catch reports
   - Storm tracking and alerts
   - Ocean temperature and ecosystem indicators

2. **Machine Learning Pipeline** (`ml/`)
   - Scikit-learn regression models
   - Feature engineering for weather, seasonal, and catch data
   - Storm impact logic for supply disruption prediction
   - Groq AI integration for insights and analysis

3. **Database Layer** (`database/`)
   - SQLite storage for all collected data
   - Daily update mechanisms
   - Historical data preservation

4. **Frontend Dashboard** (`app.py`)
   - Streamlit-based interactive interface
   - Plotly charts for predictions and trends
   - Key indicators and alerts
   - Storm impact warnings

### Key Features
- **Storm Impact Analysis**: Flags when weather conditions (>20 knots wind, storm advisories) prevent fishing
- **Supply-Price Correlation**: Adjusts predictions based on expected supply disruptions
- **Multi-day Forecasting**: 2-3 day advance price predictions
- **Visual Dashboard**: Clear indicators for buying decisions
- **Real Data Ready**: Structured to integrate actual Honolulu Fish Auction data

## Technical Stack
- **Backend**: Python with Streamlit
- **ML**: Scikit-learn, pandas, numpy
- **Database**: SQLite
- **Visualization**: Plotly, matplotlib
- **APIs**: NOAA Weather, Groq AI
- **Deployment**: Replit-ready configuration

## User Preferences
- Focus on practical buying decisions
- Clear visual indicators for price trends
- Storm impact prominently displayed
- Ready for real data integration

## Recent Changes
- **2025-08-21**: Initial project setup with comprehensive architecture
- **2025-08-21**: Removed all hardcoded/synthetic data, implemented authentic APIs only
- **2025-08-21**: Integrated NOAA Weather Service API and CoastWatch ocean data
- **2025-08-21**: Enhanced Groq AI service with advanced knowledge base and analytics
- **2025-08-21**: Added comprehensive NOAA InPort data integration (UFA Auction 1984-2002)
- **2025-08-21**: Integrated Undercurrent News global market data capabilities
- **2025-08-21**: Upgraded Groq knowledge base with species-specific behavioral patterns
- **2025-08-21**: Enhanced AI analytics with environmental suitability scoring
- **2025-08-21**: Added sophisticated market pressure analysis and risk assessment
- **2025-08-21**: Created detailed data integration status dashboard