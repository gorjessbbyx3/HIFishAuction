from typing import Dict, Any, List
import pandas as pd
import numpy as np

def format_currency(amount: float) -> str:
    """Format currency values"""
    if amount is None:
        return "$0.00"
    return f"${amount:,.2f}"

def get_confidence_color(confidence: float) -> str:
    """Get color based on confidence level"""
    if confidence >= 0.8:
        return "#28a745"  # Green
    elif confidence >= 0.6:
        return "#ffc107"  # Yellow
    else:
        return "#dc3545"  # Red

def get_recommendation_icon(action: str) -> str:
    """Get icon for recommendation action"""
    icons = {
        'buy_now': '✅',
        'wait': '⏳',
        'monitor': '🤔'
    }
    return icons.get(action, '📊')

def get_species_emoji(species: str) -> str:
    """Get emoji representation for fish species"""
    species_lower = species.lower()
    
    if 'tuna' in species_lower or 'ahi' in species_lower:
        return '🐟'
    elif 'mahi' in species_lower:
        return '🐠'
    elif 'marlin' in species_lower:
        return '🗡️'
    elif 'opah' in species_lower:
        return '🌕'
    else:
        return '🐟'

def calculate_price_change_color(change_percent: float) -> str:
    """Get color for price change percentage"""
    if change_percent > 5:
        return "#28a745"  # Strong positive - green
    elif change_percent > 0:
        return "#20c997"  # Weak positive - light green
    elif change_percent > -5:
        return "#ffc107"  # Neutral/small negative - yellow
    else:
        return "#dc3545"  # Strong negative - red

def format_percentage(value: float, include_sign: bool = True) -> str:
    """Format percentage value"""
    if value is None:
        return "0.0%"
    
    sign = "+" if value > 0 and include_sign else ""
    return f"{sign}{value:.1f}%"

def calculate_potential_savings(investment_amount: float, price_change_percent: float) -> float:
    """Calculate potential savings from price change"""
    if investment_amount <= 0 or price_change_percent == 0:
        return 0.0
    
    return investment_amount * (abs(price_change_percent) / 100)

def get_weather_severity_level(wind_speed: float, wave_height: float, storm_warnings: List[str]) -> Dict[str, Any]:
    """Determine weather severity level and description"""
    severity_score = 0
    factors = []
    
    # Storm warnings impact
    if storm_warnings:
        severity_score += len(storm_warnings) * 3
        factors.append(f"{len(storm_warnings)} storm warning(s)")
    
    # Wind impact
    if wind_speed > 30:
        severity_score += 4
        factors.append(f"extreme winds ({wind_speed:.1f} knots)")
    elif wind_speed > 25:
        severity_score += 3
        factors.append(f"very high winds ({wind_speed:.1f} knots)")
    elif wind_speed > 20:
        severity_score += 2
        factors.append(f"high winds ({wind_speed:.1f} knots)")
    elif wind_speed > 15:
        severity_score += 1
        factors.append(f"moderate winds ({wind_speed:.1f} knots)")
    
    # Wave impact
    if wave_height > 12:
        severity_score += 4
        factors.append(f"extreme waves ({wave_height:.1f} ft)")
    elif wave_height > 8:
        severity_score += 3
        factors.append(f"very high waves ({wave_height:.1f} ft)")
    elif wave_height > 6:
        severity_score += 2
        factors.append(f"high waves ({wave_height:.1f} ft)")
    elif wave_height > 4:
        severity_score += 1
        factors.append(f"moderate waves ({wave_height:.1f} ft)")
    
    # Determine severity level
    if severity_score >= 8:
        level = "extreme"
        color = "#8b0000"  # Dark red
        description = "Extremely dangerous conditions - all fishing suspended"
    elif severity_score >= 6:
        level = "severe"
        color = "#dc3545"  # Red
        description = "Severe weather - no fishing activity expected"
    elif severity_score >= 4:
        level = "high"
        color = "#fd7e14"  # Orange
        description = "High impact weather - significant fishing disruption"
    elif severity_score >= 2:
        level = "moderate"
        color = "#ffc107"  # Yellow
        description = "Moderate weather - some fishing disruption possible"
    elif severity_score >= 1:
        level = "low"
        color = "#20c997"  # Light green
        description = "Minor weather impact - fishing mostly normal"
    else:
        level = "minimal"
        color = "#28a745"  # Green
        description = "Good weather conditions - optimal for fishing"
    
    return {
        'level': level,
        'score': severity_score,
        'color': color,
        'description': description,
        'factors': factors
    }

def calculate_fishing_productivity_index(sst: float, chlorophyll: float, wind_speed: float, 
                                       species: str = "yellowfin_tuna") -> Dict[str, Any]:
    """Calculate fishing productivity index based on environmental conditions"""
    
    # Species-specific optimal ranges
    species_preferences = {
        "yellowfin_tuna": {"sst_range": [24, 28], "wind_tolerance": 20},
        "bigeye_tuna": {"sst_range": [18, 24], "wind_tolerance": 18},
        "mahi_mahi": {"sst_range": [26, 30], "wind_tolerance": 25},
        "opah": {"sst_range": [20, 26], "wind_tolerance": 15},
        "marlin": {"sst_range": [25, 29], "wind_tolerance": 22}
    }
    
    preferences = species_preferences.get(species, species_preferences["yellowfin_tuna"])
    
    productivity_score = 0
    factors = []
    
    # Temperature score (40% weight)
    sst_range = preferences["sst_range"]
    if sst_range[0] <= sst <= sst_range[1]:
        temp_score = 40
        factors.append(f"optimal temperature ({sst:.1f}°C)")
    elif sst_range[0] - 2 <= sst <= sst_range[1] + 2:
        temp_score = 25
        factors.append(f"acceptable temperature ({sst:.1f}°C)")
    else:
        temp_score = 10
        factors.append(f"suboptimal temperature ({sst:.1f}°C)")
    
    productivity_score += temp_score
    
    # Food availability score (30% weight)
    if chlorophyll > 0.25:
        food_score = 30
        factors.append(f"excellent food availability ({chlorophyll:.3f} mg/m³)")
    elif chlorophyll > 0.15:
        food_score = 25
        factors.append(f"good food availability ({chlorophyll:.3f} mg/m³)")
    elif chlorophyll > 0.1:
        food_score = 15
        factors.append(f"moderate food availability ({chlorophyll:.3f} mg/m³)")
    else:
        food_score = 5
        factors.append(f"low food availability ({chlorophyll:.3f} mg/m³)")
    
    productivity_score += food_score
    
    # Weather impact score (30% weight)
    wind_tolerance = preferences["wind_tolerance"]
    if wind_speed <= wind_tolerance * 0.7:
        weather_score = 30
        factors.append(f"excellent fishing conditions ({wind_speed:.1f} knots)")
    elif wind_speed <= wind_tolerance:
        weather_score = 20
        factors.append(f"good fishing conditions ({wind_speed:.1f} knots)")
    elif wind_speed <= wind_tolerance * 1.3:
        weather_score = 10
        factors.append(f"challenging fishing conditions ({wind_speed:.1f} knots)")
    else:
        weather_score = 0
        factors.append(f"unfishable conditions ({wind_speed:.1f} knots)")
    
    productivity_score += weather_score
    
    # Normalize to 0-100 scale
    productivity_score = min(max(productivity_score, 0), 100)
    
    # Determine productivity level
    if productivity_score >= 80:
        level = "excellent"
        color = "#28a745"
        description = "Excellent fishing conditions - high catch probability"
    elif productivity_score >= 60:
        level = "good"
        color = "#20c997"
        description = "Good fishing conditions - normal catch rates expected"
    elif productivity_score >= 40:
        level = "fair"
        color = "#ffc107"
        description = "Fair fishing conditions - below average catch rates"
    elif productivity_score >= 20:
        level = "poor"
        color = "#fd7e14"
        description = "Poor fishing conditions - low catch probability"
    else:
        level = "very_poor"
        color = "#dc3545"
        description = "Very poor fishing conditions - minimal catch expected"
    
    return {
        'score': productivity_score,
        'level': level,
        'color': color,
        'description': description,
        'factors': factors,
        'species': species
    }

def validate_environmental_data(data: Dict) -> Dict[str, Any]:
    """Validate environmental data and flag potential issues"""
    issues = []
    warnings = []

    # Check for missing critical data
    critical_fields = ['wind_speed', 'sea_surface_temp', 'chlorophyll']
    for field in critical_fields:
        if field not in data or data[field] is None:
            issues.append(f"Missing {field} data")

    # Check for extreme values that might indicate data quality issues
    if 'wind_speed' in data and data['wind_speed'] is not None:
        wind_speed = data['wind_speed']
        if wind_speed > 100:
            issues.append(f"Extreme wind speed: {wind_speed} knots")
        elif wind_speed > 50:
            warnings.append(f"Very high wind speed: {wind_speed} knots")
        elif wind_speed < 0:
            issues.append(f"Invalid wind speed: {wind_speed} knots")

    if 'sea_surface_temp' in data and data['sea_surface_temp'] is not None:
        sst = data['sea_surface_temp']
        if sst > 35 or sst < 10:
            issues.append(f"Extreme sea surface temperature: {sst}°C")
        elif sst > 32 or sst < 15:
            warnings.append(f"Unusual sea surface temperature: {sst}°C")

    if 'chlorophyll' in data and data['chlorophyll'] is not None:
        chlor = data['chlorophyll']
        if chlor > 10 or chlor < 0:
            issues.append(f"Invalid chlorophyll level: {chlor} mg/m³")
        elif chlor > 5:
            warnings.append(f"Very high chlorophyll level: {chlor} mg/m³")

    if 'wave_height' in data and data['wave_height'] is not None:
        wave_height = data['wave_height']
        if wave_height > 50 or wave_height < 0:
            issues.append(f"Invalid wave height: {wave_height} ft")
        elif wave_height > 30:
            warnings.append(f"Extreme wave height: {wave_height} ft")

    # Determine overall data quality
    if issues:
        quality = "poor"
        quality_color = "#dc3545"
    elif warnings:
        quality = "questionable" 
        quality_color = "#ffc107"
    else:
        quality = "good"
        quality_color = "#28a745"

    return {
        'quality': quality,
        'quality_color': quality_color,
        'issues': issues,
        'warnings': warnings,
        'total_problems': len(issues) + len(warnings)
    }

def calculate_market_volatility(price_history: List[float], window: int = 7) -> Dict[str, float]:
    """Calculate market volatility metrics"""
    if len(price_history) < 2:
        return {'volatility': 0.0, 'volatility_percent': 0.0, 'trend': 'stable'}
    
    # Convert to numpy array for calculations
    prices = np.array(price_history)
    
    # Calculate returns
    returns = np.diff(prices) / prices[:-1]
    
    # Calculate volatility (standard deviation of returns)
    volatility = np.std(returns) if len(returns) > 1 else 0.0
    volatility_percent = volatility * 100
    
    # Calculate trend
    if len(prices) >= window:
        recent_avg = np.mean(prices[-window:])
        earlier_avg = np.mean(prices[-window*2:-window]) if len(prices) >= window*2 else np.mean(prices[:-window])
        
        if recent_avg > earlier_avg * 1.02:
            trend = 'increasing'
        elif recent_avg < earlier_avg * 0.98:
            trend = 'decreasing'
        else:
            trend = 'stable'
    else:
        trend = 'insufficient_data'
    
    return {
        'volatility': round(volatility, 4),
        'volatility_percent': round(volatility_percent, 2),
        'trend': trend,
        'price_range': {
            'min': float(np.min(prices)),
            'max': float(np.max(prices)),
            'mean': float(np.mean(prices))
        }
    }

def format_time_ago(timestamp: str) -> str:
    """Format timestamp as 'time ago' string"""
    try:
        if isinstance(timestamp, str):
            dt = datetime.fromisoformat(timestamp.replace('Z', '+00:00'))
        else:
            dt = timestamp
        
        now = datetime.now()
        diff = now - dt.replace(tzinfo=None)
        
        if diff.days > 0:
            return f"{diff.days} day{'s' if diff.days != 1 else ''} ago"
        elif diff.seconds > 3600:
            hours = diff.seconds // 3600
            return f"{hours} hour{'s' if hours != 1 else ''} ago"
        elif diff.seconds > 60:
            minutes = diff.seconds // 60
            return f"{minutes} minute{'s' if minutes != 1 else ''} ago"
        else:
            return "just now"
    except:
        return "unknown"

def generate_summary_statistics(data: pd.DataFrame, target_column: str) -> Dict[str, Any]:
    """Generate summary statistics for a dataset"""
    if data.empty or target_column not in data.columns:
        return {}
    
    series = data[target_column].dropna()
    
    if len(series) == 0:
        return {}
    
    return {
        'count': len(series),
        'mean': round(float(series.mean()), 3),
        'median': round(float(series.median()), 3),
        'std': round(float(series.std()), 3),
        'min': round(float(series.min()), 3),
        'max': round(float(series.max()), 3),
        'q25': round(float(series.quantile(0.25)), 3),
        'q75': round(float(series.quantile(0.75)), 3),
        'range': round(float(series.max() - series.min()), 3),
        'coefficient_of_variation': round(float(series.std() / series.mean()) if series.mean() != 0 else 0, 3)
    }

def clean_species_name(species: str) -> str:
    """Clean and standardize species name for display"""
    if not species:
        return "Unknown Species"
    
    # Replace underscores with spaces
    cleaned = species.replace('_', ' ')
    
    # Title case
    cleaned = cleaned.title()
    
    # Special cases for fish names
    replacements = {
        'Yellowfin Tuna': 'Yellowfin Tuna (Ahi)',
        'Bigeye Tuna': 'Bigeye Tuna',
        'Mahi Mahi': 'Mahi-mahi (Dorado)',
        'Opah': 'Opah (Moonfish)',
        'Marlin': 'Blue Marlin'
    }
    
    return replacements.get(cleaned, cleaned)

def calculate_confidence_interval(predictions: List[float], confidence_level: float = 0.95) -> Dict[str, float]:
    """Calculate confidence interval for predictions"""
    if len(predictions) < 2:
        return {'lower': 0, 'upper': 0, 'width': 0}
    
    predictions_array = np.array(predictions)
    mean = np.mean(predictions_array)
    std = np.std(predictions_array, ddof=1)
    
    # Calculate confidence interval
    from scipy import stats
    alpha = 1 - confidence_level
    degrees_freedom = len(predictions) - 1
    t_score = stats.t.ppf(1 - alpha/2, degrees_freedom)
    
    margin_of_error = t_score * (std / np.sqrt(len(predictions)))
    
    return {
        'lower': round(mean - margin_of_error, 3),
        'upper': round(mean + margin_of_error, 3),
        'width': round(2 * margin_of_error, 3),
        'mean': round(mean, 3)
    }

def get_seasonal_context(date: str, species: str) -> Dict[str, Any]:
    """Get seasonal context for a specific date and species"""
    try:
        target_date = datetime.strptime(date, '%Y-%m-%d')
        month = target_date.month
        
        # Seasonal periods for Hawaii
        seasons = {
            'winter': [12, 1, 2],
            'spring': [3, 4, 5], 
            'summer': [6, 7, 8],
            'fall': [9, 10, 11]
        }
        
        current_season = next((season for season, months in seasons.items() if month in months), 'unknown')
        
        # Species-specific seasonal information
        species_seasons = {
            'yellowfin_tuna': {
                'peak': 'summer',
                'low': 'winter',
                'description': 'Yellowfin tuna are most abundant during summer months (June-September) when water temperatures are optimal.'
            },
            'bigeye_tuna': {
                'peak': 'spring_fall',
                'low': 'summer',
                'description': 'Bigeye tuna prefer cooler waters and are more common during spring and fall months.'
            },
            'mahi_mahi': {
                'peak': 'spring',
                'low': 'winter',
                'description': 'Mahi-mahi are most abundant in spring (March-May) and early fall.'
            },
            'opah': {
                'peak': 'winter_spring',
                'low': 'summer',
                'description': 'Opah are typically more available during cooler months.'
            },
            'marlin': {
                'peak': 'summer',
                'low': 'winter',
                'description': 'Blue marlin are most active during warm summer months.'
            }
        }
        
        species_info = species_seasons.get(species, {})
        
        return {
            'current_season': current_season,
            'month': month,
            'species_peak_season': species_info.get('peak', 'unknown'),
            'species_low_season': species_info.get('low', 'unknown'),
            'seasonal_description': species_info.get('description', 'No seasonal information available.'),
            'is_peak_season': current_season in species_info.get('peak', ''),
            'is_low_season': current_season in species_info.get('low', '')
        }
        
    except Exception as e:
        return {
            'current_season': 'unknown',
            'error': str(e)
        }

# Utility constants
WEATHER_SEVERITY_THRESHOLDS = {
    'wind_speed': {'moderate': 15, 'high': 20, 'severe': 25, 'extreme': 30},
    'wave_height': {'moderate': 4, 'high': 6, 'severe': 8, 'extreme': 12},
    'temperature': {'cold': 20, 'cool': 24, 'optimal_min': 25, 'optimal_max': 29, 'warm': 32}
}

SPECIES_CHARACTERISTICS = {
    'yellowfin_tuna': {'emoji': '🐟', 'color': '#FF6B6B', 'typical_size': 'Large'},
    'bigeye_tuna': {'emoji': '🐟', 'color': '#4ECDC4', 'typical_size': 'Large'},
    'mahi_mahi': {'emoji': '🐠', 'color': '#45B7D1', 'typical_size': 'Medium'},
    'opah': {'emoji': '🌕', 'color': '#96CEB4', 'typical_size': 'Large'},
    'marlin': {'emoji': '🗡️', 'color': '#FFEAA7', 'typical_size': 'Extra Large'}
}

CONFIDENCE_LEVELS = {
    'very_high': {'min': 0.9, 'color': '#28a745', 'description': 'Very High Confidence'},
    'high': {'min': 0.8, 'color': '#20c997', 'description': 'High Confidence'},
    'medium': {'min': 0.6, 'color': '#ffc107', 'description': 'Medium Confidence'},
    'low': {'min': 0.4, 'color': '#fd7e14', 'description': 'Low Confidence'},
    'very_low': {'min': 0.0, 'color': '#dc3545', 'description': 'Very Low Confidence'}
}