import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score, mean_absolute_error
import pickle
import os
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
import warnings
warnings.filterwarnings('ignore')

class PredictionModel:
    """Machine learning model for predicting fish auction prices"""
    
    def __init__(self):
        self.price_regressor = RandomForestRegressor(n_estimators=100, random_state=42)
        self.direction_classifier = RandomForestClassifier(n_estimators=100, random_state=42)
        self.scaler = StandardScaler()
        self.label_encoder = LabelEncoder()
        
        self.is_trained = False
        self.feature_columns = []
        self.model_path = "models/"
        
        # Create models directory if it doesn't exist
        if not os.path.exists(self.model_path):
            os.makedirs(self.model_path)
        
        # Try to load existing models
        self.load_models()
        
        # If no models exist, train with synthetic data
        if not self.is_trained:
            self.train_initial_model()
    
    def create_features(self, weather_data: Dict, ocean_data: Dict, 
                       species: str, date: str) -> np.ndarray:
        """Create feature vector from input data"""
        try:
            from data_manager import DataManager
            data_manager = DataManager()
            
            # Parse date
            target_date = datetime.strptime(date, '%Y-%m-%d')
            month = target_date.month
            day_of_year = target_date.timetuple().tm_yday
            
            # Weather features
            wind_speed = weather_data.get('wind_speed', 12.0)
            wave_height = weather_data.get('wave_height', 3.5)
            storm_warnings = len(weather_data.get('storm_warnings', []))
            temperature = weather_data.get('temperature', 26.5)
            
            # Ocean features
            sst = ocean_data.get('sea_surface_temp', 26.5)
            chlorophyll = ocean_data.get('chlorophyll', 0.15)
            current_strength = ocean_data.get('current_strength', 'Moderate')
            upwelling_index = ocean_data.get('upwelling_index', 0.4)
            
            # Encode categorical variables
            current_strength_encoded = {'Weak': 0, 'Moderate': 1, 'Strong': 2}.get(current_strength, 1)
            
            # Species-specific features
            species_data = data_manager.species_data.get(species, data_manager.species_data.get('yellowfin_tuna', {}))
            species_volatility = species_data.get('volatility', 0.15)
            species_wind_sensitivity = species_data.get('wind_sensitivity', 0.8)
            species_storm_impact = species_data.get('storm_impact', 0.9)
            
            # Seasonal features
            seasonal_patterns = data_manager.seasonal_patterns.get(species, {})
            seasonal_multiplier = seasonal_patterns.get('price_multipliers', {}).get(month, 1.0)
            
            # Calculate derived features
            storm_impact_score = storm_warnings * species_storm_impact
            wind_impact_score = max(0, (wind_speed - 15) / 10) * species_wind_sensitivity
            
            # Temperature deviation from optimal
            optimal_sst_range = species_data.get('optimal_sst_range', [24, 28])
            sst_deviation = 0
            if sst < optimal_sst_range[0]:
                sst_deviation = optimal_sst_range[0] - sst
            elif sst > optimal_sst_range[1]:
                sst_deviation = sst - optimal_sst_range[1]
            
            # Fishing condition score
            fishing_disruption = self._calculate_fishing_disruption(wind_speed, wave_height, storm_warnings)
            
            # Seasonal cyclical features
            month_sin = np.sin(2 * np.pi * month / 12)
            month_cos = np.cos(2 * np.pi * month / 12)
            
            # Feature vector
            features = np.array([
                # Weather features
                wind_speed,
                wave_height, 
                storm_warnings,
                temperature,
                
                # Ocean features
                sst,
                chlorophyll,
                current_strength_encoded,
                upwelling_index,
                
                # Derived features
                storm_impact_score,
                wind_impact_score,
                sst_deviation,
                fishing_disruption,
                
                # Seasonal features
                seasonal_multiplier,
                month_sin,
                month_cos,
                
                # Species characteristics
                species_volatility,
                species_wind_sensitivity,
                species_storm_impact
            ])
            
            return features.reshape(1, -1)
            
        except Exception as e:
            print(f"Error creating features: {str(e)}")
            # Return default feature vector
            return np.zeros((1, 18))
    
    def _calculate_fishing_disruption(self, wind_speed: float, wave_height: float, 
                                    storm_warnings: int) -> float:
        """Calculate fishing disruption score (0-1)"""
        disruption = 0.0
        
        # Wind impact
        if wind_speed > 25:
            disruption += 0.8
        elif wind_speed > 20:
            disruption += 0.5
        elif wind_speed > 15:
            disruption += 0.2
        
        # Wave impact
        if wave_height > 8:
            disruption += 0.6
        elif wave_height > 6:
            disruption += 0.3
        elif wave_height > 4:
            disruption += 0.1
        
        # Storm impact
        if storm_warnings > 0:
            disruption += 0.9
        
        return min(disruption, 1.0)
    
    def load_historical_auction_data(self) -> Tuple[pd.DataFrame, pd.Series, pd.Series]:
        """Load historical auction data from available sources"""
        try:
            # Try to load from database first
            from data_manager import DataManager
            data_manager = DataManager()
            
            historical_data = data_manager.get_historical_price_data()
            if historical_data is not None and not historical_data.empty:
                print("Using existing historical data for training")
                return self._prepare_historical_data_for_training(historical_data)
            
            # If no historical data, return minimal training set
            print("No historical auction data available. Model training requires real data.")
            print("Please provide historical fish auction data or API access.")
            return pd.DataFrame(), pd.Series(), pd.Series()
            
        except Exception as e:
            print(f"Error loading historical data: {str(e)}")
            return pd.DataFrame(), pd.Series(), pd.Series()
    
    def _prepare_historical_data_for_training(self, historical_data: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series, pd.Series]:
        """Prepare historical data for model training"""
        if len(historical_data) < 10:
            print("Insufficient historical data for training")
            return pd.DataFrame(), pd.Series(), pd.Series()
        
        # This would process real auction data with environmental conditions
        # For now, return empty until real data is provided
        print("Historical data processing requires real environmental correlation data")
        return pd.DataFrame(), pd.Series(), pd.Series()
    
    def train_initial_model(self):
        """Train initial model with available data"""
        try:
            print("Loading historical auction data for model training...")
            
            # Load real historical data
            X, y_price, y_direction = self.load_historical_auction_data()
            
            if X.empty:
                print("No training data available. Model cannot be trained without historical auction data.")
                self.is_trained = False
                return
            
            # Define feature columns
            self.feature_columns = [
                'wind_speed', 'wave_height', 'storm_warnings', 'temperature',
                'sst', 'chlorophyll', 'current_strength_encoded', 'upwelling_index',
                'storm_impact_score', 'wind_impact_score', 'sst_deviation', 'fishing_disruption',
                'seasonal_multiplier', 'month_sin', 'month_cos',
                'species_volatility', 'species_wind_sensitivity', 'species_storm_impact'
            ]
            
            X.columns = self.feature_columns
            
            # Split data
            X_train, X_test, y_price_train, y_price_test, y_dir_train, y_dir_test = train_test_split(
                X, y_price, y_direction, test_size=0.2, random_state=42
            )
            
            # Scale features
            X_train_scaled = self.scaler.fit_transform(X_train)
            X_test_scaled = self.scaler.transform(X_test)
            
            # Encode direction labels
            y_dir_train_encoded = self.label_encoder.fit_transform(y_dir_train)
            y_dir_test_encoded = self.label_encoder.transform(y_dir_test)
            
            # Train models
            self.price_regressor.fit(X_train_scaled, y_price_train)
            self.direction_classifier.fit(X_train_scaled, y_dir_train_encoded)
            
            # Evaluate models
            price_pred = self.price_regressor.predict(X_test_scaled)
            direction_pred = self.direction_classifier.predict(X_test_scaled)
            
            price_mae = mean_absolute_error(y_price_test, price_pred)
            direction_accuracy = accuracy_score(y_dir_test_encoded, direction_pred)
            
            print(f"Initial model training complete:")
            print(f"Price prediction MAE: ${price_mae:.2f}")
            print(f"Direction prediction accuracy: {direction_accuracy:.3f}")
            
            self.is_trained = True
            self.save_models()
            
        except Exception as e:
            print(f"Error training initial model: {str(e)}")
    
    def predict_price_direction(self, weather_data: Dict, ocean_data: Dict, 
                              species: str, target_date: str) -> Dict:
        """Predict price direction and magnitude"""
        try:
            if not self.is_trained:
                self.train_initial_model()
            
            # Create features
            features = self.create_features(weather_data, ocean_data, species, target_date)
            
            if len(self.feature_columns) != features.shape[1]:
                print(f"Feature dimension mismatch: expected {len(self.feature_columns)}, got {features.shape[1]}")
                return self._get_fallback_prediction()
            
            # Scale features
            features_scaled = self.scaler.transform(features)
            
            # Make predictions
            predicted_price = self.price_regressor.predict(features_scaled)[0]
            direction_proba = self.direction_classifier.predict_proba(features_scaled)[0]
            direction_pred = self.direction_classifier.predict(features_scaled)[0]
            
            # Get direction label
            direction_label = self.label_encoder.inverse_transform([direction_pred])[0]
            
            # Calculate confidence
            confidence = np.max(direction_proba)
            
            # Get current base price for species
            from data_manager import DataManager
            data_manager = DataManager()
            species_data = data_manager.species_data.get(species, data_manager.species_data.get('yellowfin_tuna', {}))
            base_price = species_data.get('base_price', 12.50)
            
            # Calculate percentage change
            price_change_percent = ((predicted_price - base_price) / base_price) * 100
            
            # Generate key factors
            key_factors = self._identify_key_factors(weather_data, ocean_data, species)
            
            prediction = {
                'species': species,
                'target_date': target_date,
                'predicted_price': round(predicted_price, 2),
                'current_base_price': base_price,
                'direction': direction_label,
                'price_change_percent': round(price_change_percent, 1),
                'confidence': round(confidence, 3),
                'direction_probabilities': {
                    label: round(prob, 3) for label, prob in 
                    zip(self.label_encoder.classes_, direction_proba)
                },
                'key_factors': key_factors,
                'volatility': species_data.get('volatility', 0.15),
                'volatility_score': min(abs(price_change_percent) / 20, 1.0)  # Normalized volatility
            }
            
            return prediction
            
        except Exception as e:
            print(f"Error making prediction: {str(e)}")
            return self._get_fallback_prediction()
    
    def _identify_key_factors(self, weather_data: Dict, ocean_data: Dict, species: str) -> List[Dict]:
        """Identify key factors influencing the prediction"""
        factors = []
        
        # Weather factors
        wind_speed = weather_data.get('wind_speed', 0)
        storm_warnings = weather_data.get('storm_warnings', [])
        wave_height = weather_data.get('wave_height', 0)
        
        if storm_warnings:
            factors.append({
                'name': 'Storm Warnings',
                'description': f"{len(storm_warnings)} active storm warning(s) - fishing disruption expected",
                'impact': 0.8,
                'type': 'weather'
            })
        
        if wind_speed > 20:
            factors.append({
                'name': 'High Winds',
                'description': f"Wind speed {wind_speed:.1f} knots - boats may stay in port",
                'impact': 0.6,
                'type': 'weather'
            })
        
        if wave_height > 6:
            factors.append({
                'name': 'High Waves',
                'description': f"Wave height {wave_height:.1f} ft - difficult fishing conditions",
                'impact': 0.4,
                'type': 'weather'
            })
        
        # Ocean factors
        sst = ocean_data.get('sea_surface_temp', 26.5)
        chlorophyll = ocean_data.get('chlorophyll', 0.15)
        
        from data_manager import DataManager
        data_manager = DataManager()
        species_data = data_manager.species_data.get(species, {})
        optimal_sst_range = species_data.get('optimal_sst_range', [24, 28])
        
        if optimal_sst_range[0] <= sst <= optimal_sst_range[1]:
            factors.append({
                'name': 'Optimal Water Temperature',
                'description': f"SST {sst:.1f}°C is optimal for {species.replace('_', ' ')}",
                'impact': -0.3,
                'type': 'ocean'
            })
        else:
            factors.append({
                'name': 'Suboptimal Water Temperature',
                'description': f"SST {sst:.1f}°C outside optimal range - fish may be harder to find",
                'impact': 0.4,
                'type': 'ocean'
            })
        
        if chlorophyll > 0.2:
            factors.append({
                'name': 'High Food Availability',
                'description': f"High chlorophyll levels ({chlorophyll:.3f} mg/m³) indicate good feeding conditions",
                'impact': -0.2,
                'type': 'ocean'
            })
        elif chlorophyll < 0.1:
            factors.append({
                'name': 'Low Food Availability',
                'description': f"Low chlorophyll levels ({chlorophyll:.3f} mg/m³) may affect fish distribution",
                'impact': 0.3,
                'type': 'ocean'
            })
        
        # Seasonal factors
        current_month = datetime.now().month
        seasonal_patterns = data_manager.seasonal_patterns.get(species, {})
        peak_months = seasonal_patterns.get('peak_months', [])
        low_months = seasonal_patterns.get('low_months', [])
        
        if current_month in peak_months:
            factors.append({
                'name': 'Peak Season',
                'description': f"Currently in peak season for {species.replace('_', ' ')} - high supply expected",
                'impact': -0.4,
                'type': 'seasonal'
            })
        elif current_month in low_months:
            factors.append({
                'name': 'Low Season',
                'description': f"Currently in low season for {species.replace('_', ' ')} - reduced supply expected",
                'impact': 0.5,
                'type': 'seasonal'
            })
        
        # Sort by absolute impact
        factors.sort(key=lambda x: abs(x['impact']), reverse=True)
        
        return factors[:5]  # Return top 5 factors
    
    def _get_fallback_prediction(self) -> Dict:
        """Return message indicating need for real data"""
        return {
            'species': 'unknown',
            'target_date': datetime.now().strftime('%Y-%m-%d'),
            'predicted_price': 0.0,
            'current_base_price': 0.0,
            'direction': 'insufficient_data',
            'price_change_percent': 0.0,
            'confidence': 0.0,
            'direction_probabilities': {'insufficient_data': 1.0},
            'key_factors': [
                {'name': 'Data Required', 'description': 'Historical auction data needed for predictions', 'impact': 0}
            ],
            'volatility': 0.0,
            'volatility_score': 0.0,
            'error_message': 'Model requires real historical auction data to make predictions'
        }
    
    def get_predictions(self, selected_species: str, prediction_days: int) -> List[Dict]:
        """Get predictions for multiple days ahead"""
        predictions = []
        
        # Get current conditions
        from weather_service import WeatherService
        from ocean_service import OceanService
        
        weather_service = WeatherService()
        ocean_service = OceanService()
        
        # Map display names to internal species names
        species_mapping = {
            "All Species": "yellowfin_tuna",
            "Yellowfin Tuna (Ahi)": "yellowfin_tuna",
            "Bigeye Tuna": "bigeye_tuna",
            "Mahi-mahi": "mahi_mahi",
            "Opah": "opah",
            "Marlin": "marlin"
        }
        
        internal_species = species_mapping.get(selected_species, "yellowfin_tuna")
        
        try:
            # Get weather and ocean forecasts
            weather_forecast = weather_service.get_forecast(prediction_days)
            ocean_forecast = ocean_service.get_ocean_forecast(prediction_days)
            
            for i in range(prediction_days):
                target_date = (datetime.now() + timedelta(days=i+1)).strftime('%Y-%m-%d')
                
                # Get conditions for this day
                if i < len(weather_forecast):
                    weather_data = weather_forecast[i]
                else:
                    weather_data = weather_service.get_current_weather()
                
                if i < len(ocean_forecast):
                    ocean_data = ocean_forecast[i]
                else:
                    ocean_data = ocean_service.get_current_ocean_conditions()
                
                # Make prediction
                prediction = self.predict_price_direction(weather_data, ocean_data, internal_species, target_date)
                predictions.append(prediction)
            
        except Exception as e:
            print(f"Error getting predictions: {str(e)}")
            # Return fallback predictions
            for i in range(prediction_days):
                target_date = (datetime.now() + timedelta(days=i+1)).strftime('%Y-%m-%d')
                prediction = self._get_fallback_prediction()
                prediction['target_date'] = target_date
                predictions.append(prediction)
        
        return predictions
    
    def save_models(self):
        """Save trained models to disk"""
        try:
            pickle.dump(self.price_regressor, open(f"{self.model_path}price_regressor.pkl", "wb"))
            pickle.dump(self.direction_classifier, open(f"{self.model_path}direction_classifier.pkl", "wb"))
            pickle.dump(self.scaler, open(f"{self.model_path}scaler.pkl", "wb"))
            pickle.dump(self.label_encoder, open(f"{self.model_path}label_encoder.pkl", "wb"))
            pickle.dump(self.feature_columns, open(f"{self.model_path}feature_columns.pkl", "wb"))
            print("Models saved successfully")
        except Exception as e:
            print(f"Error saving models: {str(e)}")
    
    def load_models(self):
        """Load trained models from disk"""
        try:
            if (os.path.exists(f"{self.model_path}price_regressor.pkl") and
                os.path.exists(f"{self.model_path}direction_classifier.pkl")):
                
                self.price_regressor = pickle.load(open(f"{self.model_path}price_regressor.pkl", "rb"))
                self.direction_classifier = pickle.load(open(f"{self.model_path}direction_classifier.pkl", "rb"))
                self.scaler = pickle.load(open(f"{self.model_path}scaler.pkl", "rb"))
                self.label_encoder = pickle.load(open(f"{self.model_path}label_encoder.pkl", "rb"))
                self.feature_columns = pickle.load(open(f"{self.model_path}feature_columns.pkl", "rb"))
                
                self.is_trained = True
                print("Models loaded successfully")
            
        except Exception as e:
            print(f"Error loading models: {str(e)}")
            self.is_trained = False
    
    def retrain_with_new_data(self, new_data: pd.DataFrame):
        """Retrain model with new actual data"""
        try:
            # This method would be used to incorporate real auction data
            # when it becomes available
            print("Retraining model with new data...")
            
            # For now, just retrain with synthetic data
            self.train_initial_model()
            
        except Exception as e:
            print(f"Error retraining model: {str(e)}")
