import sqlite3
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
import json
import os

class DatabaseManager:
    """Enhanced database operations for fish auction prediction system"""
    
    def __init__(self, db_path: str = "fish_auction.db"):
        self.db_path = db_path
        self.init_enhanced_database()
    
    def init_enhanced_database(self):
        """Initialize database with enhanced schema"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Enhanced weather data table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS weather_data (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                date TEXT NOT NULL,
                wind_speed REAL,
                wind_direction TEXT,
                wave_height REAL,
                storm_warnings TEXT,
                temperature REAL,
                humidity REAL,
                pressure REAL,
                visibility REAL,
                forecast_confidence REAL,
                data_source TEXT DEFAULT 'NOAA',
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                UNIQUE(date, data_source)
            )
        ''')
        
        # Enhanced ocean conditions table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS ocean_conditions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                date TEXT NOT NULL,
                sea_surface_temp REAL,
                chlorophyll REAL,
                current_strength TEXT,
                current_direction TEXT,
                upwelling_index REAL,
                salinity REAL,
                dissolved_oxygen REAL,
                ph_level REAL,
                turbidity REAL,
                data_source TEXT DEFAULT 'satellite',
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                UNIQUE(date, data_source)
            )
        ''')
        
        # Enhanced catch data table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS catch_data (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                date TEXT NOT NULL,
                species TEXT NOT NULL,
                weight_landed REAL,
                vessel_count INTEGER,
                avg_size REAL,
                fishing_method TEXT,
                fishing_area TEXT,
                cpue REAL,
                fishing_hours REAL,
                moon_phase TEXT,
                data_source TEXT DEFAULT 'simulated',
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        # Enhanced price predictions table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS price_predictions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                prediction_date TEXT NOT NULL,
                target_date TEXT NOT NULL,
                species TEXT NOT NULL,
                predicted_direction TEXT,
                predicted_change_percent REAL,
                predicted_price REAL,
                confidence REAL,
                model_version TEXT,
                feature_importance TEXT,
                actual_price REAL,
                actual_direction TEXT,
                prediction_error REAL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        # Enhanced market data table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS market_data (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                date TEXT NOT NULL,
                species TEXT NOT NULL,
                price_per_lb REAL,
                volume REAL,
                grade TEXT,
                size_category TEXT,
                auction_house TEXT,
                buyer_type TEXT,
                settlement_time TEXT,
                source TEXT DEFAULT 'simulated',
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        # Buyer recommendations table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS buyer_recommendations (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                recommendation_date TEXT NOT NULL,
                target_date TEXT NOT NULL,
                species TEXT NOT NULL,
                action TEXT NOT NULL,
                confidence REAL,
                potential_savings REAL,
                investment_amount REAL,
                reasoning TEXT,
                risk_level TEXT,
                actual_outcome TEXT,
                actual_savings REAL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        # Model performance tracking table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS model_performance (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                evaluation_date TEXT NOT NULL,
                model_version TEXT,
                species TEXT,
                metric_name TEXT,
                metric_value REAL,
                sample_size INTEGER,
                time_period_days INTEGER,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        # Economic indicators table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS economic_indicators (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                date TEXT NOT NULL,
                fuel_price REAL,
                tourism_index REAL,
                restaurant_demand_index REAL,
                export_volume REAL,
                import_volume REAL,
                competitor_regions TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        # User feedback table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS user_feedback (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                feedback_date TEXT NOT NULL,
                prediction_id INTEGER,
                user_rating INTEGER,
                accuracy_rating INTEGER,
                usefulness_rating INTEGER,
                comments TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (prediction_id) REFERENCES price_predictions (id)
            )
        ''')
        
        # Create indexes for better performance
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_weather_date ON weather_data(date)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_ocean_date ON ocean_conditions(date)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_predictions_target_date ON price_predictions(target_date)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_market_date_species ON market_data(date, species)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_recommendations_date ON buyer_recommendations(recommendation_date)')
        
        conn.commit()
        conn.close()
    
    def store_comprehensive_weather_data(self, weather_data: Dict):
        """Store comprehensive weather data"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        try:
            cursor.execute('''
                INSERT OR REPLACE INTO weather_data 
                (date, wind_speed, wind_direction, wave_height, storm_warnings, temperature, 
                 humidity, pressure, visibility, forecast_confidence, data_source)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                weather_data['date'],
                weather_data.get('wind_speed'),
                weather_data.get('wind_direction'),
                weather_data.get('wave_height'),
                json.dumps(weather_data.get('storm_warnings', [])),
                weather_data.get('temperature'),
                weather_data.get('humidity'),
                weather_data.get('pressure'),
                weather_data.get('visibility'),
                weather_data.get('forecast_confidence'),
                weather_data.get('data_source', 'NOAA')
            ))
            
            conn.commit()
        except Exception as e:
            print(f"Error storing weather data: {str(e)}")
        finally:
            conn.close()
    
    def store_comprehensive_ocean_data(self, ocean_data: Dict):
        """Store comprehensive ocean conditions data"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        try:
            cursor.execute('''
                INSERT OR REPLACE INTO ocean_conditions 
                (date, sea_surface_temp, chlorophyll, current_strength, current_direction,
                 upwelling_index, salinity, dissolved_oxygen, ph_level, turbidity, data_source)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                ocean_data['date'],
                ocean_data.get('sea_surface_temp'),
                ocean_data.get('chlorophyll'),
                ocean_data.get('current_strength'),
                ocean_data.get('current_direction'),
                ocean_data.get('upwelling_index'),
                ocean_data.get('salinity'),
                ocean_data.get('dissolved_oxygen'),
                ocean_data.get('ph_level'),
                ocean_data.get('turbidity'),
                ocean_data.get('data_source', 'satellite')
            ))
            
            conn.commit()
        except Exception as e:
            print(f"Error storing ocean data: {str(e)}")
        finally:
            conn.close()
    
    def store_buyer_recommendation(self, recommendation: Dict):
        """Store buyer recommendation for tracking performance"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        try:
            cursor.execute('''
                INSERT INTO buyer_recommendations 
                (recommendation_date, target_date, species, action, confidence, 
                 potential_savings, investment_amount, reasoning, risk_level)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                recommendation['recommendation_date'],
                recommendation['target_date'],
                recommendation['species'],
                recommendation['action'],
                recommendation['confidence'],
                recommendation['potential_savings'],
                recommendation['investment_amount'],
                recommendation['reasoning'],
                recommendation['risk_level']
            ))
            
            conn.commit()
        except Exception as e:
            print(f"Error storing recommendation: {str(e)}")
        finally:
            conn.close()
    
    def update_recommendation_outcome(self, recommendation_id: int, actual_outcome: str, actual_savings: float):
        """Update recommendation with actual outcome"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        try:
            cursor.execute('''
                UPDATE buyer_recommendations 
                SET actual_outcome = ?, actual_savings = ?
                WHERE id = ?
            ''', (actual_outcome, actual_savings, recommendation_id))
            
            conn.commit()
        except Exception as e:
            print(f"Error updating recommendation outcome: {str(e)}")
        finally:
            conn.close()
    
    def store_model_performance(self, performance_data: Dict):
        """Store model performance metrics"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        try:
            cursor.execute('''
                INSERT INTO model_performance 
                (evaluation_date, model_version, species, metric_name, metric_value, 
                 sample_size, time_period_days)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            ''', (
                performance_data['evaluation_date'],
                performance_data['model_version'],
                performance_data['species'],
                performance_data['metric_name'],
                performance_data['metric_value'],
                performance_data['sample_size'],
                performance_data['time_period_days']
            ))
            
            conn.commit()
        except Exception as e:
            print(f"Error storing model performance: {str(e)}")
        finally:
            conn.close()
    
    def get_recommendation_performance(self, days_back: int = 30) -> Dict:
        """Get recommendation performance over specified period"""
        conn = sqlite3.connect(self.db_path)
        
        try:
            query = '''
                SELECT 
                    action,
                    COUNT(*) as total_recommendations,
                    AVG(actual_savings) as avg_savings,
                    SUM(CASE WHEN actual_savings > 0 THEN 1 ELSE 0 END) as successful_recommendations,
                    AVG(confidence) as avg_confidence
                FROM buyer_recommendations 
                WHERE recommendation_date >= date('now', '-{} days')
                AND actual_outcome IS NOT NULL
                GROUP BY action
            '''.format(days_back)
            
            results = pd.read_sql_query(query, conn)
            
            performance_summary = {}
            for _, row in results.iterrows():
                action = row['action']
                performance_summary[action] = {
                    'total_recommendations': row['total_recommendations'],
                    'avg_savings': row['avg_savings'] or 0,
                    'success_rate': (row['successful_recommendations'] / row['total_recommendations']) if row['total_recommendations'] > 0 else 0,
                    'avg_confidence': row['avg_confidence'] or 0
                }
            
            return performance_summary
            
        except Exception as e:
            print(f"Error getting recommendation performance: {str(e)}")
            return {}
        finally:
            conn.close()
    
    def get_model_accuracy_trends(self, species: str = None) -> pd.DataFrame:
        """Get model accuracy trends over time"""
        conn = sqlite3.connect(self.db_path)
        
        try:
            query = '''
                SELECT 
                    evaluation_date,
                    species,
                    metric_name,
                    metric_value
                FROM model_performance 
                WHERE metric_name IN ('accuracy', 'mae', 'rmse')
            '''
            
            if species:
                query += f" AND (species = '{species}' OR species IS NULL)"
            
            query += " ORDER BY evaluation_date DESC LIMIT 100"
            
            return pd.read_sql_query(query, conn)
            
        except Exception as e:
            print(f"Error getting accuracy trends: {str(e)}")
            return pd.DataFrame()
        finally:
            conn.close()
    
    def get_environmental_correlations(self, species: str, days_back: int = 90) -> Dict:
        """Get correlations between environmental factors and prices"""
        conn = sqlite3.connect(self.db_path)
        
        try:
            query = '''
                SELECT 
                    m.date,
                    m.price_per_lb,
                    w.wind_speed,
                    w.wave_height,
                    w.temperature,
                    o.sea_surface_temp,
                    o.chlorophyll,
                    o.upwelling_index
                FROM market_data m
                LEFT JOIN weather_data w ON m.date = w.date
                LEFT JOIN ocean_conditions o ON m.date = o.date
                WHERE m.species = ? 
                AND m.date >= date('now', '-{} days')
                ORDER BY m.date
            '''.format(days_back)
            
            df = pd.read_sql_query(query, conn, params=[species])
            
            if df.empty:
                return {}
            
            # Calculate correlations
            correlations = {}
            price_col = 'price_per_lb'
            
            for col in ['wind_speed', 'wave_height', 'temperature', 'sea_surface_temp', 'chlorophyll', 'upwelling_index']:
                if col in df.columns and not df[col].isna().all():
                    corr = df[price_col].corr(df[col])
                    if not pd.isna(corr):
                        correlations[col] = round(corr, 3)
            
            return correlations
            
        except Exception as e:
            print(f"Error calculating correlations: {str(e)}")
            return {}
        finally:
            conn.close()
    
    def store_user_feedback(self, feedback: Dict):
        """Store user feedback on predictions"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        try:
            cursor.execute('''
                INSERT INTO user_feedback 
                (feedback_date, prediction_id, user_rating, accuracy_rating, 
                 usefulness_rating, comments)
                VALUES (?, ?, ?, ?, ?, ?)
            ''', (
                feedback['feedback_date'],
                feedback.get('prediction_id'),
                feedback['user_rating'],
                feedback['accuracy_rating'],
                feedback['usefulness_rating'],
                feedback.get('comments', '')
            ))
            
            conn.commit()
        except Exception as e:
            print(f"Error storing user feedback: {str(e)}")
        finally:
            conn.close()
    
    def get_data_quality_metrics(self) -> Dict:
        """Get data quality metrics"""
        conn = sqlite3.connect(self.db_path)
        
        try:
            metrics = {}
            
            # Weather data quality
            weather_query = '''
                SELECT 
                    COUNT(*) as total_records,
                    COUNT(wind_speed) as wind_speed_records,
                    COUNT(temperature) as temperature_records,
                    MAX(date) as latest_date,
                    MIN(date) as earliest_date
                FROM weather_data
                WHERE date >= date('now', '-30 days')
            '''
            weather_result = pd.read_sql_query(weather_query, conn).iloc[0]
            
            metrics['weather'] = {
                'total_records': weather_result['total_records'],
                'completeness_wind': (weather_result['wind_speed_records'] / weather_result['total_records']) if weather_result['total_records'] > 0 else 0,
                'completeness_temp': (weather_result['temperature_records'] / weather_result['total_records']) if weather_result['total_records'] > 0 else 0,
                'latest_date': weather_result['latest_date'],
                'data_age_days': (datetime.now() - datetime.strptime(weather_result['latest_date'], '%Y-%m-%d')).days if weather_result['latest_date'] else None
            }
            
            # Ocean data quality
            ocean_query = '''
                SELECT 
                    COUNT(*) as total_records,
                    COUNT(sea_surface_temp) as sst_records,
                    COUNT(chlorophyll) as chlorophyll_records,
                    MAX(date) as latest_date
                FROM ocean_conditions
                WHERE date >= date('now', '-30 days')
            '''
            ocean_result = pd.read_sql_query(ocean_query, conn).iloc[0]
            
            metrics['ocean'] = {
                'total_records': ocean_result['total_records'],
                'completeness_sst': (ocean_result['sst_records'] / ocean_result['total_records']) if ocean_result['total_records'] > 0 else 0,
                'completeness_chlorophyll': (ocean_result['chlorophyll_records'] / ocean_result['total_records']) if ocean_result['total_records'] > 0 else 0,
                'latest_date': ocean_result['latest_date'],
                'data_age_days': (datetime.now() - datetime.strptime(ocean_result['latest_date'], '%Y-%m-%d')).days if ocean_result['latest_date'] else None
            }
            
            return metrics
            
        except Exception as e:
            print(f"Error getting data quality metrics: {str(e)}")
            return {}
        finally:
            conn.close()
    
    def cleanup_old_data(self, days_to_keep: int = 365):
        """Clean up old data to maintain database performance"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        try:
            cutoff_date = (datetime.now() - timedelta(days=days_to_keep)).strftime('%Y-%m-%d')
            
            # Clean up old weather data (keep recent predictions)
            cursor.execute('DELETE FROM weather_data WHERE date < ? AND data_source != "forecast"', (cutoff_date,))
            
            # Clean up old ocean data
            cursor.execute('DELETE FROM ocean_conditions WHERE date < ?', (cutoff_date,))
            
            # Clean up old market data (keep for analysis)
            cursor.execute('DELETE FROM market_data WHERE date < ? AND source = "simulated"', (cutoff_date,))
            
            # Clean up old user feedback
            cursor.execute('DELETE FROM user_feedback WHERE feedback_date < ?', (cutoff_date,))
            
            deleted_rows = cursor.rowcount
            conn.commit()
            
            print(f"Cleaned up {deleted_rows} old records")
            
        except Exception as e:
            print(f"Error cleaning up data: {str(e)}")
        finally:
            conn.close()
    
    def export_data_for_analysis(self, start_date: str, end_date: str, species: str = None) -> Dict[str, pd.DataFrame]:
        """Export data for external analysis"""
        conn = sqlite3.connect(self.db_path)
        
        try:
            tables = {}
            
            # Market data
            market_query = "SELECT * FROM market_data WHERE date BETWEEN ? AND ?"
            params = [start_date, end_date]
            
            if species:
                market_query += " AND species = ?"
                params.append(species)
            
            tables['market_data'] = pd.read_sql_query(market_query, conn, params=params)
            
            # Weather data
            tables['weather_data'] = pd.read_sql_query(
                "SELECT * FROM weather_data WHERE date BETWEEN ? AND ?",
                conn, params=[start_date, end_date]
            )
            
            # Ocean data
            tables['ocean_conditions'] = pd.read_sql_query(
                "SELECT * FROM ocean_conditions WHERE date BETWEEN ? AND ?",
                conn, params=[start_date, end_date]
            )
            
            # Predictions
            tables['predictions'] = pd.read_sql_query(
                "SELECT * FROM price_predictions WHERE target_date BETWEEN ? AND ?",
                conn, params=[start_date, end_date]
            )
            
            return tables
            
        except Exception as e:
            print(f"Error exporting data: {str(e)}")
            return {}
        finally:
            conn.close()
