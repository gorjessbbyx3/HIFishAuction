import schedule
import time
import threading
from datetime import datetime, timedelta
import logging
import os
from typing import Dict, List

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class DataUpdateScheduler:
    """Scheduler for automated data updates"""
    
    def __init__(self):
        self.is_running = False
        self.scheduler_thread = None
        
        # Import services
        from weather_service import WeatherService
        from ocean_service import OceanService
        from data_manager import DataManager
        from prediction_model import PredictionModel
        
        self.weather_service = WeatherService()
        self.ocean_service = OceanService()
        self.data_manager = DataManager()
        self.prediction_model = PredictionModel()
        
        logger.info("DataUpdateScheduler initialized")
    
    def update_weather_data(self):
        """Update weather data from NOAA API"""
        try:
            logger.info("Starting weather data update...")
            self.weather_service.update_weather_data()
            logger.info("Weather data update completed successfully")
        except Exception as e:
            logger.error(f"Error updating weather data: {str(e)}")
    
    def update_ocean_data(self):
        """Update ocean conditions data"""
        try:
            logger.info("Starting ocean data update...")
            self.ocean_service.update_ocean_data()
            logger.info("Ocean data update completed successfully")
        except Exception as e:
            logger.error(f"Error updating ocean data: {str(e)}")
    
    def generate_daily_predictions(self):
        """Generate price predictions for the next few days"""
        try:
            logger.info("Starting daily prediction generation...")
            
            species_list = ["yellowfin_tuna", "bigeye_tuna", "mahi_mahi", "opah", "marlin"]
            prediction_days = 3
            
            for species in species_list:
                try:
                    predictions = self.prediction_model.get_predictions(species, prediction_days)
                    
                    for prediction in predictions:
                        # Store prediction in database
                        prediction_data = {
                            'prediction_date': datetime.now().strftime('%Y-%m-%d'),
                            'target_date': prediction['target_date'],
                            'species': prediction['species'],
                            'predicted_direction': prediction['direction'],
                            'predicted_change_percent': prediction['price_change_percent'],
                            'confidence': prediction['confidence']
                        }
                        
                        self.data_manager.store_prediction(prediction_data)
                    
                    logger.info(f"Generated predictions for {species}")
                    
                except Exception as e:
                    logger.error(f"Error generating predictions for {species}: {str(e)}")
            
            logger.info("Daily prediction generation completed")
            
        except Exception as e:
            logger.error(f"Error in daily prediction generation: {str(e)}")
    
    def simulate_market_data(self):
        """Simulate daily market data for model training"""
        try:
            logger.info("Starting market data simulation...")
            
            from datetime import datetime
            import random
            
            species_list = ["yellowfin_tuna", "bigeye_tuna", "mahi_mahi", "opah", "marlin"]
            current_date = datetime.now().strftime('%Y-%m-%d')
            
            # Get current conditions for realistic simulation
            current_conditions = self.data_manager.get_current_conditions()
            
            for species in species_list:
                try:
                    species_data = self.data_manager.species_data.get(species, {})
                    base_price = species_data.get('base_price', 12.50)
                    volatility = species_data.get('volatility', 0.15)
                    
                    # Add market factors influence
                    price_factor = 1.0
                    
                    # Storm impact
                    storm_risk = current_conditions.get('storm_risk', 0.1)
                    if storm_risk > 0.7:
                        price_factor *= (1 + 0.1 * storm_risk)
                    
                    # Seasonal influence
                    current_month = datetime.now().month
                    seasonal_patterns = self.data_manager.seasonal_patterns.get(species, {})
                    seasonal_multiplier = seasonal_patterns.get('price_multipliers', {}).get(current_month, 1.0)
                    price_factor *= seasonal_multiplier
                    
                    # Random daily variation
                    daily_variation = random.uniform(1 - volatility, 1 + volatility)
                    
                    simulated_price = base_price * price_factor * daily_variation
                    simulated_volume = random.uniform(50, 500)  # lbs
                    
                    # Store market data
                    market_entry = {
                        'date': current_date,
                        'species': species,
                        'price_per_lb': round(simulated_price, 2),
                        'volume': round(simulated_volume, 1),
                        'grade': random.choice(['Premium', 'Standard', 'Commercial']),
                        'size_category': random.choice(['Large', 'Medium', 'Small']),
                        'auction_house': 'Honolulu Fish Auction',
                        'source': 'simulated'
                    }
                    
                    # Store in database using database.py if available
                    try:
                        from database import DatabaseManager
                        db_manager = DatabaseManager()
                        
                        conn = db_manager.db_path
                        import sqlite3
                        
                        with sqlite3.connect(conn) as connection:
                            cursor = connection.cursor()
                            cursor.execute('''
                                INSERT INTO market_data 
                                (date, species, price_per_lb, volume, grade, size_category, auction_house, source)
                                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                            ''', (
                                market_entry['date'],
                                market_entry['species'],
                                market_entry['price_per_lb'],
                                market_entry['volume'],
                                market_entry['grade'],
                                market_entry['size_category'],
                                market_entry['auction_house'],
                                market_entry['source']
                            ))
                            connection.commit()
                    except:
                        # Fallback to basic data manager
                        pass
                    
                    logger.info(f"Simulated market data for {species}: ${simulated_price:.2f}/lb")
                    
                except Exception as e:
                    logger.error(f"Error simulating market data for {species}: {str(e)}")
            
            logger.info("Market data simulation completed")
            
        except Exception as e:
            logger.error(f"Error in market data simulation: {str(e)}")
    
    def cleanup_old_data(self):
        """Clean up old data to maintain database performance"""
        try:
            logger.info("Starting data cleanup...")
            
            # Use enhanced database manager if available
            try:
                from database import DatabaseManager
                db_manager = DatabaseManager()
                db_manager.cleanup_old_data(days_to_keep=365)
            except:
                # Fallback cleanup
                pass
            
            logger.info("Data cleanup completed")
            
        except Exception as e:
            logger.error(f"Error in data cleanup: {str(e)}")
    
    def evaluate_model_performance(self):
        """Evaluate and log model performance"""
        try:
            logger.info("Starting model performance evaluation...")
            
            # Get recent predictions and actual outcomes
            performance_metrics = self.data_manager.get_model_performance()
            
            # Log current performance
            accuracy = performance_metrics.get('accuracy', 0.0)
            logger.info(f"Current model accuracy: {accuracy*100:.1f}%")
            
            # Store performance metrics if database manager is available
            try:
                from database import DatabaseManager
                db_manager = DatabaseManager()
                
                performance_data = {
                    'evaluation_date': datetime.now().strftime('%Y-%m-%d'),
                    'model_version': 'v1.0',
                    'species': None,  # Overall performance
                    'metric_name': 'accuracy',
                    'metric_value': accuracy,
                    'sample_size': performance_metrics.get('total_predictions', 0),
                    'time_period_days': 30
                }
                
                db_manager.store_model_performance(performance_data)
                
            except:
                pass
            
            logger.info("Model performance evaluation completed")
            
        except Exception as e:
            logger.error(f"Error in model performance evaluation: {str(e)}")
    
    def run_full_update_cycle(self):
        """Run complete data update cycle"""
        try:
            logger.info("=== Starting full update cycle ===")
            
            # Update external data sources
            self.update_weather_data()
            time.sleep(2)  # Brief pause between updates
            
            self.update_ocean_data()
            time.sleep(2)
            
            # Generate predictions
            self.generate_daily_predictions()
            time.sleep(1)
            
            # Simulate market activity
            self.simulate_market_data()
            time.sleep(1)
            
            # Performance evaluation (weekly)
            if datetime.now().weekday() == 0:  # Monday
                self.evaluate_model_performance()
                time.sleep(1)
            
            # Cleanup (monthly)
            if datetime.now().day == 1:  # First day of month
                self.cleanup_old_data()
            
            logger.info("=== Full update cycle completed successfully ===")
            
        except Exception as e:
            logger.error(f"Error in full update cycle: {str(e)}")
    
    def setup_schedule(self):
        """Setup scheduled tasks"""
        logger.info("Setting up scheduled tasks...")
        
        # Daily updates
        schedule.every().day.at("06:00").do(self.run_full_update_cycle)
        schedule.every().day.at("12:00").do(self.update_weather_data)
        schedule.every().day.at("18:00").do(self.update_weather_data)
        
        # More frequent weather updates during storm season
        schedule.every(6).hours.do(self.update_weather_data)
        
        # Ocean data updates (less frequent)
        schedule.every(12).hours.do(self.update_ocean_data)
        
        # Prediction generation
        schedule.every().day.at("07:00").do(self.generate_daily_predictions)
        schedule.every().day.at("19:00").do(self.generate_daily_predictions)
        
        # Market simulation
        schedule.every().day.at("08:00").do(self.simulate_market_data)
        schedule.every().day.at("14:00").do(self.simulate_market_data)
        schedule.every().day.at("20:00").do(self.simulate_market_data)
        
        # Weekly performance evaluation
        schedule.every().monday.at("09:00").do(self.evaluate_model_performance)
        
        # Monthly cleanup
        schedule.every().month.do(self.cleanup_old_data)
        
        logger.info("Scheduled tasks configured")
    
    def run_scheduler(self):
        """Run the scheduler in background thread"""
        logger.info("Scheduler thread started")
        
        while self.is_running:
            try:
                schedule.run_pending()
                time.sleep(60)  # Check every minute
            except Exception as e:
                logger.error(f"Error in scheduler thread: {str(e)}")
                time.sleep(300)  # Wait 5 minutes before retrying
    
    def start(self):
        """Start the scheduler"""
        if self.is_running:
            logger.warning("Scheduler is already running")
            return
        
        self.setup_schedule()
        self.is_running = True
        
        # Start scheduler in background thread
        self.scheduler_thread = threading.Thread(target=self.run_scheduler, daemon=True)
        self.scheduler_thread.start()
        
        logger.info("Data update scheduler started successfully")
        
        # Run initial update
        try:
            logger.info("Running initial data update...")
            self.run_full_update_cycle()
        except Exception as e:
            logger.error(f"Error in initial update: {str(e)}")
    
    def stop(self):
        """Stop the scheduler"""
        if not self.is_running:
            logger.warning("Scheduler is not running")
            return
        
        self.is_running = False
        schedule.clear()
        
        if self.scheduler_thread and self.scheduler_thread.is_alive():
            self.scheduler_thread.join(timeout=5)
        
        logger.info("Data update scheduler stopped")
    
    def get_status(self) -> Dict:
        """Get scheduler status"""
        return {
            'is_running': self.is_running,
            'scheduled_jobs': len(schedule.jobs),
            'next_run': str(schedule.next_run()) if schedule.jobs else None,
            'thread_alive': self.scheduler_thread.is_alive() if self.scheduler_thread else False
        }
    
    def run_manual_update(self, update_type: str = "full"):
        """Run manual update of specific type"""
        try:
            logger.info(f"Running manual {update_type} update...")
            
            if update_type == "weather":
                self.update_weather_data()
            elif update_type == "ocean":
                self.update_ocean_data()
            elif update_type == "predictions":
                self.generate_daily_predictions()
            elif update_type == "market":
                self.simulate_market_data()
            elif update_type == "performance":
                self.evaluate_model_performance()
            elif update_type == "full":
                self.run_full_update_cycle()
            else:
                logger.error(f"Unknown update type: {update_type}")
                return False
            
            logger.info(f"Manual {update_type} update completed")
            return True
            
        except Exception as e:
            logger.error(f"Error in manual {update_type} update: {str(e)}")
            return False


# Global scheduler instance
scheduler_instance = None

def get_scheduler():
    """Get global scheduler instance"""
    global scheduler_instance
    if scheduler_instance is None:
        scheduler_instance = DataUpdateScheduler()
    return scheduler_instance

def start_scheduler():
    """Start the global scheduler"""
    scheduler = get_scheduler()
    scheduler.start()
    return scheduler

def stop_scheduler():
    """Stop the global scheduler"""
    global scheduler_instance
    if scheduler_instance:
        scheduler_instance.stop()
        scheduler_instance = None

# CLI interface for standalone operation
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Fish Auction Data Update Scheduler")
    parser.add_argument("--action", choices=["start", "stop", "update", "status"], 
                       default="start", help="Action to perform")
    parser.add_argument("--update-type", choices=["weather", "ocean", "predictions", "market", "performance", "full"],
                       default="full", help="Type of update to run")
    parser.add_argument("--daemon", action="store_true", help="Run as daemon process")
    
    args = parser.parse_args()
    
    if args.action == "start":
        logger.info("Starting Fish Auction Data Scheduler...")
        scheduler = start_scheduler()
        
        if args.daemon:
            logger.info("Running in daemon mode. Press Ctrl+C to stop.")
            try:
                while True:
                    time.sleep(60)
            except KeyboardInterrupt:
                logger.info("Shutting down scheduler...")
                stop_scheduler()
        else:
            logger.info("Scheduler started. Use --daemon to run continuously.")
    
    elif args.action == "update":
        logger.info(f"Running manual {args.update_type} update...")
        scheduler = get_scheduler()
        success = scheduler.run_manual_update(args.update_type)
        if success:
            logger.info("Manual update completed successfully")
        else:
            logger.error("Manual update failed")
    
    elif args.action == "status":
        scheduler = get_scheduler()
        status = scheduler.get_status()
        print(f"Scheduler Status: {status}")
    
    elif args.action == "stop":
        logger.info("Stopping scheduler...")
        stop_scheduler()
        logger.info("Scheduler stopped")
