
import sqlite3
import pandas as pd
from datetime import datetime, timedelta
import numpy as np
import json

def initialize_demo_data():
    """Initialize database with demo data for testing"""
    print("Initializing demo data...")
    
    db_path = "fish_auction.db"
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    # Clear existing data
    cursor.execute("DELETE FROM market_data")
    cursor.execute("DELETE FROM weather_data") 
    cursor.execute("DELETE FROM ocean_conditions")
    
    # Generate 30 days of historical market data
    species_list = ['yellowfin_tuna', 'bigeye_tuna', 'mahi_mahi', 'opah', 'marlin']
    base_prices = {'yellowfin_tuna': 12.50, 'bigeye_tuna': 15.75, 'mahi_mahi': 8.25, 'opah': 10.50, 'marlin': 14.00}
    
    start_date = datetime.now() - timedelta(days=30)
    
    for i in range(30):
        current_date = (start_date + timedelta(days=i)).strftime('%Y-%m-%d')
        
        # Generate weather data
        wind_speed = max(5, np.random.normal(15, 5))
        wave_height = max(1, np.random.normal(4, 2))
        temperature = np.random.normal(26.5, 2)
        storm_warnings = '[]' if np.random.random() > 0.1 else '["Small Craft Advisory"]'
        
        cursor.execute('''
            INSERT INTO weather_data (date, wind_speed, wave_height, storm_warnings, temperature)
            VALUES (?, ?, ?, ?, ?)
        ''', (current_date, wind_speed, wave_height, storm_warnings, temperature))
        
        # Generate ocean data
        sst = np.random.normal(26.5, 1.5)
        chlorophyll = np.random.lognormal(-2, 0.3)
        current_strength = np.random.choice(['Weak', 'Moderate', 'Strong'])
        
        cursor.execute('''
            INSERT INTO ocean_conditions (date, sea_surface_temp, chlorophyll, current_strength)
            VALUES (?, ?, ?, ?)
        ''', (current_date, sst, chlorophyll, current_strength))
        
        # Generate market data for each species
        for species in species_list:
            base_price = base_prices[species]
            
            # Add seasonal and random variation
            seasonal_factor = 1 + 0.2 * np.sin((i / 365) * 2 * np.pi)
            weather_factor = 1 + (wind_speed - 15) * 0.01  # Higher wind = higher prices
            random_factor = np.random.normal(1, 0.1)
            
            price = base_price * seasonal_factor * weather_factor * random_factor
            price = max(5, min(30, price))  # Keep prices realistic
            
            volume = np.random.uniform(50, 200)
            
            cursor.execute('''
                INSERT INTO market_data (date, species, price_per_lb, volume, source)
                VALUES (?, ?, ?, ?, ?)
            ''', (current_date, species, price, volume, 'demo_data'))
    
    conn.commit()
    conn.close()
    
    print(f"Demo data initialized with 30 days of historical data")

if __name__ == "__main__":
    initialize_demo_data()
