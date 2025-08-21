import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import sqlite3
import os

class DemoDataGenerator:
    """Generate demonstration data that mimics authentic UFA auction structure"""
    
    def __init__(self):
        self.db_path = "fish_auction.db"
        
    def create_demo_ufa_auction_data(self):
        """Create demonstration data with authentic UFA auction structure"""
        print("Creating demonstration data with authentic UFA auction structure...")
        
        # Species from actual Hawaii auctions
        species_data = {
            'Yellowfin Tuna': {'base_price': 12.50, 'volatility': 0.15},
            'Bigeye Tuna': {'base_price': 15.75, 'volatility': 0.18},
            'Mahi-mahi': {'base_price': 8.25, 'volatility': 0.22},
            'Opah': {'base_price': 6.80, 'volatility': 0.25},
            'Marlin': {'base_price': 5.90, 'volatility': 0.20},
            'Ono': {'base_price': 11.40, 'volatility': 0.16}
        }
        
        # Generate 2 years of demo data
        start_date = datetime.now() - timedelta(days=730)
        end_date = datetime.now() - timedelta(days=1)
        
        demo_records = []
        
        current_date = start_date
        while current_date <= end_date:
            # Skip some days to simulate realistic auction frequency
            if np.random.random() > 0.7:  # 30% chance of auction on any given day
                current_date += timedelta(days=1)
                continue
                
            for species, info in species_data.items():
                # Simulate multiple lots per species per day
                num_lots = np.random.randint(1, 4)
                
                for lot in range(num_lots):
                    # Price with seasonal and random variation
                    seasonal_factor = 1 + 0.1 * np.sin(2 * np.pi * current_date.timetuple().tm_yday / 365)
                    price_variation = np.random.normal(1, info['volatility'])
                    price_per_lb = info['base_price'] * seasonal_factor * price_variation
                    price_per_lb = max(price_per_lb, 1.0)  # Minimum $1/lb
                    
                    # Quantity sold (pounds)
                    quantity = np.random.exponential(50) + 10  # 10-500 lbs typical
                    
                    demo_records.append({
                        'date': current_date.strftime('%Y-%m-%d'),
                        'species': species,
                        'price_per_lb': round(price_per_lb, 2),
                        'volume': round(quantity, 1),
                        'weight': round(quantity, 1),
                        'market_location': 'Honolulu Fish Auction',
                        'origin': 'Hawaii Longline Fleet',
                        'product_form': 'Whole',
                        'preservation_method': 'Fresh',
                        'data_source': 'demo_ufa_structure',
                        'import_date': datetime.now().strftime('%Y-%m-%d')
                    })
            
            current_date += timedelta(days=1)
        
        # Create DataFrame and store
        df = pd.DataFrame(demo_records)
        self._store_demo_data(df)
        
        print(f"Generated {len(df)} demo records following UFA auction structure")
        return df
    
    def _store_demo_data(self, df):
        """Store demonstration data in database"""
        conn = sqlite3.connect(self.db_path)
        
        # Create tables if they don't exist
        cursor = conn.cursor()
        
        # Enhanced market_data table for prediction model
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS market_data (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                date TEXT NOT NULL,
                species TEXT NOT NULL,
                price_per_lb REAL NOT NULL,
                volume REAL,
                source TEXT DEFAULT 'demo',
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        # Historical market data table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS historical_market_data (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                date TEXT NOT NULL,
                species TEXT NOT NULL,
                price_per_lb REAL NOT NULL,
                volume REAL,
                weight REAL,
                market_location TEXT,
                origin TEXT,
                product_form TEXT,
                preservation_method TEXT,
                data_source TEXT NOT NULL,
                import_date TEXT NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        # Clear existing demo data
        cursor.execute("DELETE FROM market_data WHERE source = 'demo'")
        cursor.execute("DELETE FROM historical_market_data WHERE data_source = 'demo_ufa_structure'")
        
        # Insert into both tables
        df.to_sql('historical_market_data', conn, if_exists='append', index=False)
        
        # Insert aggregated data for prediction model
        for _, row in df.iterrows():
            cursor.execute('''
                INSERT INTO market_data (date, species, price_per_lb, volume, source)
                VALUES (?, ?, ?, ?, ?)
            ''', (row['date'], row['species'], row['price_per_lb'], row['volume'], 'demo'))
        
        conn.commit()
        conn.close()
        
        print("Demonstration data stored in database")
    
    def create_recent_market_conditions(self):
        """Create recent market conditions for current predictions"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Create conditions table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS current_conditions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                date TEXT NOT NULL,
                condition_type TEXT NOT NULL,
                value REAL NOT NULL,
                unit TEXT,
                source TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        # Clear existing data
        cursor.execute("DELETE FROM current_conditions")
        
        # Add recent conditions
        today = datetime.now().strftime('%Y-%m-%d')
        conditions = [
            (today, 'wind_speed', 15.5, 'knots', 'demo_weather'),
            (today, 'wave_height', 3.2, 'feet', 'demo_weather'),
            (today, 'sea_surface_temp', 26.8, 'celsius', 'demo_ocean'),
            (today, 'chlorophyll', 0.18, 'mg/m3', 'demo_ocean'),
            (today, 'market_activity', 0.85, 'normalized', 'demo_market')
        ]
        
        for condition in conditions:
            cursor.execute('''
                INSERT INTO current_conditions (date, condition_type, value, unit, source)
                VALUES (?, ?, ?, ?, ?)
            ''', condition)
        
        conn.commit()
        conn.close()
        
        print("Recent market conditions created")
    
    def initialize_demo_system(self):
        """Initialize complete demonstration system"""
        print("Initializing demonstration Hawaii Fish Auction system...")
        print("Note: This uses demonstration data following authentic UFA auction structure")
        
        # Generate demo data
        self.create_demo_ufa_auction_data()
        self.create_recent_market_conditions()
        
        print("\nDemonstration system ready!")
        print("- Historical price data: 2 years of demo UFA auction records")
        print("- Current conditions: Demo weather and ocean data")
        print("- Prediction model: Ready for training with demo data")
        print("\nTo use authentic data, contact NOAA PIFSC for UFA auction access")
        
        return True