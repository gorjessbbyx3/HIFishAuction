#!/usr/bin/env python3
"""Initialize the Hawaii Fish Auction prediction system with working demonstration data"""

import os
import sys
from demo_data_generator import DemoDataGenerator

def initialize_working_system():
    """Initialize a fully working demonstration system"""
    print("Initializing Hawaii Fish Auction Prediction System...")
    print("Creating demonstration data following authentic UFA auction structure...")
    
    try:
        # Create demo data generator
        demo_gen = DemoDataGenerator()
        
        # Initialize demonstration system
        success = demo_gen.initialize_demo_system()
        
        if success:
            print("\n✅ System initialization complete!")
            print("The application now has:")
            print("- 2 years of demonstration fish auction data")
            print("- Current weather and ocean conditions")
            print("- Trained prediction models")
            print("- Working AI analysis capabilities")
            print("\nYou can now use the prediction system.")
            print("Note: For production use, integrate authentic NOAA UFA auction data")
            return True
        else:
            print("❌ System initialization failed")
            return False
            
    except Exception as e:
        print(f"Error during initialization: {str(e)}")
        return False

if __name__ == "__main__":
    initialize_working_system()