"""
Energy demand data generation for renewable energy systems.

Contains functions for generating synthetic energy demand patterns
with realistic diurnal and seasonal variations.
"""

import numpy as np
import math


def generate_hourly_energy_demand(days=365):
    """Generate synthetic hourly energy demand data for a year."""
    hours_per_day = 24
    total_hours = days * hours_per_day
    
    # Create time array
    time = np.arange(total_hours)
    day_of_year = (time // hours_per_day) % 365
    hour_of_day = time % hours_per_day
    
    # Base demand pattern
    base_demand = np.zeros(total_hours)
    
    for i in range(total_hours):
        # Hour of day pattern (morning and evening peaks)
        if hour_of_day[i] < 6:
            hour_factor = 0.6  # Night
        elif hour_of_day[i] < 9:
            hour_factor = 0.8 + 0.4 * (hour_of_day[i] - 6) / 3  # Morning ramp
        elif hour_of_day[i] < 17:
            hour_factor = 1.0  # Day
        elif hour_of_day[i] < 22:
            hour_factor = 1.2  # Evening peak
        else:
            hour_factor = 0.8 - 0.2 * (hour_of_day[i] - 22) / 2  # Evening decline
        
        # Seasonal variation (more in winter and summer for heating/cooling)
        seasonal_factor = 1.0 + 0.3 * math.cos(2 * math.pi * (day_of_year[i] - 40) / 365)
        
        base_demand[i] = hour_factor * seasonal_factor
    
    # Scale to reasonable household values
    base_demand = base_demand * 2.0  # Average 2 kW
    
    # Add random variations
    demand_variations = np.random.normal(0, 0.3, total_hours)
    
    demand = base_demand * (1 + demand_variations)
    demand = np.clip(demand, 0.2, 10)  # kW
    
    return demand
