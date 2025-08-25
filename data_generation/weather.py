"""
Weather data generation for renewable energy systems.

Contains functions for generating synthetic solar irradiance and wind speed data
with realistic seasonal and diurnal patterns.
"""

import numpy as np
import math


def generate_hourly_solar_irradiance(days=365):
    """Generate synthetic hourly solar irradiance data for a year."""
    hours_per_day = 24
    total_hours = days * hours_per_day
    
    # Create time array
    time = np.arange(total_hours)
    day_of_year = (time // hours_per_day) % 365
    hour_of_day = time % hours_per_day
    
    # Base irradiance pattern (diurnal cycle)
    max_irradiance = 1000  # W/m²
    base_irradiance = np.zeros(total_hours)
    
    for i in range(total_hours):
        if 6 <= hour_of_day[i] <= 18:  # Daylight hours
            # Simulate bell curve for daily irradiance
            hour_factor = math.sin(math.pi * (hour_of_day[i] - 6) / 12)
            # Seasonal variation (maximum in summer, minimum in winter)
            seasonal_factor = 0.5 + 0.5 * math.sin(math.pi * (day_of_year[i] - 80) / 180)
            base_irradiance[i] = max_irradiance * hour_factor * seasonal_factor
    
    # Add random variations for cloud cover
    cloud_variations = np.random.normal(0, 0.2, total_hours)
    cloud_variations = np.clip(cloud_variations, -0.8, 0.2)  # Clouds reduce irradiance
    
    irradiance = base_irradiance * (1 + cloud_variations)
    irradiance = np.clip(irradiance, 0, max_irradiance)
    
    return irradiance


def generate_hourly_wind_speed(days=365):
    """Generate synthetic hourly wind speed data for a year."""
    hours_per_day = 24
    total_hours = days * hours_per_day
    
    # Create time array
    time = np.arange(total_hours)
    day_of_year = (time // hours_per_day) % 365
    hour_of_day = time % hours_per_day
    
    # Base wind pattern (stronger in winter)
    base_wind = np.zeros(total_hours)
    
    for i in range(total_hours):
        # Seasonal variation (maximum in winter, minimum in summer)
        seasonal_factor = 0.5 + 0.5 * math.sin(math.pi * (day_of_year[i] + 100) / 180)
        # Daily variation (stronger during day)
        daily_factor = 0.8 + 0.2 * math.sin(math.pi * (hour_of_day[i] - 3) / 12)
        base_wind[i] = 5 + 5 * seasonal_factor * daily_factor
    
    # Add random variations
    wind_variations = np.random.normal(0, 1.5, total_hours)
    
    wind_speed = base_wind + wind_variations
    wind_speed = np.clip(wind_speed, 0, 25)  # m/s
    
    return wind_speed
