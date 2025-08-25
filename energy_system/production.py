"""
Energy production calculations for renewable energy systems.

Contains functions for calculating energy production from solar panels
and wind turbines.
"""

import math
from .parameters import SOLAR_PANEL_CAPACITY, WIND_TURBINE_CAPACITY


def calculate_solar_energy_production(num_panels, irradiance):
    """Calculate daily energy production from solar panels."""
    efficiency = 0.15  # Solar panel efficiency
    # Irradiance is in W/m², convert to kWh/day
    energy = num_panels * SOLAR_PANEL_CAPACITY * (irradiance / 1000) * efficiency * 24
    return energy


def calculate_wind_energy_production(num_turbines, wind_speed):
    """Calculate daily energy production from wind turbines."""
    # Power curve approximation
    # No power below cut-in speed or above cut-out speed
    cut_in_speed = 3.0  # m/s
    cut_out_speed = 25.0  # m/s
    rated_speed = 12.0  # m/s
    
    if wind_speed < cut_in_speed or wind_speed > cut_out_speed:
        power = 0
    elif wind_speed < rated_speed:
        # Cubic relationship between wind speed and power
        power_factor = ((wind_speed - cut_in_speed) / (rated_speed - cut_in_speed))**3
        power = WIND_TURBINE_CAPACITY * power_factor
    else:
        power = WIND_TURBINE_CAPACITY
    
    # Convert power (kW) to energy (kWh/day)
    energy = num_turbines * power * 24
    return energy
