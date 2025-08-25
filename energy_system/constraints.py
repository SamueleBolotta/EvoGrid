"""
Constraint checking for energy system optimization.

Contains functions for checking system constraints such as land use,
budget, and minimum energy requirements.
"""

import numpy as np
from .parameters import (
    MAX_LAND_AREA, MAX_BUDGET, MIN_ENERGY_REQUIREMENT,
    SOLAR_PANEL_LAND_USE, WIND_TURBINE_LAND_USE,
    SOLAR_PANEL_COST, WIND_TURBINE_COST, BATTERY_COST
)
from .production import calculate_solar_energy_production, calculate_wind_energy_production


def check_constraints(individual, daily_solar_irradiance=None, daily_wind_speed=None):
    """Check if the individual satisfies all constraints."""
    num_solar_panels, num_wind_turbines, num_batteries = individual
    
    # Land use constraint
    total_land_use = (num_solar_panels * SOLAR_PANEL_LAND_USE + 
                     num_wind_turbines * WIND_TURBINE_LAND_USE)
    land_constraint_satisfied = total_land_use <= MAX_LAND_AREA
    
    # Budget constraint
    total_capital_cost = (num_solar_panels * SOLAR_PANEL_COST + 
                         num_wind_turbines * WIND_TURBINE_COST + 
                         num_batteries * BATTERY_COST)
    budget_constraint_satisfied = total_capital_cost <= MAX_BUDGET
    
    # Minimum energy requirement
    if daily_solar_irradiance is not None and daily_wind_speed is not None:
        avg_daily_solar = calculate_solar_energy_production(num_solar_panels, np.mean(daily_solar_irradiance))
        avg_daily_wind = calculate_wind_energy_production(num_wind_turbines, np.mean(daily_wind_speed))
    else:
        # Use approximate values if data not provided
        avg_daily_solar = calculate_solar_energy_production(num_solar_panels, 500)  # Approximate irradiance
        avg_daily_wind = calculate_wind_energy_production(num_wind_turbines, 8)     # Approximate wind speed
    
    avg_daily_production = avg_daily_solar + avg_daily_wind
    energy_constraint_satisfied = avg_daily_production >= MIN_ENERGY_REQUIREMENT
    
    return land_constraint_satisfied and budget_constraint_satisfied and energy_constraint_satisfied
