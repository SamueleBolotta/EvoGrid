"""
Objective functions for energy system optimization.

Contains functions for calculating cost, reliability, and environmental impact
of renewable energy system configurations.
"""

import numpy as np
from .parameters import (
    SOLAR_PANEL_COST, WIND_TURBINE_COST, BATTERY_COST, BATTERY_CYCLE_LIFE,
    SOLAR_PANEL_LIFECYCLE_EMISSIONS, WIND_TURBINE_LIFECYCLE_EMISSIONS, BATTERY_LIFECYCLE_EMISSIONS,
    PROJECT_LIFETIME
)
from .simulation import simulate_energy_system
from .production import calculate_solar_energy_production, calculate_wind_energy_production


def calculate_total_cost(individual):
    """Calculate the total lifecycle cost of the energy system."""
    num_solar_panels, num_wind_turbines, num_batteries = individual
    
    # Capital costs
    solar_capital_cost = num_solar_panels * SOLAR_PANEL_COST
    wind_capital_cost = num_wind_turbines * WIND_TURBINE_COST
    battery_capital_cost = num_batteries * BATTERY_COST
    
    # Operational costs (simplified)
    solar_annual_opex = solar_capital_cost * 0.02  # 2% of capital cost per year
    wind_annual_opex = wind_capital_cost * 0.03  # 3% of capital cost per year
    battery_annual_opex = 0  # Simplified
    
    # Battery replacement costs
    avg_daily_cycles = 0.8  # Assumed average daily battery cycles
    total_cycles = avg_daily_cycles * 365 * PROJECT_LIFETIME
    replacements_needed = max(0, (total_cycles / BATTERY_CYCLE_LIFE) - 1)  # -1 because first set is in capital
    battery_replacement_cost = replacements_needed * battery_capital_cost
    
    # Total cost
    total_capital_cost = solar_capital_cost + wind_capital_cost + battery_capital_cost
    total_opex = (solar_annual_opex + wind_annual_opex + battery_annual_opex) * PROJECT_LIFETIME
    
    total_cost = total_capital_cost + total_opex + battery_replacement_cost
    
    return total_cost


def calculate_reliability(individual, solar_irradiance, wind_speed, energy_demand):
    """Calculate the energy supply reliability (loss of load probability)."""
    simulation_results = simulate_energy_system(individual, solar_irradiance, wind_speed, energy_demand)
    
    # Calculate loss of load probability (LOLP)
    days_with_deficit = sum(1 for deficit in simulation_results['energy_deficit'] if deficit > 0)
    total_days = len(energy_demand)
    lolp = days_with_deficit / total_days
    
    # Calculate energy index of reliability (EIR)
    total_demand = sum(energy_demand)
    total_deficit = sum(simulation_results['energy_deficit'])
    eir = 1 - (total_deficit / total_demand)
    
    # Return reliability (higher is better)
    reliability = eir
    return reliability


def calculate_environmental_impact(individual, daily_solar_irradiance=None, daily_wind_speed=None):
    """Calculate the environmental impact based on lifecycle emissions."""
    num_solar_panels, num_wind_turbines, num_batteries = individual
    
    # Lifecycle emissions
    solar_emissions = num_solar_panels * SOLAR_PANEL_LIFECYCLE_EMISSIONS
    wind_emissions = num_wind_turbines * WIND_TURBINE_LIFECYCLE_EMISSIONS
    battery_emissions = num_batteries * BATTERY_LIFECYCLE_EMISSIONS
    
    # Battery replacement emissions
    avg_daily_cycles = 0.8  # Assumed average daily battery cycles
    total_cycles = avg_daily_cycles * 365 * PROJECT_LIFETIME
    replacements_needed = max(0, (total_cycles / BATTERY_CYCLE_LIFE) - 1)  # -1 because first set is in capital
    battery_replacement_emissions = replacements_needed * battery_emissions
    
    # Total emissions
    total_emissions = solar_emissions + wind_emissions + battery_emissions + battery_replacement_emissions
    
    # Normalize by energy produced over lifetime (simplified calculation)
    if daily_solar_irradiance is not None and daily_wind_speed is not None:
        avg_daily_solar = calculate_solar_energy_production(num_solar_panels, np.mean(daily_solar_irradiance))
        avg_daily_wind = calculate_wind_energy_production(num_wind_turbines, np.mean(daily_wind_speed))
        total_energy_produced = (avg_daily_solar + avg_daily_wind) * 365 * PROJECT_LIFETIME
    else:
        # Use approximate values if data not provided
        avg_daily_solar = calculate_solar_energy_production(num_solar_panels, 500)  # Approximate irradiance
        avg_daily_wind = calculate_wind_energy_production(num_wind_turbines, 8)     # Approximate wind speed
        total_energy_produced = (avg_daily_solar + avg_daily_wind) * 365 * PROJECT_LIFETIME
    
    if total_energy_produced > 0:
        emissions_intensity = total_emissions / total_energy_produced  # kg CO2 / kWh
    else:
        emissions_intensity = float('inf')
    
    return emissions_intensity
