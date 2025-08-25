"""
Energy system simulation for renewable energy systems.

Contains the main simulation function that models energy production,
consumption, and storage over time.
"""

import numpy as np
from .production import calculate_solar_energy_production, calculate_wind_energy_production
from .parameters import BATTERY_CAPACITY


def simulate_energy_system(individual, solar_irradiance, wind_speed, energy_demand):
    """Simulate the energy system performance over a year."""
    num_solar_panels, num_wind_turbines, num_batteries = individual
    
    total_days = len(solar_irradiance)
    battery_capacity = num_batteries * BATTERY_CAPACITY  # kWh
    battery_charge = 0.5 * battery_capacity  # Start with half-charged batteries
    
    # Track energy metrics
    energy_supplied = np.zeros(total_days)
    energy_deficit = np.zeros(total_days)
    battery_state = np.zeros(total_days)
    curtailed_energy = np.zeros(total_days)
    
    for day in range(total_days):
        # Calculate energy production
        solar_energy = calculate_solar_energy_production(num_solar_panels, solar_irradiance[day])
        wind_energy = calculate_wind_energy_production(num_wind_turbines, wind_speed[day])
        total_production = solar_energy + wind_energy
        
        # Calculate energy balance
        daily_demand = energy_demand[day]
        energy_balance = total_production - daily_demand
        
        if energy_balance >= 0:
            # Excess energy, charge battery
            battery_charge += energy_balance
            if battery_charge > battery_capacity:
                curtailed_energy[day] = battery_charge - battery_capacity
                battery_charge = battery_capacity
            energy_supplied[day] = daily_demand
            energy_deficit[day] = 0
        else:
            # Energy deficit, discharge battery
            energy_deficit_amount = abs(energy_balance)
            if battery_charge >= energy_deficit_amount:
                battery_charge -= energy_deficit_amount
                energy_supplied[day] = daily_demand
                energy_deficit[day] = 0
            else:
                energy_supplied[day] = total_production + battery_charge
                energy_deficit[day] = daily_demand - energy_supplied[day]
                battery_charge = 0
        
        battery_state[day] = battery_charge
    
    return {
        'energy_supplied': energy_supplied,
        'energy_deficit': energy_deficit,
        'battery_state': battery_state,
        'curtailed_energy': curtailed_energy
    }
