"""
Sensitivity analysis for energy system optimization.

Contains functions for performing sensitivity analysis on system parameters
and visualizing the results.
"""

import numpy as np
import matplotlib.pyplot as plt
import os
from energy_system.parameters import (
    SOLAR_PANEL_COST, WIND_TURBINE_COST, BATTERY_COST,
    SOLAR_PANEL_CAPACITY, WIND_TURBINE_CAPACITY,
    SOLAR_PANEL_LIFECYCLE_EMISSIONS, WIND_TURBINE_LIFECYCLE_EMISSIONS,
    BATTERY_LIFECYCLE_EMISSIONS
)


def perform_sensitivity_analysis(base_solution, evaluate_func, daily_data, save_path=None):
    """Perform sensitivity analysis on key parameters."""
    # Import necessary modules and variables - need to access the modules where functions import from
    from energy_system import parameters
    import energy_system.objective_functions as obj_funcs
    import energy_system.production as production
    
    daily_solar_irradiance, daily_wind_speed, daily_energy_demand = daily_data
    
    # Extract parameters from base solution
    base_solar_panels = base_solution["solar_panels"]
    base_wind_turbines = base_solution["wind_turbines"]
    base_batteries = base_solution["batteries"]
    base_individual = [base_solar_panels, base_wind_turbines, base_batteries]
    
    # Calculate base metrics
    base_cost, base_reliability, base_env_impact = evaluate_func(base_individual)
    
    print(f"\n{'='*80}")
    print(f"SENSITIVITY ANALYSIS FOR: {base_solution['name']}")
    print(f"{'='*80}")
    print(f"Base configuration: {base_solar_panels} Solar Panels, {base_wind_turbines} Wind Turbines, {base_batteries} Batteries")
    print(f"Base metrics: Cost=${base_cost:.2f}, Reliability={base_reliability:.4f}, Env. Impact={base_env_impact:.6f} kg CO2/kWh")
    
    # Define parameter ranges for sensitivity analysis
    parameter_ranges = {
        "Solar Panel Cost": np.linspace(parameters.SOLAR_PANEL_COST * 0.5, parameters.SOLAR_PANEL_COST * 1.5, 10),
        "Wind Turbine Cost": np.linspace(parameters.WIND_TURBINE_COST * 0.5, parameters.WIND_TURBINE_COST * 1.5, 10),
        "Battery Cost": np.linspace(parameters.BATTERY_COST * 0.5, parameters.BATTERY_COST * 1.5, 10),
        "Solar Panel Capacity": np.linspace(parameters.SOLAR_PANEL_CAPACITY * 0.8, parameters.SOLAR_PANEL_CAPACITY * 1.2, 10),
        "Wind Turbine Capacity": np.linspace(parameters.WIND_TURBINE_CAPACITY * 0.8, parameters.WIND_TURBINE_CAPACITY * 1.2, 10),
        "Solar Panel Emissions": np.linspace(parameters.SOLAR_PANEL_LIFECYCLE_EMISSIONS * 0.5, parameters.SOLAR_PANEL_LIFECYCLE_EMISSIONS * 1.5, 10),
        "Wind Turbine Emissions": np.linspace(parameters.WIND_TURBINE_LIFECYCLE_EMISSIONS * 0.5, parameters.WIND_TURBINE_LIFECYCLE_EMISSIONS * 1.5, 10)
    }
    
    # Results dictionaries
    results = {}
    
    # Iterate through each parameter and perform analysis
    for param_name, param_range in parameter_ranges.items():
        print(f"\nAnalyzing sensitivity to {param_name}...")
        
        # Initialize results for this parameter
        results[param_name] = {
            "values": param_range,
            "cost": [],
            "reliability": [],
            "env_impact": []
        }
        
        for value in param_range:
            # Store original values
            original_value = None
            
            # Modify parameter based on name - need to modify in the modules that import them
            if param_name == "Solar Panel Cost":
                original_value = parameters.SOLAR_PANEL_COST
                parameters.SOLAR_PANEL_COST = value
                # Also modify in objective_functions module
                obj_funcs.SOLAR_PANEL_COST = value
            elif param_name == "Wind Turbine Cost":
                original_value = parameters.WIND_TURBINE_COST
                parameters.WIND_TURBINE_COST = value
                obj_funcs.WIND_TURBINE_COST = value
            elif param_name == "Battery Cost":
                original_value = parameters.BATTERY_COST
                parameters.BATTERY_COST = value
                obj_funcs.BATTERY_COST = value
            elif param_name == "Solar Panel Capacity":
                original_value = parameters.SOLAR_PANEL_CAPACITY
                parameters.SOLAR_PANEL_CAPACITY = value
                production.SOLAR_PANEL_CAPACITY = value
            elif param_name == "Wind Turbine Capacity":
                original_value = parameters.WIND_TURBINE_CAPACITY
                parameters.WIND_TURBINE_CAPACITY = value
                production.WIND_TURBINE_CAPACITY = value
            elif param_name == "Solar Panel Emissions":
                original_value = parameters.SOLAR_PANEL_LIFECYCLE_EMISSIONS
                parameters.SOLAR_PANEL_LIFECYCLE_EMISSIONS = value
                obj_funcs.SOLAR_PANEL_LIFECYCLE_EMISSIONS = value
            elif param_name == "Wind Turbine Emissions":
                original_value = parameters.WIND_TURBINE_LIFECYCLE_EMISSIONS
                parameters.WIND_TURBINE_LIFECYCLE_EMISSIONS = value
                obj_funcs.WIND_TURBINE_LIFECYCLE_EMISSIONS = value
            elif param_name == "Battery Emissions":
                original_value = parameters.BATTERY_LIFECYCLE_EMISSIONS
                parameters.BATTERY_LIFECYCLE_EMISSIONS = value
                obj_funcs.BATTERY_LIFECYCLE_EMISSIONS = value
            
            # Calculate metrics with modified parameter
            cost, reliability, env_impact = evaluate_func(base_individual)
            
            # Store results
            results[param_name]["cost"].append(cost)
            results[param_name]["reliability"].append(reliability)
            results[param_name]["env_impact"].append(env_impact)
            
            # Restore original value
            if param_name == "Solar Panel Cost":
                parameters.SOLAR_PANEL_COST = original_value
                obj_funcs.SOLAR_PANEL_COST = original_value
            elif param_name == "Wind Turbine Cost":
                parameters.WIND_TURBINE_COST = original_value
                obj_funcs.WIND_TURBINE_COST = original_value
            elif param_name == "Battery Cost":
                parameters.BATTERY_COST = original_value
                obj_funcs.BATTERY_COST = original_value
            elif param_name == "Solar Panel Capacity":
                parameters.SOLAR_PANEL_CAPACITY = original_value
                production.SOLAR_PANEL_CAPACITY = original_value
            elif param_name == "Wind Turbine Capacity":
                parameters.WIND_TURBINE_CAPACITY = original_value
                production.WIND_TURBINE_CAPACITY = original_value
            elif param_name == "Solar Panel Emissions":
                parameters.SOLAR_PANEL_LIFECYCLE_EMISSIONS = original_value
                obj_funcs.SOLAR_PANEL_LIFECYCLE_EMISSIONS = original_value
            elif param_name == "Wind Turbine Emissions":
                parameters.WIND_TURBINE_LIFECYCLE_EMISSIONS = original_value
                obj_funcs.WIND_TURBINE_LIFECYCLE_EMISSIONS = original_value
            elif param_name == "Battery Emissions":
                parameters.BATTERY_LIFECYCLE_EMISSIONS = original_value
                obj_funcs.BATTERY_LIFECYCLE_EMISSIONS = original_value
    
    # Visualize sensitivity results
    # Create separate figures for each metric
    metrics = ["cost", "reliability", "env_impact"]
    metric_names = ["Cost ($)", "Reliability", "Environmental Impact (kg CO2/kWh)"]
    
    for i, metric in enumerate(metrics):
        plt.figure(figsize=(15, 8))
        
        for param_name, param_data in results.items():
            # Normalize parameter values for x-axis
            param_range_norm = [val / param_data["values"][0] for val in param_data["values"]]
            
            # Normalize metric values for y-axis
            if metric == "reliability":
                # For reliability, higher is better
                metric_norm = [val / results[param_name][metric][0] for val in results[param_name][metric]]
            else:
                # For cost and env_impact, lower is better
                metric_norm = [results[param_name][metric][0] / val for val in results[param_name][metric]]
            
            plt.plot(param_range_norm, metric_norm, 'o-', label=param_name, linewidth=2)
        
        plt.axhline(y=1, color='k', linestyle='--', alpha=0.5)
        plt.axvline(x=1, color='k', linestyle='--', alpha=0.5)
        plt.xlabel('Parameter Value (Relative to Base Case)')
        plt.ylabel(f'Relative {metric_names[i]} (higher is better)')
        plt.title(f'Sensitivity of {metric_names[i]} to Parameter Changes')
        plt.grid(True, alpha=0.3)
        plt.legend()
        
        plt.tight_layout()
        if save_path:
            plt.savefig(f"{save_path}_sensitivity_{metric}.png", dpi=300, bbox_inches='tight')
        plt.show()
    
    return results