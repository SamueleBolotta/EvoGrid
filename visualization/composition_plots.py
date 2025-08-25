"""
Technology composition visualization for energy systems.

Contains functions for visualizing the technology mix and composition
of renewable energy system solutions.
"""

import matplotlib.pyplot as plt
import numpy as np
from energy_system.parameters import SOLAR_PANEL_CAPACITY, WIND_TURBINE_CAPACITY, BATTERY_CAPACITY


def visualize_solutions_composition(pareto_solutions, save_path=None):
    """Visualize the composition of the Pareto optimal solutions."""
    # Plot technology mix for each solution
    plt.figure(figsize=(15, 10))
    
    # Sort solutions by cost for better visualization
    pareto_solutions_sorted = pareto_solutions.sort_values('cost')
    
    # Stacked bar chart of system components
    solar_capacity = pareto_solutions_sorted['solar_panels'] * SOLAR_PANEL_CAPACITY
    wind_capacity = pareto_solutions_sorted['wind_turbines'] * WIND_TURBINE_CAPACITY
    battery_capacity = pareto_solutions_sorted['batteries'] * BATTERY_CAPACITY
    
    solution_ids = pareto_solutions_sorted['id']
    
    plt.subplot(211)
    plt.bar(solution_ids, solar_capacity, label='Solar Capacity (kW)', color='#FFD700', alpha=0.7)
    plt.bar(solution_ids, wind_capacity, bottom=solar_capacity, label='Wind Capacity (kW)', color='#87CEEB', alpha=0.7)
    
    plt.xlabel('Solution ID')
    plt.ylabel('Generation Capacity (kW)')
    plt.title('Generation Mix of Pareto Optimal Solutions')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.subplot(212)
    plt.bar(solution_ids, battery_capacity, label='Battery Capacity (kWh)', color='#32CD32', alpha=0.7)
    
    plt.xlabel('Solution ID')
    plt.ylabel('Storage Capacity (kWh)')
    plt.title('Storage Capacity of Pareto Optimal Solutions')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(f"{save_path}_composition.png", dpi=300, bbox_inches='tight')
    
    # Plot technology proportions
    plt.figure(figsize=(15, 8))
    
    total_capacity = solar_capacity + wind_capacity
    solar_proportion = (solar_capacity / total_capacity) * 100
    wind_proportion = (wind_capacity / total_capacity) * 100
    
    plt.subplot(121)
    plt.scatter(pareto_solutions_sorted['cost'], pareto_solutions_sorted['reliability'], 
                c=solar_proportion, cmap='YlOrRd', s=100, alpha=0.8)
    plt.colorbar(label='Solar Proportion (%)')
    plt.xlabel('Cost ($)')
    plt.ylabel('Reliability')
    plt.title('Solution Performance by Solar Proportion')
    plt.grid(True, alpha=0.3)
    
    plt.subplot(122)
    plt.scatter(pareto_solutions_sorted['environmental_impact'], pareto_solutions_sorted['reliability'], 
                c=battery_capacity, cmap='Greens', s=100, alpha=0.8)
    plt.colorbar(label='Battery Capacity (kWh)')
    plt.xlabel('Emissions (kg CO2/kWh)')
    plt.ylabel('Reliability')
    plt.title('Solution Performance by Battery Capacity')
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(f"{save_path}_technology_influence.png", dpi=300, bbox_inches='tight')
    
    plt.show()
