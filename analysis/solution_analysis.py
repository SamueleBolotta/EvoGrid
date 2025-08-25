"""
Detailed solution analysis for energy systems.

Contains functions for performing detailed analysis of individual solutions
including energy balance, deficit analysis, and performance metrics.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import os
from energy_system.simulation import simulate_energy_system
from energy_system.production import calculate_solar_energy_production, calculate_wind_energy_production
from energy_system.objective_functions import calculate_total_cost, calculate_reliability, calculate_environmental_impact
from energy_system.parameters import BATTERY_CAPACITY


def analyze_solution_detail(solution, daily_data, save_dir=None):
    """
    Perform detailed analysis of a single solution.
    
    Args:
        solution: Dictionary with solution details
        daily_data: Tuple of (solar_irradiance, wind_speed, energy_demand)
        save_dir: Directory to save plots
    """
    daily_solar_irradiance, daily_wind_speed, daily_energy_demand = daily_data
    
    # Extract parameters
    name = solution["name"]
    num_solar_panels = solution["solar_panels"]
    num_wind_turbines = solution["wind_turbines"]
    num_batteries = solution["batteries"]
    
    # Create individual array
    individual = [num_solar_panels, num_wind_turbines, num_batteries]
    
    # Calculate key metrics
    cost = calculate_total_cost(individual)
    reliability = calculate_reliability(individual, daily_solar_irradiance, daily_wind_speed, daily_energy_demand)
    env_impact = calculate_environmental_impact(individual, daily_solar_irradiance, daily_wind_speed)
    
    # Run simulation
    sim_results = simulate_energy_system(individual, daily_solar_irradiance, daily_wind_speed, daily_energy_demand)
    
    # Calculate energy production by source
    solar_energy = np.array([calculate_solar_energy_production(num_solar_panels, irr) 
                             for irr in daily_solar_irradiance])
    wind_energy = np.array([calculate_wind_energy_production(num_wind_turbines, ws) 
                           for ws in daily_wind_speed])
    
    # Calculate summary statistics
    total_demand = daily_energy_demand.sum()
    total_solar = solar_energy.sum()
    total_wind = wind_energy.sum()
    total_deficit = sim_results['energy_deficit'].sum()
    total_curtailed = sim_results['curtailed_energy'].sum()
    
    days_with_deficit = sum(1 for deficit in sim_results['energy_deficit'] if deficit > 0)
    max_deficit_day = np.argmax(sim_results['energy_deficit'])
    
    # Print summary
    print(f"\n{'='*80}")
    print(f"DETAILED ANALYSIS FOR: {name}")
    print(f"{'='*80}")
    print(f"Configuration: {num_solar_panels} Solar Panels, {num_wind_turbines} Wind Turbines, {num_batteries} Batteries")
    print(f"Cost: ${cost:.2f}")
    print(f"Reliability: {reliability:.4f} ({100*reliability:.2f}%)")
    print(f"Environmental Impact: {env_impact:.6f} kg CO2/kWh")
    print(f"\nSummary Statistics:")
    print(f"  • Total Annual Energy Demand: {total_demand:.2f} kWh")
    print(f"  • Solar Energy Production: {total_solar:.2f} kWh ({100*total_solar/total_demand:.2f}% of demand)")
    print(f"  • Wind Energy Production: {total_wind:.2f} kWh ({100*total_wind/total_demand:.2f}% of demand)")
    print(f"  • Total Energy Deficit: {total_deficit:.2f} kWh ({100*total_deficit/total_demand:.2f}% of demand)")
    print(f"  • Curtailed Energy: {total_curtailed:.2f} kWh ({100*total_curtailed/(total_solar+total_wind):.2f}% of production)")
    print(f"  • Days with Energy Deficit: {days_with_deficit} days ({100*days_with_deficit/365:.2f}% of year)")
    if days_with_deficit > 0:
        print(f"  • Worst Day: Day {max_deficit_day} with {sim_results['energy_deficit'][max_deficit_day]:.2f} kWh deficit")
    print(f"  • Battery Capacity: {num_batteries * BATTERY_CAPACITY:.2f} kWh")
    print(f"  • Average Battery State: {sim_results['battery_state'].mean():.2f} kWh " + 
          f"({100*sim_results['battery_state'].mean()/(num_batteries * BATTERY_CAPACITY + 1e-10):.2f}% of capacity)")
    
    # Create figure with multiple subplots
    fig = plt.figure(figsize=(15, 12))
    gs = GridSpec(3, 2, figure=fig)
    
    # Energy balance over the year
    ax1 = fig.add_subplot(gs[0, :])
    days = np.arange(len(daily_energy_demand))
    
    # Plot energy demand, production, and battery state
    ax1.plot(days, daily_energy_demand, 'k-', label='Energy Demand', alpha=0.7)
    ax1.plot(days, solar_energy, 'y-', label='Solar Production', alpha=0.7)
    ax1.plot(days, wind_energy, 'b-', label='Wind Production', alpha=0.7)
    ax1.plot(days, solar_energy + wind_energy, 'g-', label='Total Production', alpha=0.7)
    if num_batteries > 0:
        battery_plot = ax1.plot(days, sim_results['battery_state'], 'r-', label='Battery State (kWh)', alpha=0.7)
        # Add second y-axis for battery state
        ax1_twin = ax1.twinx()
        ax1_twin.set_ylim(0, num_batteries * BATTERY_CAPACITY)
        ax1_twin.set_ylabel('Battery State (kWh)')
        # Set color of twin y-axis to match the battery plot
        ax1_twin.spines['right'].set_color(battery_plot[0].get_color())
        ax1_twin.tick_params(axis='y', colors=battery_plot[0].get_color())
    
    ax1.set_xlabel('Day of Year')
    ax1.set_ylabel('Energy (kWh/day)')
    ax1.set_title(f'{name} - Energy Balance Over Year')
    ax1.grid(True, alpha=0.3)
    ax1.legend(loc='upper right')
    
    # Energy deficit distribution
    ax2 = fig.add_subplot(gs[1, 0])
    deficit_days = np.where(sim_results['energy_deficit'] > 0)[0]
    if len(deficit_days) > 0:
        ax2.bar(deficit_days, sim_results['energy_deficit'][deficit_days], color='r', alpha=0.7)
        ax2.set_xlabel('Day of Year')
        ax2.set_ylabel('Energy Deficit (kWh)')
        ax2.set_title('Days with Energy Deficit')
        ax2.grid(True, alpha=0.3)
    else:
        ax2.text(0.5, 0.5, 'No Energy Deficit!', horizontalalignment='center',
                 verticalalignment='center', transform=ax2.transAxes, fontsize=14)
        ax2.set_title('Energy Deficit Analysis')
    
    # Curtailed energy distribution
    ax3 = fig.add_subplot(gs[1, 1])
    curtailed_days = np.where(sim_results['curtailed_energy'] > 0)[0]
    if len(curtailed_days) > 0:
        ax3.bar(curtailed_days, sim_results['curtailed_energy'][curtailed_days], color='g', alpha=0.7)
        ax3.set_xlabel('Day of Year')
        ax3.set_ylabel('Curtailed Energy (kWh)')
        ax3.set_title('Days with Curtailed Energy')
        ax3.grid(True, alpha=0.3)
    else:
        ax3.text(0.5, 0.5, 'No Curtailed Energy!', horizontalalignment='center',
                 verticalalignment='center', transform=ax3.transAxes, fontsize=14)
        ax3.set_title('Curtailed Energy Analysis')
    
    # Monthly energy balance
    ax4 = fig.add_subplot(gs[2, 0])
    # Group by month
    months = np.array([i // 30 for i in range(len(daily_energy_demand))])
    monthly_demand = np.array([daily_energy_demand[months == i].sum() for i in range(12)])
    monthly_solar = np.array([solar_energy[months == i].sum() for i in range(12)])
    monthly_wind = np.array([wind_energy[months == i].sum() for i in range(12)])
    monthly_deficit = np.array([sim_results['energy_deficit'][months == i].sum() for i in range(12)])
    
    month_names = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
    bar_width = 0.35
    x = np.arange(len(month_names))
    
    ax4.bar(x, monthly_demand, bar_width, label='Demand', color='gray', alpha=0.7)
    ax4.bar(x, monthly_solar, bar_width, bottom=0, label='Solar', color='gold', alpha=0.7)
    ax4.bar(x, monthly_wind, bar_width, bottom=monthly_solar, label='Wind', color='skyblue', alpha=0.7)
    
    ax4.set_xlabel('Month')
    ax4.set_ylabel('Energy (kWh)')
    ax4.set_title('Monthly Energy Production vs. Demand')
    ax4.set_xticks(x)
    ax4.set_xticklabels(month_names)
    ax4.grid(True, alpha=0.3)
    ax4.legend()
    
    # Energy production distribution
    ax5 = fig.add_subplot(gs[2, 1])
    energy_sources = ['Solar', 'Wind']
    energy_values = [total_solar, total_wind]
    colors = ['gold', 'skyblue']
    
    ax5.pie(energy_values, labels=energy_sources, colors=colors, autopct='%1.1f%%', startangle=90)
    ax5.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle
    ax5.set_title('Energy Production Distribution')
    
    plt.tight_layout()
    filename = f"{name.lower().replace(' ', '_')}_detailed_analysis.png"
    if save_dir:
        if not os.path.isdir(save_dir):
            os.makedirs(save_dir, exist_ok=True)
        out_path = os.path.join(save_dir, filename)
    else:
        out_path = filename
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.show()

    return {"name": name, "cost": cost, "reliability": reliability, "env_impact": env_impact, 
            "solar_energy": total_solar, "wind_energy": total_wind, "deficit": total_deficit, 
            "curtailed": total_curtailed, "deficit_days": days_with_deficit}
