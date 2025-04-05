import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import seaborn as sns
import math

# Note: These imports will be accessed dynamically when this module is imported
# We're listing them here for clarity, but they'll be accessed via the main module
# Parameters, environmental data, and functions will be imported from main.py

def analyze_results(population, pareto_front):
    """Analyze the optimization results."""
    # Print Pareto optimal solutions
    print("\nPareto Optimal Solutions:")
    print("Solar Panels | Wind Turbines | Batteries | Cost ($) | Reliability | Emissions (kg CO2/kWh)")
    print("-" * 95)
    
    pareto_solutions = []
    for i, ind in enumerate(pareto_front):
        cost, reliability, env_impact = ind.fitness.values
        pareto_solutions.append({
            'id': i,
            'solar_panels': ind[0],
            'wind_turbines': ind[1],
            'batteries': ind[2],
            'cost': cost,
            'reliability': reliability,
            'environmental_impact': env_impact
        })
        print(f"{ind[0]:12} | {ind[1]:13} | {ind[2]:9} | {cost:8.0f} | {reliability:10.4f} | {env_impact:8.4f}")
    
    return pd.DataFrame(pareto_solutions)

def visualize_pareto_front(pareto_solutions, save_path=None):
    """Visualize the Pareto front in 3D and as pairwise scatter plots."""
    costs = pareto_solutions['cost']
    reliability = pareto_solutions['reliability']
    env_impact = pareto_solutions['environmental_impact']
    
    # 3D Pareto front
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection='3d')
    scatter = ax.scatter(costs, reliability, env_impact, c=reliability, cmap='viridis', s=100, alpha=0.7)
    
    ax.set_xlabel('Cost ($)')
    ax.set_ylabel('Reliability')
    ax.set_zlabel('Emissions (kg CO2/kWh)')
    ax.set_title('Pareto Front for Energy System Optimization')
    
    cbar = plt.colorbar(scatter)
    cbar.set_label('Reliability')
    
    if save_path:
        plt.savefig(f"{save_path}_pareto3d.png", dpi=300, bbox_inches='tight')
    
    plt.figure(figsize=(18, 6))
    
    # Cost vs. Reliability
    plt.subplot(131)
    plt.scatter(costs, reliability, c=env_impact, cmap='plasma', s=80, alpha=0.7)
    plt.xlabel('Cost ($)')
    plt.ylabel('Reliability')
    plt.colorbar(label='Emissions (kg CO2/kWh)')
    plt.title('Cost vs. Reliability')
    plt.grid(True, alpha=0.3)
    
    # Cost vs. Environmental Impact
    plt.subplot(132)
    plt.scatter(costs, env_impact, c=reliability, cmap='viridis', s=80, alpha=0.7)
    plt.xlabel('Cost ($)')
    plt.ylabel('Emissions (kg CO2/kWh)')
    plt.colorbar(label='Reliability')
    plt.title('Cost vs. Environmental Impact')
    plt.grid(True, alpha=0.3)
    
    # Reliability vs. Environmental Impact
    plt.subplot(133)
    plt.scatter(reliability, env_impact, c=costs, cmap='copper', s=80, alpha=0.7)
    plt.xlabel('Reliability')
    plt.ylabel('Emissions (kg CO2/kWh)')
    plt.colorbar(label='Cost ($)')
    plt.title('Reliability vs. Environmental Impact')
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(f"{save_path}_pareto2d.png", dpi=300, bbox_inches='tight')
    
    plt.show()

def perform_sensitivity_analysis(base_solution, save_path=None):
    """Perform sensitivity analysis on key parameters."""
    # Import necessary modules and variables from main
    import main
    
    # Extract parameters from base solution
    base_solar_panels = base_solution["solar_panels"]
    base_wind_turbines = base_solution["wind_turbines"]
    base_batteries = base_solution["batteries"]
    base_individual = [base_solar_panels, base_wind_turbines, base_batteries]
    
    # Calculate base metrics
    base_cost = main.calculate_total_cost(base_individual)
    base_reliability = main.calculate_reliability(base_individual, main.daily_solar_irradiance, 
                                            main.daily_wind_speed, main.daily_energy_demand)
    base_env_impact = main.calculate_environmental_impact(base_individual)
    
    print(f"\n{'='*80}")
    print(f"SENSITIVITY ANALYSIS FOR: {base_solution['name']}")
    print(f"{'='*80}")
    print(f"Base configuration: {base_solar_panels} Solar Panels, {base_wind_turbines} Wind Turbines, {base_batteries} Batteries")
    print(f"Base metrics: Cost=${base_cost:.2f}, Reliability={base_reliability:.4f}, Env. Impact={base_env_impact:.6f} kg CO2/kWh")
    
    # Define parameter ranges for sensitivity analysis
    parameter_ranges = {
        "Solar Panel Cost": np.linspace(main.SOLAR_PANEL_COST * 0.5, main.SOLAR_PANEL_COST * 1.5, 10),
        "Wind Turbine Cost": np.linspace(main.WIND_TURBINE_COST * 0.5, main.WIND_TURBINE_COST * 1.5, 10),
        "Battery Cost": np.linspace(main.BATTERY_COST * 0.5, main.BATTERY_COST * 1.5, 10),
        "Solar Panel Capacity": np.linspace(main.SOLAR_PANEL_CAPACITY * 0.8, main.SOLAR_PANEL_CAPACITY * 1.2, 10),
        "Wind Turbine Capacity": np.linspace(main.WIND_TURBINE_CAPACITY * 0.8, main.WIND_TURBINE_CAPACITY * 1.2, 10),
        "Solar Panel Emissions": np.linspace(main.SOLAR_PANEL_LIFECYCLE_EMISSIONS * 0.5, main.SOLAR_PANEL_LIFECYCLE_EMISSIONS * 1.5, 10),
        "Wind Turbine Emissions": np.linspace(main.WIND_TURBINE_LIFECYCLE_EMISSIONS * 0.5, main.WIND_TURBINE_LIFECYCLE_EMISSIONS * 1.5, 10)
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
            
            # Modify parameter based on name
            if param_name == "Solar Panel Cost":
                original_value = main.SOLAR_PANEL_COST
                main.SOLAR_PANEL_COST = value
            elif param_name == "Wind Turbine Cost":
                original_value = main.WIND_TURBINE_COST
                main.WIND_TURBINE_COST = value
            elif param_name == "Battery Cost":
                original_value = main.BATTERY_COST
                main.BATTERY_COST = value
            elif param_name == "Solar Panel Capacity":
                original_value = main.SOLAR_PANEL_CAPACITY
                main.SOLAR_PANEL_CAPACITY = value
            elif param_name == "Wind Turbine Capacity":
                original_value = main.WIND_TURBINE_CAPACITY
                main.WIND_TURBINE_CAPACITY = value
            elif param_name == "Solar Panel Emissions":
                original_value = main.SOLAR_PANEL_LIFECYCLE_EMISSIONS
                main.SOLAR_PANEL_LIFECYCLE_EMISSIONS = value
            elif param_name == "Wind Turbine Emissions":
                original_value = main.WIND_TURBINE_LIFECYCLE_EMISSIONS
                main.WIND_TURBINE_LIFECYCLE_EMISSIONS = value
            elif param_name == "Battery Emissions":
                original_value = main.BATTERY_LIFECYCLE_EMISSIONS
                main.BATTERY_LIFECYCLE_EMISSIONS = value
            
            # Calculate metrics with modified parameter
            cost = main.calculate_total_cost(base_individual)
            reliability = main.calculate_reliability(base_individual, main.daily_solar_irradiance, 
                                         main.daily_wind_speed, main.daily_energy_demand)
            env_impact = main.calculate_environmental_impact(base_individual)
            
            # Store results
            results[param_name]["cost"].append(cost)
            results[param_name]["reliability"].append(reliability)
            results[param_name]["env_impact"].append(env_impact)
            
            # Restore original value
            if param_name == "Solar Panel Cost":
                main.SOLAR_PANEL_COST = original_value
            elif param_name == "Wind Turbine Cost":
                main.WIND_TURBINE_COST = original_value
            elif param_name == "Battery Cost":
                main.BATTERY_COST = original_value
            elif param_name == "Solar Panel Capacity":
                main.SOLAR_PANEL_CAPACITY = original_value
            elif param_name == "Wind Turbine Capacity":
                main.WIND_TURBINE_CAPACITY = original_value
            elif param_name == "Solar Panel Emissions":
                main.SOLAR_PANEL_LIFECYCLE_EMISSIONS = original_value
            elif param_name == "Wind Turbine Emissions":
                main.WIND_TURBINE_LIFECYCLE_EMISSIONS = original_value
            elif param_name == "Battery Emissions":
                main.BATTERY_LIFECYCLE_EMISSIONS = original_value
    
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

def analyze_solution_detail(solution):
    """Perform detailed analysis of a single solution."""
    # Import necessary modules and variables from main
    import main
    
    # Extract parameters
    name = solution["name"]
    num_solar_panels = solution["solar_panels"]
    num_wind_turbines = solution["wind_turbines"]
    num_batteries = solution["batteries"]
    
    # Create individual array
    individual = [num_solar_panels, num_wind_turbines, num_batteries]
    
    # Calculate key metrics
    cost = main.calculate_total_cost(individual)
    reliability = main.calculate_reliability(individual, main.daily_solar_irradiance, 
                                       main.daily_wind_speed, main.daily_energy_demand)
    env_impact = main.calculate_environmental_impact(individual)
    
    # Run simulation
    sim_results = main.simulate_energy_system(individual, main.daily_solar_irradiance, 
                                         main.daily_wind_speed, main.daily_energy_demand)
    
    # Calculate energy production by source
    solar_energy = np.array([main.calculate_solar_energy_production(num_solar_panels, irr) 
                             for irr in main.daily_solar_irradiance])
    wind_energy = np.array([main.calculate_wind_energy_production(num_wind_turbines, ws) 
                           for ws in main.daily_wind_speed])
    
    # Calculate summary statistics
    total_demand = main.daily_energy_demand.sum()
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
    print(f"  • Battery Capacity: {num_batteries * main.BATTERY_CAPACITY:.2f} kWh")
    print(f"  • Average Battery State: {sim_results['battery_state'].mean():.2f} kWh " + 
          f"({100*sim_results['battery_state'].mean()/(num_batteries * main.BATTERY_CAPACITY + 1e-10):.2f}% of capacity)")
    
    # Create figure with multiple subplots
    fig = plt.figure(figsize=(15, 12))
    gs = GridSpec(3, 2, figure=fig)
    
    # Energy balance over the year
    ax1 = fig.add_subplot(gs[0, :])
    days = np.arange(len(main.daily_energy_demand))
    
    # Plot energy demand, production, and battery state
    ax1.plot(days, main.daily_energy_demand, 'k-', label='Energy Demand', alpha=0.7)
    ax1.plot(days, solar_energy, 'y-', label='Solar Production', alpha=0.7)
    ax1.plot(days, wind_energy, 'b-', label='Wind Production', alpha=0.7)
    ax1.plot(days, solar_energy + wind_energy, 'g-', label='Total Production', alpha=0.7)
    if num_batteries > 0:
        battery_plot = ax1.plot(days, sim_results['battery_state'], 'r-', label='Battery State (kWh)', alpha=0.7)
        # Add second y-axis for battery state
        ax1_twin = ax1.twinx()
        ax1_twin.set_ylim(0, num_batteries * main.BATTERY_CAPACITY)
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
    months = np.array([i // 30 for i in range(len(main.daily_energy_demand))])
    monthly_demand = np.array([main.daily_energy_demand[months == i].sum() for i in range(12)])
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
    plt.savefig(f"{name.lower().replace(' ', '_')}_detailed_analysis.png", dpi=300, bbox_inches='tight')
    plt.show()
    
    # Create comparative visualization of all solutions
    plt.figure(figsize=(15, 10))
    
    return {"name": name, "cost": cost, "reliability": reliability, "env_impact": env_impact, 
            "solar_energy": total_solar, "wind_energy": total_wind, "deficit": total_deficit, 
            "curtailed": total_curtailed, "deficit_days": days_with_deficit}

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
