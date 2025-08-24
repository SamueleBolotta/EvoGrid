import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import seaborn as sns
from deap import base, creator, tools, algorithms
import random
import math
from scipy.stats import norm
from tqdm import tqdm
import argparse
import os
import datetime

from utils import (
    analyze_results,
    visualize_pareto_front,
    visualize_solutions_composition,
    perform_sensitivity_analysis,
    analyze_solution_detail
)

# Set random seed for reproducibility
np.random.seed(42)
random.seed(42)

# ------- Energy system parameters -------

# Solar panel parameters
SOLAR_PANEL_CAPACITY = 0.4  # kW per panel
SOLAR_PANEL_COST = 300      # $ per panel
SOLAR_PANEL_LAND_USE = 3    # m² per panel
SOLAR_PANEL_LIFECYCLE_EMISSIONS = 40  # kg CO2 per panel

# Wind turbine parameters
WIND_TURBINE_CAPACITY = 3.0   # kW per turbine
WIND_TURBINE_COST = 6000      # $ per turbine 
WIND_TURBINE_LAND_USE = 100   # m² per turbine
WIND_TURBINE_LIFECYCLE_EMISSIONS = 800  # kg CO2 per turbine

# Battery storage parameters
BATTERY_CAPACITY = 5.0   # kWh per battery unit
BATTERY_COST = 2000      # $ per battery unit
BATTERY_CYCLE_LIFE = 4000  # cycles before replacement
BATTERY_LIFECYCLE_EMISSIONS = 150  # kg CO2 per battery unit

# System constraints
MAX_LAND_AREA = 10000    # m²
MAX_BUDGET = 200000      # $
MIN_ENERGY_REQUIREMENT = 50  # kWh/day
PROJECT_LIFETIME = 20    # years

# ------- Weather and demand data generation -------

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

# ------- Energy system modeling -------

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

# ------- Objective functions -------

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

def calculate_environmental_impact(individual):
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
    avg_daily_solar = calculate_solar_energy_production(num_solar_panels, np.mean(daily_solar_irradiance))
    avg_daily_wind = calculate_wind_energy_production(num_wind_turbines, np.mean(daily_wind_speed))
    total_energy_produced = (avg_daily_solar + avg_daily_wind) * 365 * PROJECT_LIFETIME
    
    if total_energy_produced > 0:
        emissions_intensity = total_emissions / total_energy_produced  # kg CO2 / kWh
    else:
        emissions_intensity = float('inf')
    
    return emissions_intensity

def check_constraints(individual):
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
    avg_daily_solar = calculate_solar_energy_production(num_solar_panels, np.mean(daily_solar_irradiance))
    avg_daily_wind = calculate_wind_energy_production(num_wind_turbines, np.mean(daily_wind_speed))
    avg_daily_production = avg_daily_solar + avg_daily_wind
    energy_constraint_satisfied = avg_daily_production >= MIN_ENERGY_REQUIREMENT
    
    return land_constraint_satisfied and budget_constraint_satisfied and energy_constraint_satisfied

# ------- NSGA-II setup -------

# Create fitness and individual classes
creator.create("FitnessMulti", base.Fitness, weights=(-1.0, 1.0, -1.0))  # Minimize cost, maximize reliability, minimize environmental impact
creator.create("Individual", list, fitness=creator.FitnessMulti)

# Initialize toolbox
toolbox = base.Toolbox()

# Define bounds for each decision variable
BOUNDS = [(0, 500), (0, 50), (0, 50)]  # (solar panels, wind turbines, batteries)

# Register the individual and population creation functions
def create_individual():
    return creator.Individual([
        random.randint(BOUNDS[0][0], BOUNDS[0][1]),  # num_solar_panels
        random.randint(BOUNDS[1][0], BOUNDS[1][1]),  # num_wind_turbines
        random.randint(BOUNDS[2][0], BOUNDS[2][1])   # num_batteries
    ])

toolbox.register("individual", create_individual)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)

# Register the evaluation function
def evaluate(individual):
    """Evaluate an individual and return a tuple of objective values."""
    # Check constraints first
    if not check_constraints(individual):
        return (float('inf'), 0.0, float('inf'))  # Heavily penalize infeasible solutions
    
    # Calculate objective values
    cost = calculate_total_cost(individual)
    reliability = calculate_reliability(individual, daily_solar_irradiance, daily_wind_speed, daily_energy_demand)
    environmental_impact = calculate_environmental_impact(individual)
    
    return (cost, reliability, environmental_impact)

toolbox.register("evaluate", evaluate)

# Register the genetic operators
toolbox.register("mate", tools.cxTwoPoint)
toolbox.register("mutate", tools.mutUniformInt, low=[BOUNDS[i][0] for i in range(3)], 
                 up=[BOUNDS[i][1] for i in range(3)], indpb=0.2)
toolbox.register("select", tools.selNSGA2)

# ------- Run Optimization -------

def run_nsga2(pop_size=100, num_generations=50, cxpb=0.7, mutpb=0.2):
    """Run the NSGA-II algorithm."""
    # Initialize statistics
    stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register("avg", np.mean, axis=0)
    stats.register("min", np.min, axis=0)
    stats.register("max", np.max, axis=0)
    
    # Create initial population
    population = toolbox.population(n=pop_size)
    
    # Evaluate initial population
    fitnesses = list(map(toolbox.evaluate, population))
    for ind, fit in zip(population, fitnesses):
        ind.fitness.values = fit
    
    # Extract hall of fame (Pareto front)
    hof = tools.ParetoFront()
    
    # Run the algorithm
    print("Starting NSGA-II optimization...")
    population, logbook = algorithms.eaSimple(
        population, 
        toolbox,
        cxpb=cxpb,   # Crossover probability
        mutpb=mutpb,  # Mutation probability
        ngen=num_generations,
        stats=stats,
        halloffame=hof,
        verbose=True
    )
    
    return population, logbook, hof

# ------- Simple experiment I/O helpers -------

def _ensure_dir(path: str):
    if path and not os.path.isdir(path):
        os.makedirs(path, exist_ok=True)


def save_run_outputs(pareto_df, logbook, out_dir: str, config_path: str = None):
    """Save structured outputs for an experiment run.

    - pareto.csv: solutions on the Pareto front
    - logbook.csv: DEAP logbook stats (best-effort)
    - config.used.yaml: copy of the YAML used for reproducibility
    """
    _ensure_dir(out_dir)
    pareto_csv = os.path.join(out_dir, "pareto.csv")
    pareto_df.to_csv(pareto_csv, index=False)

    # Save logbook if convertible
    try:
        pd.DataFrame(logbook).to_csv(os.path.join(out_dir, "logbook.csv"), index=False)
    except Exception:
        pass

    if config_path and os.path.isfile(config_path):
        try:
            import shutil
            shutil.copy(config_path, os.path.join(out_dir, "config.used.yaml"))
        except Exception:
            pass

    return out_dir


def load_data_from_config(cfg: dict):
    """Load environmental data based on config dict.

    Supports synthetic generation (default) and simple CSV ingestion with
    columns: irradiance, wind_speed, demand (hourly). Aggregates to daily.
    """
    data_cfg = (cfg or {}).get("data", {})
    source = data_cfg.get("source", "synthetic")
    aggregate = data_cfg.get("aggregate_to_daily", True)

    if source == "csv":
        csv_path = data_cfg.get("csv_path")
        if not csv_path:
            raise ValueError("data.source is 'csv' but data.csv_path is not set in config")
        df = pd.read_csv(csv_path)
        irr = df["irradiance"].values
        wind = df["wind_speed"].values
        dem = df["demand"].values
        if not aggregate:
            raise ValueError("Hourly simulation not implemented; set aggregate_to_daily: true")
        daily_irr = np.array([irr[i:i+24].mean() for i in range(0, len(irr), 24)])
        daily_wind = np.array([wind[i:i+24].mean() for i in range(0, len(wind), 24)])
        daily_dem = np.array([dem[i:i+24].sum() for i in range(0, len(dem), 24)])
        return daily_irr, daily_wind, daily_dem

    # Default: synthetic
    days = int(data_cfg.get("days", 365))
    h_irr = generate_hourly_solar_irradiance(days)
    h_wind = generate_hourly_wind_speed(days)
    h_dem = generate_hourly_energy_demand(days)
    daily_irr = np.array([h_irr[i:i+24].mean() for i in range(0, len(h_irr), 24)])
    daily_wind = np.array([h_wind[i:i+24].mean() for i in range(0, len(h_wind), 24)])
    daily_dem = np.array([h_dem[i:i+24].sum() for i in range(0, len(h_dem), 24)])
    return daily_irr, daily_wind, daily_dem

# Generate environmental data
days_to_simulate = 365  # One year
hourly_solar_irradiance = generate_hourly_solar_irradiance(days_to_simulate)
hourly_wind_speed = generate_hourly_wind_speed(days_to_simulate)
hourly_energy_demand = generate_hourly_energy_demand(days_to_simulate)

# Convert to daily data for faster simulation
daily_solar_irradiance = np.array([hourly_solar_irradiance[i:i+24].mean() for i in range(0, len(hourly_solar_irradiance), 24)])
daily_wind_speed = np.array([hourly_wind_speed[i:i+24].mean() for i in range(0, len(hourly_wind_speed), 24)])
daily_energy_demand = np.array([hourly_energy_demand[i:i+24].sum() for i in range(0, len(hourly_energy_demand), 24)])

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="EvoGrid NSGA-II experiments")
    parser.add_argument("--config", type=str, default="config.yaml", help="Path to YAML config file")
    parser.add_argument("--output-dir", type=str, default=None, help="Directory to save outputs (optional)")
    parser.add_argument("--csv-path", type=str, default=None, help="Optional CSV path for data ingestion")
    args = parser.parse_args()

    # Load YAML config (if present)
    cfg = {}
    if args.config and os.path.isfile(args.config):
        try:
            from config import load_config, apply_to_main
            cfg = load_config(args.config)
            # Apply config to globals
            apply_to_main(cfg, main_mod=__import__(__name__))
        except Exception as e:
            print(f"Warning: failed to load config '{args.config}': {e}")

    # Parameters are taken exclusively from YAML now
    tag = cfg.get("tag", "run")
    seed = cfg.get("seed", 42)
    nsga_cfg = cfg.get("nsga", {})
    pop_size = int(nsga_cfg.get("pop_size", 100))
    num_generations = int(nsga_cfg.get("num_generations", 50))
    cxpb = float(nsga_cfg.get("cxpb", 0.7))
    mutpb = float(nsga_cfg.get("mutpb", 0.2))

    # Seed for reproducibility
    np.random.seed(seed)
    random.seed(seed)

    # Data config setup (support CLI csv override)
    cfg.setdefault("data", {})
    if args.csv_path:
        cfg["data"]["source"] = "csv"
        cfg["data"]["csv_path"] = args.csv_path

    # Override module-level daily_* with configured data
    daily_solar_irradiance, daily_wind_speed, daily_energy_demand = load_data_from_config(cfg)

    # Run optimization
    final_population, logbook, pareto_front = run_nsga2(pop_size, num_generations, cxpb=cxpb, mutpb=mutpb)

    # Analyze and visualize
    pareto_solutions_df = analyze_results(final_population, pareto_front)

    # Prepare output directory
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    base_out = cfg.get("output", {}).get("base_dir", "results")
    run_dir = args.output_dir or os.path.join(base_out, tag, timestamp)
    save_run_outputs(pareto_solutions_df, logbook, run_dir, args.config if os.path.isfile(args.config) else None)

    # Visualizations
    visualize_pareto_front(pareto_solutions_df, os.path.join(run_dir, "pareto"))
    visualize_solutions_composition(pareto_solutions_df, os.path.join(run_dir, "composition"))

    # Find the best trade-off solution for further analysis
    cost_norm = (pareto_solutions_df['cost'] - pareto_solutions_df['cost'].min()) / (pareto_solutions_df['cost'].max() - pareto_solutions_df['cost'].min())
    reliability_norm = 1 - (pareto_solutions_df['reliability'] - pareto_solutions_df['reliability'].min()) / (pareto_solutions_df['reliability'].max() - pareto_solutions_df['reliability'].min())
    impact_norm = (pareto_solutions_df['environmental_impact'] - pareto_solutions_df['environmental_impact'].min()) / (pareto_solutions_df['environmental_impact'].max() - pareto_solutions_df['environmental_impact'].min())
    pareto_solutions_df['distance'] = np.sqrt(cost_norm**2 + reliability_norm**2 + impact_norm**2)
    best_tradeoff_sol = pareto_solutions_df.loc[pareto_solutions_df['distance'].idxmin()]

    # Perform sensitivity analysis on the best trade-off solution
    # Convert the selected Series into the dict format expected by utils.perform_sensitivity_analysis
    best_tradeoff_payload = {
        "name": "Best Trade-off",
        "solar_panels": int(best_tradeoff_sol["solar_panels"]),
        "wind_turbines": int(best_tradeoff_sol["wind_turbines"]),
        "batteries": int(best_tradeoff_sol["batteries"]),
        "cost": float(best_tradeoff_sol["cost"]),
        "reliability": float(best_tradeoff_sol["reliability"]),
        "environmental_impact": float(best_tradeoff_sol["environmental_impact"]),
    }
    sensitivity_results = perform_sensitivity_analysis(best_tradeoff_payload, os.path.join(run_dir, "sensitivity"))
    
    # Define the selected solutions from Pareto front for detailed analysis
    solutions = [
        {"name": "Lowest Cost", "solar_panels": 83, "wind_turbines": 5, "batteries": 3, "cost": 91620},
        {"name": "Highest Reliability (Cost-Effective)", "solar_panels": 125, "wind_turbines": 5, "batteries": 3, "cost": 109260},
        {"name": "Lowest Emissions", "solar_panels": 29, "wind_turbines": 17, "batteries": 0, "cost": 175380},
        {"name": "Balanced", "solar_panels": 138, "wind_turbines": 6, "batteries": 2, "cost": 121400}
    ]
    
    # Analyze each selected solution
    analysis_results = []
    for solution in solutions:
        result = analyze_solution_detail(solution, save_dir=os.path.join(run_dir, "details"))
        analysis_results.append(result)
    
    print("\nAnalysis complete! All visualizations and CSVs have been saved in:")
    print(run_dir)
