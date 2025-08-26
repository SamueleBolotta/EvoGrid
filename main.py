"""
EvoGrid: Bio-inspired renewable energy system optimization.

Main entry point for running multi-objective optimization algorithms
on renewable energy system design problems.
"""

import numpy as np
import pandas as pd
import random
import argparse
import os
import datetime

# Import all modules
from algorithms import run_nsga2, run_mopso_optimization, run_weighted_sum_ga
from energy_system import (
    calculate_total_cost, calculate_reliability, calculate_environmental_impact,
    check_constraints
)
from energy_system.parameters import BOUNDS
from data_generation import load_data_from_config
from analysis import analyze_results, perform_sensitivity_analysis, analyze_solution_detail
from visualization import (
    visualize_pareto_front, visualize_solutions_composition,
    compare_methods, compare_all_methods_detailed
)
from config import load_config, apply_to_main

# Set random seed for reproducibility
seed = 42
np.random.seed(seed)
random.seed(seed)

# Default MOPSO parameters (will be overridden by config)
MOPSO_SWARM_SIZE = 100
MOPSO_MAX_ITERATIONS = 50
MOPSO_ARCHIVE_SIZE = 100
MOPSO_INERTIA_WEIGHT = 0.5
MOPSO_COGNITIVE_COEFF = 1.5
MOPSO_SOCIAL_COEFF = 1.5


def create_evaluation_function(daily_solar_irradiance, daily_wind_speed, daily_energy_demand):
    """Create evaluation function with data closure."""
    def evaluate(individual):
        """Evaluate an individual and return a tuple of objective values."""
        # Check constraints first
        if not check_constraints(individual, daily_solar_irradiance, daily_wind_speed):
            return (float('inf'), 0.0, float('inf'))  # Heavily penalize infeasible solutions
        
        # Calculate objective values
        cost = calculate_total_cost(individual)
        reliability = calculate_reliability(individual, daily_solar_irradiance, daily_wind_speed, daily_energy_demand)
        environmental_impact = calculate_environmental_impact(individual, daily_solar_irradiance, daily_wind_speed)
        
        return (cost, reliability, environmental_impact)

    return evaluate


def run_mopso_with_params(bounds, evaluate_func, swarm_size=100, max_iterations=50, 
                         archive_size=100, inertia_weight=0.5, cognitive_coeff=1.5, social_coeff=1.5):
    """Run MOPSO optimization with parameters."""
    print("Starting MOPSO optimization...")
    pareto_solutions, mopso_instance = run_mopso_optimization(
        bounds=bounds,
        evaluate_func=evaluate_func,
        swarm_size=swarm_size,
        max_iterations=max_iterations,
        archive_size=archive_size,
        w=inertia_weight,
        c1=cognitive_coeff,
        c2=social_coeff,
        verbose=True
    )
    
    return pareto_solutions, mopso_instance


def combine_pareto_fronts(*solution_dfs):
    """
    Combine solutions from multiple algorithms and apply Pareto dominance filtering.
    
    Args:
        *solution_dfs: Variable number of DataFrames containing solutions with columns:
                      ['solar_panels', 'wind_turbines', 'batteries', 'cost', 'reliability', 'environmental_impact']
    
    Returns:
        DataFrame with unified Pareto optimal solutions
    """
    if not solution_dfs:
        return pd.DataFrame()
    
    # Combine all solutions
    all_solutions = []
    algorithm_names = ['NSGA-II', 'MOPSO', 'Weighted-Sum']  # Default names
    
    for i, df in enumerate(solution_dfs):
        if df is not None and not df.empty:
            df_copy = df.copy()
            # Add source algorithm identifier
            algorithm_name = algorithm_names[i] if i < len(algorithm_names) else f'Algorithm_{i}'
            df_copy['source_algorithm'] = algorithm_name
            all_solutions.append(df_copy)
    
    if not all_solutions:
        return pd.DataFrame()
    
    combined_df = pd.concat(all_solutions, ignore_index=True)
    
    # Apply Pareto dominance filtering
    pareto_optimal = []
    solutions = combined_df[['cost', 'reliability', 'environmental_impact']].values
    
    for i, sol1 in enumerate(solutions):
        is_dominated = False
        for j, sol2 in enumerate(solutions):
            if i != j:
                # Check if sol1 is dominated by sol2
                # For cost and env_impact: lower is better (minimize)
                # For reliability: higher is better (maximize)
                if (sol2[0] <= sol1[0] and  # cost
                    sol2[1] >= sol1[1] and  # reliability  
                    sol2[2] <= sol1[2] and  # environmental_impact
                    (sol2[0] < sol1[0] or sol2[1] > sol1[1] or sol2[2] < sol1[2])):
                    is_dominated = True
                    break
        
        if not is_dominated:
            pareto_optimal.append(i)
    
    # Create unified Pareto front DataFrame
    unified_pareto = combined_df.iloc[pareto_optimal].copy()
    unified_pareto.reset_index(drop=True, inplace=True)
    unified_pareto['id'] = range(len(unified_pareto))
    
    print(f"\nUnified Pareto Front Statistics:")
    print(f"Total solutions considered: {len(combined_df)}")
    print(f"Pareto optimal solutions: {len(unified_pareto)}")
    
    # Show contribution by algorithm
    if 'source_algorithm' in unified_pareto.columns:
        contribution = unified_pareto['source_algorithm'].value_counts()
        print(f"Contribution by algorithm:")
        for algo, count in contribution.items():
            percentage = (count / len(unified_pareto)) * 100
            print(f"  • {algo}: {count} solutions ({percentage:.1f}%)")
    
    return unified_pareto


def extract_representative_solutions(pareto_df):
    """
    Extract representative solutions from the Pareto front for detailed analysis.
    
    Args:
        pareto_df: DataFrame containing Pareto optimal solutions
        
    Returns:
        List of solution dictionaries with representative solutions
    """
    if pareto_df.empty:
        return []
    
    solutions = []
    
    # Find lowest cost solution
    lowest_cost_idx = pareto_df['cost'].idxmin()
    lowest_cost = pareto_df.loc[lowest_cost_idx]
    solutions.append({
        "name": "Lowest Cost",
        "solar_panels": int(lowest_cost['solar_panels']),
        "wind_turbines": int(lowest_cost['wind_turbines']),
        "batteries": int(lowest_cost['batteries']),
        "cost": float(lowest_cost['cost']),
        "reliability": float(lowest_cost['reliability']),
        "environmental_impact": float(lowest_cost['environmental_impact'])
    })
    
    # Find highest reliability solution
    highest_reliability_idx = pareto_df['reliability'].idxmax()
    highest_reliability = pareto_df.loc[highest_reliability_idx]
    solutions.append({
        "name": "Highest Reliability",
        "solar_panels": int(highest_reliability['solar_panels']),
        "wind_turbines": int(highest_reliability['wind_turbines']),
        "batteries": int(highest_reliability['batteries']),
        "cost": float(highest_reliability['cost']),
        "reliability": float(highest_reliability['reliability']),
        "environmental_impact": float(highest_reliability['environmental_impact'])
    })
    
    # Find lowest environmental impact solution
    lowest_emissions_idx = pareto_df['environmental_impact'].idxmin()
    lowest_emissions = pareto_df.loc[lowest_emissions_idx]
    solutions.append({
        "name": "Lowest Emissions",
        "solar_panels": int(lowest_emissions['solar_panels']),
        "wind_turbines": int(lowest_emissions['wind_turbines']),
        "batteries": int(lowest_emissions['batteries']),
        "cost": float(lowest_emissions['cost']),
        "reliability": float(lowest_emissions['reliability']),
        "environmental_impact": float(lowest_emissions['environmental_impact'])
    })
    
    # Find balanced solution (closest to ideal point using normalized distances)
    # Normalize objectives (cost and env_impact: minimize, reliability: maximize)
    cost_norm = (pareto_df['cost'] - pareto_df['cost'].min()) / (pareto_df['cost'].max() - pareto_df['cost'].min() + 1e-10)
    reliability_norm = 1 - (pareto_df['reliability'] - pareto_df['reliability'].min()) / (pareto_df['reliability'].max() - pareto_df['reliability'].min() + 1e-10)
    impact_norm = (pareto_df['environmental_impact'] - pareto_df['environmental_impact'].min()) / (pareto_df['environmental_impact'].max() - pareto_df['environmental_impact'].min() + 1e-10)
    
    # Calculate distance to ideal point (0,0,0)
    distances = np.sqrt(cost_norm**2 + reliability_norm**2 + impact_norm**2)
    balanced_idx = distances.idxmin()
    balanced = pareto_df.loc[balanced_idx]
    solutions.append({
        "name": "Balanced",
        "solar_panels": int(balanced['solar_panels']),
        "wind_turbines": int(balanced['wind_turbines']),
        "batteries": int(balanced['batteries']),
        "cost": float(balanced['cost']),
        "reliability": float(balanced['reliability']),
        "environmental_impact": float(balanced['environmental_impact'])
    })
    
    # Remove duplicates (if any solution appears multiple times)
    unique_solutions = []
    seen_configs = set()
    for sol in solutions:
        config_key = (sol['solar_panels'], sol['wind_turbines'], sol['batteries'])
        if config_key not in seen_configs:
            seen_configs.add(config_key)
            unique_solutions.append(sol)
    
    return unique_solutions


def save_run_outputs(pareto_df, logbook, out_dir: str, config_path: str = None, 
                    weighted_df: pd.DataFrame = None, mopso_df: pd.DataFrame = None, 
                    nsga_df: pd.DataFrame = None):
    """Save structured outputs for an experiment run."""
    
    def _ensure_dir(path: str):
        if path and not os.path.isdir(path):
            os.makedirs(path, exist_ok=True)

    _ensure_dir(out_dir)
    pareto_csv = os.path.join(out_dir, "pareto.csv")
    pareto_df.to_csv(pareto_csv, index=False)

    # Save logbook if convertible
    try:
        pd.DataFrame(logbook).to_csv(os.path.join(out_dir, "logbook.csv"), index=False)
    except Exception:
        pass

    # Save weighted baseline results if provided
    if weighted_df is not None:
        try:
            weighted_df.to_csv(os.path.join(out_dir, "weighted_solutions.csv"), index=False)
        except Exception:
            pass

    # Save MOPSO results if provided
    if mopso_df is not None:
        try:
            mopso_df.to_csv(os.path.join(out_dir, "mopso_solutions.csv"), index=False)
        except Exception:
            pass

    # Save NSGA-II results if provided
    if nsga_df is not None:
        try:
            nsga_df.to_csv(os.path.join(out_dir, "nsga2_solutions.csv"), index=False)
        except Exception:
            pass

    if config_path and os.path.isfile(config_path):
        try:
            import shutil
            shutil.copy(config_path, os.path.join(out_dir, "config.used.yaml"))
        except Exception:
            pass

    return out_dir


def main():
    """Main execution function."""
    parser = argparse.ArgumentParser(description="EvoGrid NSGA-II experiments")
    parser.add_argument("--config", type=str, default="config.yaml", help="Path to YAML config file")
    parser.add_argument("--output-dir", type=str, default=None, help="Directory to save outputs (optional)")
    parser.add_argument("--csv-path", type=str, default=None, help="Optional CSV path for data ingestion")
    args = parser.parse_args()

    # Load YAML config (if present)
    cfg = {}
    if args.config and os.path.isfile(args.config):
        try:
            cfg = load_config(args.config)
            # Apply config to globals
            apply_to_main(cfg, __import__(__name__))
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

    # Weighted-sum baseline config (defaults mirror NSGA-II unless overridden)
    weighted_cfg = cfg.get("weighted", {})
    w_pop_size = int(weighted_cfg.get("pop_size", pop_size))
    w_num_generations = int(weighted_cfg.get("num_generations", num_generations))
    w_cxpb = float(weighted_cfg.get("cxpb", cxpb))
    w_mutpb = float(weighted_cfg.get("mutpb", mutpb))
    weights_cfg = weighted_cfg.get("weights", [0.6, 0.3, 0.1])
    try:
        w_cost, w_reliability, w_impact = [float(x) for x in weights_cfg]
    except Exception:
        w_cost, w_reliability, w_impact = 0.6, 0.3, 0.1

    # MOPSO config (defaults mirror NSGA-II unless overridden)
    mopso_cfg = cfg.get("mopso", {})
    mopso_swarm_size = int(mopso_cfg.get("swarm_size", pop_size))
    mopso_max_iterations = int(mopso_cfg.get("max_iterations", num_generations))
    mopso_archive_size = int(mopso_cfg.get("archive_size", pop_size))
    mopso_inertia_weight = float(mopso_cfg.get("inertia_weight", 0.5))
    mopso_cognitive_coeff = float(mopso_cfg.get("cognitive_coeff", 1.5))
    mopso_social_coeff = float(mopso_cfg.get("social_coeff", 1.5))

    # Seed for reproducibility
    np.random.seed(seed)
    random.seed(seed)

    # Data config setup (support CLI csv override)
    cfg.setdefault("data", {})
    if args.csv_path:
        cfg["data"]["source"] = "csv"
        cfg["data"]["csv_path"] = args.csv_path

    # Load environmental data
    daily_solar_irradiance, daily_wind_speed, daily_energy_demand = load_data_from_config(cfg)

    # Create evaluation function
    evaluate_func = create_evaluation_function(daily_solar_irradiance, daily_wind_speed, daily_energy_demand)

    # Run all optimization approaches
    print("="*80)
    print("RUNNING NSGA-II (Multi-objective)")
    print("="*80)
    nsga_population, nsga_logbook, nsga_pareto = run_nsga2(
        bounds=BOUNDS,
        evaluate_func=evaluate_func,
        pop_size=pop_size,
        num_generations=num_generations,
        cxpb=cxpb,
        mutpb=mutpb
    )

    print("\n" + "="*80)
    print("RUNNING MOPSO (Multi-objective)")
    print("="*80)
    mopso_pareto_solutions, mopso_instance = run_mopso_with_params(
        bounds=BOUNDS,
        evaluate_func=evaluate_func,
        swarm_size=mopso_swarm_size, 
        max_iterations=mopso_max_iterations,
        archive_size=mopso_archive_size,
        inertia_weight=mopso_inertia_weight,
        cognitive_coeff=mopso_cognitive_coeff,
        social_coeff=mopso_social_coeff
    )

    print("\n" + "="*80)
    print("RUNNING WEIGHTED SUM BASELINE (Single-objective)")
    print("="*80)
    weighted_population, weighted_logbook, weighted_hof = run_weighted_sum_ga(
        bounds=BOUNDS,
        base_evaluate_func=evaluate_func,
        pop_size=w_pop_size,
        num_generations=w_num_generations,
        cxpb=w_cxpb,
        mutpb=w_mutpb,
        w_cost=w_cost,
        w_reliability=w_reliability,
        w_impact=w_impact
    )

    # Analyze results from each algorithm separately first
    nsga_pareto_df = analyze_results(nsga_population, nsga_pareto)
    
    # Convert MOPSO results to DataFrame
    mopso_pareto_df = pd.DataFrame(mopso_instance.get_pareto_front())

    # Save results and create visualizations
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    base_out = cfg.get("output", {}).get("base_dir", "results")
    run_dir = args.output_dir or os.path.join(base_out, tag, timestamp)

    # Process weighted sum results
    weighted_records = []
    for i, ind in enumerate(weighted_hof):
        cost_cached = getattr(ind, "_true_cost", None)
        reli_cached = getattr(ind, "_true_reliability", None)
        impact_cached = getattr(ind, "_true_impact", None)
        cost_val = cost_cached if cost_cached is not None else calculate_total_cost(ind)
        reli_val = reli_cached if reli_cached is not None else calculate_reliability(
            ind, daily_solar_irradiance, daily_wind_speed, daily_energy_demand)
        impact_val = impact_cached if impact_cached is not None else calculate_environmental_impact(
            ind, daily_solar_irradiance, daily_wind_speed)
        weighted_records.append({
            'id': i,
            'solar_panels': ind[0],
            'wind_turbines': ind[1],
            'batteries': ind[2],
            'cost': cost_val,
            'reliability': reli_val,
            'environmental_impact': impact_val
        })
    weighted_df = pd.DataFrame(weighted_records)
    
    # Create unified Pareto front combining all algorithms
    print("\n" + "="*80)
    print("CREATING UNIFIED PARETO FRONT")
    print("="*80)
    pareto_solutions_df = combine_pareto_fronts(nsga_pareto_df, mopso_pareto_df, weighted_df)

    # Persist all outputs (unified Pareto front + individual algorithm results)
    save_run_outputs(pareto_solutions_df, nsga_logbook, run_dir, 
                    args.config if os.path.isfile(args.config) else None, 
                    weighted_df=weighted_df, mopso_df=mopso_pareto_df, nsga_df=nsga_pareto_df)

    # Create comparisons and visualizations using the original individual results
    compare_methods(nsga_pareto_df, weighted_df, os.path.join(run_dir, "methods"), mopso_df=mopso_pareto_df)
    
    # Detailed comparison including MOPSO (using individual algorithm results)
    comparison_metrics = compare_all_methods_detailed(nsga_pareto_df, weighted_df, mopso_pareto_df, 
                                                     os.path.join(run_dir, "methods"))

    # Continue with existing visualizations using the unified Pareto front
    visualize_pareto_front(pareto_solutions_df, os.path.join(run_dir, "pareto"))
    visualize_solutions_composition(pareto_solutions_df, os.path.join(run_dir, "composition"))

    # Find the best trade-off solution for further analysis
    cost_norm = (pareto_solutions_df['cost'] - pareto_solutions_df['cost'].min()) / (pareto_solutions_df['cost'].max() - pareto_solutions_df['cost'].min())
    reliability_norm = 1 - (pareto_solutions_df['reliability'] - pareto_solutions_df['reliability'].min()) / (pareto_solutions_df['reliability'].max() - pareto_solutions_df['reliability'].min())
    impact_norm = (pareto_solutions_df['environmental_impact'] - pareto_solutions_df['environmental_impact'].min()) / (pareto_solutions_df['environmental_impact'].max() - pareto_solutions_df['environmental_impact'].min())
    pareto_solutions_df['distance'] = np.sqrt(cost_norm**2 + reliability_norm**2 + impact_norm**2)
    best_tradeoff_sol = pareto_solutions_df.loc[pareto_solutions_df['distance'].idxmin()]

    # Perform simplified sensitivity analysis
    best_tradeoff_payload = {
        "name": "Best Trade-off",
        "solar_panels": int(best_tradeoff_sol["solar_panels"]),
        "wind_turbines": int(best_tradeoff_sol["wind_turbines"]),
        "batteries": int(best_tradeoff_sol["batteries"]),
        "cost": float(best_tradeoff_sol["cost"]),
        "reliability": float(best_tradeoff_sol["reliability"]),
        "environmental_impact": float(best_tradeoff_sol["environmental_impact"]),
    }
    sensitivity_results = perform_sensitivity_analysis(
        best_tradeoff_payload, 
        evaluate_func,
        (daily_solar_irradiance, daily_wind_speed, daily_energy_demand),
        os.path.join(run_dir, "sensitivity")
    )
    
    # Extract representative solutions from the unified Pareto front for detailed analysis
    solutions = extract_representative_solutions(pareto_solutions_df)
    
    # Analyze each selected solution
    analysis_results = []
    for solution in solutions:
        result = analyze_solution_detail(
            solution, 
            (daily_solar_irradiance, daily_wind_speed, daily_energy_demand),
            save_dir=os.path.join(run_dir, "details")
        )
        analysis_results.append(result)
    
    print("\nAnalysis complete! All visualizations and CSVs have been saved in:")
    print(run_dir)
    print(f"\nKey outputs:")
    print(f"  • pareto.csv: Unified Pareto front from all algorithms")
    print(f"  • nsga2_solutions.csv: NSGA-II solutions")
    print(f"  • mopso_solutions.csv: MOPSO solutions") 
    print(f"  • weighted_solutions.csv: Weighted sum solutions")
    print(f"  • methods_comparison.png: Individual algorithm comparison")
    print(f"  • pareto_pareto2d.png & pareto_pareto3d.png: Unified Pareto front plots")


if __name__ == "__main__":
    main()
