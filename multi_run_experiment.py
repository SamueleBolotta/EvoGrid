"""
Multi-run experiment script for comprehensive algorithm comparison.

This script runs multiple independent experiments with different random seeds
and compiles statistical comparison of NSGA-II, MOPSO, and Weighted Sum algorithms.
"""

import os
import sys
import random
import numpy as np
import pandas as pd
import argparse
from typing import Tuple
import datetime

# Add project root to path
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, project_root)

from algorithms import run_nsga2, run_mopso_optimization, run_weighted_sum_ga, run_weighted_sum_sweep
from energy_system import calculate_total_cost, calculate_reliability, calculate_environmental_impact
from energy_system.parameters import BOUNDS
from data_generation import load_data_from_config
from analysis import analyze_results
from analysis.algorithm_comparison import run_multi_run_comparison, AlgorithmComparison
from config import load_config, apply_to_main


def create_evaluation_function(daily_solar_irradiance, daily_wind_speed, daily_energy_demand):
    """Create evaluation function with data closure."""
    def evaluate(individual):
        """Evaluate an individual and return a tuple of objective values."""
        from energy_system.constraints import check_constraints
        
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
    pareto_solutions, mopso_instance = run_mopso_optimization(
        bounds=bounds,
        evaluate_func=evaluate_func,
        swarm_size=swarm_size,
        max_iterations=max_iterations,
        archive_size=archive_size,
        w=inertia_weight,
        c1=cognitive_coeff,
        c2=social_coeff,
        verbose=False  # Disable verbose for multi-run
    )
    
    return pareto_solutions, mopso_instance


def single_run_all_algorithms(config_path: str = None, seed: int = None) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Run all three algorithms once and return their results.
    
    Args:
        config_path: Path to configuration file
        seed: Random seed for reproducibility
    
    Returns:
        Tuple of (nsga_df, mopso_df, weighted_df)
    """
    # Set random seed if provided
    if seed is not None:
        np.random.seed(seed)
        random.seed(seed)
    
    # Load configuration
    cfg = {}
    if config_path and os.path.isfile(config_path):
        cfg = load_config(config_path)
        # Import parameters module to get default values
        from energy_system import parameters
        # Apply config to parameters module instead
        apply_to_main(cfg, parameters)
    
    # Extract parameters from config
    nsga_cfg = cfg.get("nsga", {})
    pop_size = int(nsga_cfg.get("pop_size", 100))
    num_generations = int(nsga_cfg.get("num_generations", 50))
    cxpb = float(nsga_cfg.get("cxpb", 0.7))
    mutpb = float(nsga_cfg.get("mutpb", 0.2))

    # Weighted-sum baseline config
    weighted_cfg = cfg.get("weighted", {})
    w_pop_size = int(weighted_cfg.get("pop_size", pop_size))
    w_num_generations = int(weighted_cfg.get("num_generations", num_generations))
    w_cxpb = float(weighted_cfg.get("cxpb", cxpb))
    w_mutpb = float(weighted_cfg.get("mutpb", mutpb))
    weights_cfg = weighted_cfg.get("weights", [0.6, 0.3, 0.1])
    weights_list = weighted_cfg.get("weights_list", None)
    try:
        w_cost, w_reliability, w_impact = [float(x) for x in weights_cfg]
    except Exception:
        w_cost, w_reliability, w_impact = 0.6, 0.3, 0.1

    # MOPSO config
    mopso_cfg = cfg.get("mopso", {})
    mopso_swarm_size = int(mopso_cfg.get("swarm_size", pop_size))
    mopso_max_iterations = int(mopso_cfg.get("max_iterations", num_generations))
    mopso_archive_size = int(mopso_cfg.get("archive_size", pop_size))
    mopso_inertia_weight = float(mopso_cfg.get("inertia_weight", 0.5))
    mopso_cognitive_coeff = float(mopso_cfg.get("cognitive_coeff", 1.5))
    mopso_social_coeff = float(mopso_cfg.get("social_coeff", 1.5))
    
    # Load environmental data
    daily_solar_irradiance, daily_wind_speed, daily_energy_demand = load_data_from_config(cfg)
    
    # Create evaluation function
    evaluate_func = create_evaluation_function(daily_solar_irradiance, daily_wind_speed, daily_energy_demand)
    
    # Run NSGA-II
    nsga_population, nsga_logbook, nsga_pareto = run_nsga2(
        bounds=BOUNDS,
        evaluate_func=evaluate_func,
        pop_size=pop_size,
        num_generations=num_generations,
        cxpb=cxpb,
        mutpb=mutpb
    )
    
    # Run MOPSO
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
    
    # Run Weighted Sum (sweep over weights_list if provided, otherwise single baseline)
    if weights_list and isinstance(weights_list, (list, tuple)) and len(weights_list) > 0:
        weighted_df = run_weighted_sum_sweep(
            bounds=BOUNDS,
            base_evaluate_func=evaluate_func,
            weights_list=weights_list,
            pop_size=w_pop_size,
            num_generations=w_num_generations,
            cxpb=w_cxpb,
            mutpb=w_mutpb
        )
    else:
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

        # Convert Weighted Sum results
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
    
    # Convert results to DataFrames
    nsga_df = analyze_results(nsga_population, nsga_pareto)
    
    # Convert MOPSO results
    mopso_df = pd.DataFrame(mopso_instance.get_pareto_front())
    
    return nsga_df, mopso_df, weighted_df


def main():
    """Main execution function for multi-run experiments."""
    parser = argparse.ArgumentParser(description="Multi-run algorithm comparison")
    parser.add_argument("--config", type=str, default="config.yaml", help="Path to YAML config file")
    parser.add_argument("--runs", type=int, default=10, help="Number of independent runs")
    parser.add_argument("--output-dir", type=str, default=None, help="Directory to save results")
    parser.add_argument("--base-seed", type=int, default=42, help="Base random seed")
    args = parser.parse_args()
    
    # Setup output directory
    if args.output_dir is None:
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        args.output_dir = f"results/multi_run_{timestamp}"
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    print(f"Multi-run experiment configuration:")
    print(f"  • Number of runs: {args.runs}")
    print(f"  • Config file: {args.config}")
    print(f"  • Output directory: {args.output_dir}")
    print(f"  • Base seed: {args.base_seed}")
    
    # Create run function with proper seed management
    def run_with_seed(config_path):
        # Generate a unique seed for this run
        run_seed = random.randint(1, 1000000)
        return single_run_all_algorithms(config_path, run_seed)
    
    # Run multi-run comparison
    results_df, test_results = run_multi_run_comparison(
        run_function=run_with_seed,
        n_runs=args.runs,
        config_path=args.config,
        output_dir=args.output_dir
    )
    
    # Save detailed results
    print(f"\n{'='*80}")
    print("SAVING RESULTS")
    print("="*80)
    
    # Save statistical summary
    summary_path = os.path.join(args.output_dir, "statistical_summary.txt")
    with open(summary_path, 'w') as f:
        f.write(f"Multi-run Algorithm Comparison Results\n")
        f.write(f"{'='*50}\n")
        f.write(f"Number of runs: {args.runs}\n")
        f.write(f"Config file: {args.config}\n")
        f.write(f"Generated on: {datetime.datetime.now()}\n\n")
        
        # Write metric summaries
        for metric in results_df['Metric'].unique():
            f.write(f"\n{metric.replace('_', ' ').upper()}:\n")
            f.write("-" * 30 + "\n")
            metric_data = results_df[results_df['Metric'] == metric].sort_values('Mean')
            
            for _, row in metric_data.iterrows():
                f.write(f"  {row['Algorithm']:15}: {row['Mean']:10.4f} ± {row['Std']:8.4f}\n")
        
        # Write statistical test results
        f.write(f"\n\nSTATISTICAL SIGNIFICANCE TESTS:\n")
        f.write("="*40 + "\n")
        
        for metric, tests in test_results.items():
            f.write(f"\n{metric.replace('_', ' ').upper()}:\n")
            if tests.get('kruskal_wallis'):
                kw = tests['kruskal_wallis']
                significance = "SIGNIFICANT" if kw['significant'] else "NOT SIGNIFICANT"
                f.write(f"  Kruskal-Wallis: p = {kw['p_value']:.4f} ({significance})\n")
            
            if tests.get('pairwise'):
                f.write("  Pairwise comparisons:\n")
                for comparison, result in tests['pairwise'].items():
                    if result:
                        significance = "***" if result['significant'] else "   "
                        f.write(f"    {comparison:20}: p = {result['p_value']:.4f} {significance}\n")
    
    print(f"Results saved to: {args.output_dir}")
    print(f"Key files:")
    print(f"  • multi_run_comparison.csv: Raw statistical data")
    print(f"  • statistical_summary.txt: Human-readable summary")
    print(f"  • metrics_comparison.png: Bar charts with error bars")
    print(f"  • distribution_comparison.png: Distribution comparison plots")
    
    # Print best performing algorithm for each metric
    print(f"\n{'='*80}")
    print("BEST PERFORMING ALGORITHMS")
    print("="*80)
    
    for metric in results_df['Metric'].unique():
        metric_data = results_df[results_df['Metric'] == metric]
        
        if metric in ['igd', 'spacing']:  # Lower is better
            best_row = metric_data.loc[metric_data['Mean'].idxmin()]
            print(f"{metric.replace('_', ' ').title():20}: {best_row['Algorithm']} "
                  f"({best_row['Mean']:.4f} ± {best_row['Std']:.4f}) - lower is better")
        else:  # Higher is better
            best_row = metric_data.loc[metric_data['Mean'].idxmax()]
            print(f"{metric.replace('_', ' ').title():20}: {best_row['Algorithm']} "
                  f"({best_row['Mean']:.4f} ± {best_row['Std']:.4f}) - higher is better")


if __name__ == "__main__":
    main()
