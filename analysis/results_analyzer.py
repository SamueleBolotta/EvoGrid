"""
Results analysis for optimization algorithms.

Contains functions for analyzing and processing optimization results
from various algorithms.
"""

import pandas as pd


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
