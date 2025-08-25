"""
Weighted Sum Genetic Algorithm implementation.

Single-objective genetic algorithm using weighted sum approach for 
multi-objective renewable energy system optimization.
"""

import numpy as np
import random
from deap import base, creator, tools, algorithms


# Normalization bounds (estimate from problem domain)
COST_MIN, COST_MAX = 0, 500000
RELIABILITY_MIN, RELIABILITY_MAX = 0.0, 1.0  
IMPACT_MIN, IMPACT_MAX = 0.0, 1.0


def normalize(value, vmin, vmax):
    """Normalize value to [0,1] range."""
    if vmax - vmin == 0:
        return 0.0
    return (value - vmin) / (vmax - vmin)


def create_weighted_evaluate_func(base_evaluate_func, w_cost=0.6, w_reliability=0.3, w_impact=0.1):
    """
    Create a weighted sum evaluation function.
    
    Args:
        base_evaluate_func: Base multi-objective evaluation function
        w_cost: Weight for cost objective
        w_reliability: Weight for reliability objective  
        w_impact: Weight for environmental impact objective
        
    Returns:
        Weighted sum evaluation function
    """
    def evaluate_weighted(individual):
        """Evaluate individual using weighted sum of normalized objectives."""
        # Get multi-objective values
        result = base_evaluate_func(individual)
        
        # Check if infeasible
        if result[0] == float('inf'):
            return (float('inf'),)
        
        cost, reliability, environmental_impact = result
        
        # Cache true objective metrics on the individual for later reuse
        try:
            individual._true_cost = float(cost)
            individual._true_reliability = float(reliability)
            individual._true_impact = float(environmental_impact)
        except Exception:
            pass
        
        # Normalize objectives
        cost_norm = normalize(cost, COST_MIN, COST_MAX)
        reliability_norm = normalize(reliability, RELIABILITY_MIN, RELIABILITY_MAX)
        impact_norm = normalize(environmental_impact, IMPACT_MIN, IMPACT_MAX)
        
        # Weighted sum (minimize cost and impact, maximize reliability)
        weighted_objective = w_cost * cost_norm + w_impact * impact_norm - w_reliability * reliability_norm
        
        return (weighted_objective,)
    
    return evaluate_weighted


def run_weighted_sum_ga(bounds, base_evaluate_func, pop_size=100, num_generations=50, 
                       cxpb=0.7, mutpb=0.2, w_cost=0.6, w_reliability=0.3, w_impact=0.1):
    """
    Run GA with weighted sum approach.
    
    Args:
        bounds: List of (min, max) tuples for each decision variable
        base_evaluate_func: Base multi-objective evaluation function
        pop_size: Population size
        num_generations: Number of generations
        cxpb: Crossover probability
        mutpb: Mutation probability
        w_cost: Weight for cost objective
        w_reliability: Weight for reliability objective
        w_impact: Weight for environmental impact objective
        
    Returns:
        Tuple of (population, logbook, hall_of_fame)
    """
    
    # Create single-objective fitness and individual classes (avoid redefinition warnings)
    if not hasattr(creator, "FitnessWeighted"):
        creator.create("FitnessWeighted", base.Fitness, weights=(-1.0,))
    if not hasattr(creator, "IndividualWeighted"):
        creator.create("IndividualWeighted", list, fitness=creator.FitnessWeighted)
    
    # Create new toolbox for weighted sum
    toolbox_weighted = base.Toolbox()
    
    # Create weighted sum individual function
    def create_individual_weighted():
        return creator.IndividualWeighted([
            random.randint(bounds[0][0], bounds[0][1]),  # num_solar_panels
            random.randint(bounds[1][0], bounds[1][1]),  # num_wind_turbines
            random.randint(bounds[2][0], bounds[2][1])   # num_batteries
        ])
    
    # Create weighted evaluation function
    weighted_evaluate = create_weighted_evaluate_func(
        base_evaluate_func, w_cost, w_reliability, w_impact
    )
    
    toolbox_weighted.register("individual", create_individual_weighted)
    toolbox_weighted.register("population", tools.initRepeat, list, toolbox_weighted.individual)
    toolbox_weighted.register("evaluate", weighted_evaluate)
    toolbox_weighted.register("mate", tools.cxTwoPoint)
    toolbox_weighted.register("mutate", tools.mutUniformInt, 
                            low=[bounds[i][0] for i in range(3)], 
                            up=[bounds[i][1] for i in range(3)], indpb=0.2)
    toolbox_weighted.register("select", tools.selTournament, tournsize=3)
    
    # Initialize statistics
    stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register("avg", np.mean)
    stats.register("min", np.min)
    
    # Create and evaluate initial population
    population = toolbox_weighted.population(n=pop_size)
    fitnesses = list(map(toolbox_weighted.evaluate, population))
    for ind, fit in zip(population, fitnesses):
        ind.fitness.values = fit
    
    # Hall of fame for best solutions
    hof = tools.HallOfFame(1)
    
    print(f"Starting Weighted Sum GA (w_cost={w_cost}, w_reliability={w_reliability}, w_impact={w_impact})...")
    population, logbook = algorithms.eaSimple(
        population, 
        toolbox_weighted,
        cxpb=cxpb,
        mutpb=mutpb,
        ngen=num_generations,
        stats=stats,
        halloffame=hof,
        verbose=True
    )
    
    return population, logbook, hof
