"""
NSGA-II (Non-dominated Sorting Genetic Algorithm II) implementation.

Multi-objective genetic algorithm for renewable energy system optimization.
"""

import numpy as np
import random
from deap import base, creator, tools, algorithms


def run_nsga2(bounds, evaluate_func, pop_size=100, num_generations=50, cxpb=0.7, mutpb=0.2):
    """
    Run the NSGA-II algorithm.
    
    Args:
        bounds: List of (min, max) tuples for each decision variable
        evaluate_func: Function to evaluate individual fitness
        pop_size: Population size
        num_generations: Number of generations
        cxpb: Crossover probability
        mutpb: Mutation probability
        
    Returns:
        Tuple of (population, logbook, pareto_front)
    """
    
    # Create fitness and individual classes (avoid redefinition warnings)
    if not hasattr(creator, "FitnessMulti"):
        creator.create("FitnessMulti", base.Fitness, weights=(-1.0, 1.0, -1.0))
    if not hasattr(creator, "Individual"):
        creator.create("Individual", list, fitness=creator.FitnessMulti)
    
    # Initialize toolbox
    toolbox = base.Toolbox()
    
    # Register the individual and population creation functions
    def create_individual():
        return creator.Individual([
            random.randint(bounds[0][0], bounds[0][1]),  # num_solar_panels
            random.randint(bounds[1][0], bounds[1][1]),  # num_wind_turbines
            random.randint(bounds[2][0], bounds[2][1])   # num_batteries
        ])
    
    toolbox.register("individual", create_individual)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)
    toolbox.register("evaluate", evaluate_func)
    
    # Register the genetic operators
    toolbox.register("mate", tools.cxTwoPoint)
    toolbox.register("mutate", tools.mutUniformInt, 
                     low=[bounds[i][0] for i in range(3)], 
                     up=[bounds[i][1] for i in range(3)], 
                     indpb=0.2)
    toolbox.register("select", tools.selNSGA2)
    
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
