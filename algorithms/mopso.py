import numpy as np
import random
import copy
from typing import List, Tuple, Dict, Any
import math


class Particle:
    """Represents a particle in the MOPSO algorithm."""
    
    def __init__(self, bounds: List[Tuple[int, int]]):
        """
        Initialize a particle with random position and velocity.
        
        Args:
            bounds: List of (min, max) tuples for each dimension
        """
        self.bounds = bounds
        self.dimensions = len(bounds)
        
        # Initialize position randomly within bounds
        self.position = [random.randint(bounds[i][0], bounds[i][1]) for i in range(self.dimensions)]
        
        # Initialize velocity (discrete PSO uses velocity to determine position changes)
        max_velocity = [(bounds[i][1] - bounds[i][0]) // 4 for i in range(self.dimensions)]
        self.velocity = [random.randint(-max_velocity[i], max_velocity[i]) for i in range(self.dimensions)]
        
        # Personal best position and fitness
        self.pbest_position = copy.deepcopy(self.position)
        self.pbest_fitness = None
        
        # Current fitness
        self.fitness = None
        
        # Domination count and dominated solutions (for NSGA-II style ranking)
        self.domination_count = 0
        self.dominated_solutions = []
        self.rank = None
        self.crowding_distance = 0.0
    
    def update_velocity(self, gbest_position: List[int], w: float = 0.5, c1: float = 1.5, c2: float = 1.5):
        """
        Update particle velocity using PSO formula.
        
        Args:
            gbest_position: Global best position (from archive)
            w: Inertia weight
            c1: Cognitive coefficient
            c2: Social coefficient
        """
        for i in range(self.dimensions):
            r1, r2 = random.random(), random.random()
            
            cognitive = c1 * r1 * (self.pbest_position[i] - self.position[i])
            social = c2 * r2 * (gbest_position[i] - self.position[i])
            
            self.velocity[i] = w * self.velocity[i] + cognitive + social
            
            # Limit velocity to reasonable bounds
            max_vel = (self.bounds[i][1] - self.bounds[i][0]) // 2
            self.velocity[i] = max(-max_vel, min(max_vel, self.velocity[i]))
    
    def update_position(self):
        """Update particle position based on velocity."""
        for i in range(self.dimensions):
            # Apply velocity (rounded to integer for discrete optimization)
            self.position[i] += int(round(self.velocity[i]))
            
            # Ensure position stays within bounds
            self.position[i] = max(self.bounds[i][0], min(self.bounds[i][1], self.position[i]))
    
    def update_pbest(self):
        """Update personal best if current position dominates previous best."""
        if self.pbest_fitness is None or self.dominates_fitness(self.fitness, self.pbest_fitness):
            self.pbest_position = copy.deepcopy(self.position)
            self.pbest_fitness = copy.deepcopy(self.fitness)
    
    @staticmethod
    def dominates_fitness(fitness1: Tuple[float, ...], fitness2: Tuple[float, ...]) -> bool:
        """
        Check if fitness1 dominates fitness2 (assuming minimization for all objectives).
        
        Args:
            fitness1: First fitness tuple
            fitness2: Second fitness tuple
            
        Returns:
            True if fitness1 dominates fitness2
        """
        if fitness1 is None or fitness2 is None:
            return False
            
        # Convert to numpy arrays for easier comparison
        f1 = np.array(fitness1)
        f2 = np.array(fitness2)
        
        # For our problem: minimize cost, maximize reliability, minimize environmental impact
        # Convert to all minimization: cost, -reliability, environmental_impact
        f1_min = np.array([f1[0], -f1[1], f1[2]])
        f2_min = np.array([f2[0], -f2[1], f2[2]])
        
        # Check if f1 dominates f2
        return np.all(f1_min <= f2_min) and np.any(f1_min < f2_min)


class Archive:
    """External archive to store non-dominated solutions."""
    
    def __init__(self, max_size: int = 100):
        """
        Initialize archive.
        
        Args:
            max_size: Maximum number of solutions to store
        """
        self.max_size = max_size
        self.solutions = []
    
    def add_solution(self, particle: Particle):
        """
        Add a particle to the archive if it's non-dominated.
        
        Args:
            particle: Particle to potentially add
        """
        # Create a copy to avoid reference issues
        new_particle = copy.deepcopy(particle)
        
        # Check if new particle is dominated by existing solutions
        dominated_by_existing = False
        for existing in self.solutions:
            if Particle.dominates_fitness(existing.fitness, new_particle.fitness):
                dominated_by_existing = True
                break
        
        if dominated_by_existing:
            return  # Don't add dominated solution
        
        # Remove existing solutions dominated by new particle
        self.solutions = [existing for existing in self.solutions 
                         if not Particle.dominates_fitness(new_particle.fitness, existing.fitness)]
        
        # Add new particle
        self.solutions.append(new_particle)
        
        # Maintain archive size
        if len(self.solutions) > self.max_size:
            self._truncate_archive()
    
    def _truncate_archive(self):
        """Truncate archive using crowding distance when it exceeds max size."""
        # Calculate crowding distance for all solutions
        self._calculate_crowding_distance()
        
        # Sort by crowding distance (descending) and remove solutions with smallest distance
        self.solutions.sort(key=lambda x: x.crowding_distance, reverse=True)
        self.solutions = self.solutions[:self.max_size]
    
    def _calculate_crowding_distance(self):
        """Calculate crowding distance for all solutions in archive."""
        n = len(self.solutions)
        
        if n <= 2:
            for sol in self.solutions:
                sol.crowding_distance = float('inf')
            return
        
        # Initialize crowding distance
        for sol in self.solutions:
            sol.crowding_distance = 0.0
        
        # Number of objectives
        num_objectives = len(self.solutions[0].fitness)
        
        for obj_idx in range(num_objectives):
            # Sort by objective
            self.solutions.sort(key=lambda x: x.fitness[obj_idx])
            
            # Set boundary solutions to infinite distance
            self.solutions[0].crowding_distance = float('inf')
            self.solutions[-1].crowding_distance = float('inf')
            
            # Calculate range
            obj_min = self.solutions[0].fitness[obj_idx]
            obj_max = self.solutions[-1].fitness[obj_idx]
            obj_range = obj_max - obj_min
            
            if obj_range == 0:
                continue
            
            # Calculate crowding distance for intermediate solutions
            for i in range(1, n - 1):
                if self.solutions[i].crowding_distance != float('inf'):
                    distance = (self.solutions[i + 1].fitness[obj_idx] - 
                               self.solutions[i - 1].fitness[obj_idx]) / obj_range
                    self.solutions[i].crowding_distance += distance
    
    def get_random_solution(self) -> Particle:
        """Get a random solution from the archive."""
        if not self.solutions:
            return None
        return random.choice(self.solutions)
    
    def get_best_solution(self, objective_idx: int = 0) -> Particle:
        """
        Get the best solution for a specific objective.
        
        Args:
            objective_idx: Index of objective (0=cost, 1=reliability, 2=environmental_impact)
            
        Returns:
            Best particle for the given objective
        """
        if not self.solutions:
            return None
        
        if objective_idx == 1:  # Reliability (maximize)
            return max(self.solutions, key=lambda x: x.fitness[1])
        else:  # Cost and environmental impact (minimize)
            return min(self.solutions, key=lambda x: x.fitness[objective_idx])
    
    def size(self) -> int:
        """Return current archive size."""
        return len(self.solutions)


class MOPSO:
    """Multi-Objective Particle Swarm Optimization algorithm."""
    
    def __init__(self, 
                 bounds: List[Tuple[int, int]],
                 swarm_size: int = 100,
                 archive_size: int = 100,
                 w: float = 0.5,
                 c1: float = 1.5,
                 c2: float = 1.5):
        """
        Initialize MOPSO algorithm.
        
        Args:
            bounds: List of (min, max) tuples for each dimension
            swarm_size: Number of particles in swarm
            archive_size: Maximum size of external archive
            w: Inertia weight
            c1: Cognitive coefficient  
            c2: Social coefficient
        """
        self.bounds = bounds
        self.swarm_size = swarm_size
        self.archive = Archive(archive_size)
        self.w = w
        self.c1 = c1
        self.c2 = c2
        
        # Initialize swarm
        self.swarm = [Particle(bounds) for _ in range(swarm_size)]
        
        # Statistics tracking
        self.generation = 0
        self.best_fitness_history = []
        
    def optimize(self, 
                 evaluate_func,
                 max_iterations: int = 100,
                 verbose: bool = True) -> List[Particle]:
        """
        Run the MOPSO optimization algorithm.
        
        Args:
            evaluate_func: Function to evaluate particle fitness
            max_iterations: Maximum number of iterations
            verbose: Whether to print progress
            
        Returns:
            List of non-dominated solutions from archive
        """
        # Evaluate initial swarm
        for particle in self.swarm:
            particle.fitness = evaluate_func(particle.position)
            particle.update_pbest()
            self.archive.add_solution(particle)
        
        if verbose:
            print(f"Initial archive size: {self.archive.size()}")
        
        # Main optimization loop
        for iteration in range(max_iterations):
            self.generation = iteration
            
            for particle in self.swarm:
                # Select guide from archive
                guide = self._select_guide()
                
                if guide is not None:
                    # Update velocity and position
                    particle.update_velocity(guide.position, self.w, self.c1, self.c2)
                    particle.update_position()
                    
                    # Evaluate new position
                    particle.fitness = evaluate_func(particle.position)
                    
                    # Update personal best
                    particle.update_pbest()
                    
                    # Add to archive
                    self.archive.add_solution(particle)
            
            # Adaptive parameters (optional)
            self.w = max(0.1, self.w * 0.99)  # Decrease inertia weight over time
            
            # Track progress
            if self.archive.size() > 0:
                # Get best cost solution for tracking
                best_cost_particle = self.archive.get_best_solution(0)
                self.best_fitness_history.append(best_cost_particle.fitness)
            
            if verbose and (iteration + 1) % 10 == 0:
                print(f"Iteration {iteration + 1}/{max_iterations}, Archive size: {self.archive.size()}")
        
        if verbose:
            print(f"Final archive size: {self.archive.size()}")
        
        return self.archive.solutions
    
    def _select_guide(self) -> Particle:
        """
        Select a guide solution from the archive using tournament selection.
        
        Returns:
            Selected guide particle
        """
        if self.archive.size() == 0:
            return None
        
        if self.archive.size() == 1:
            return self.archive.solutions[0]
        
        # Tournament selection based on crowding distance
        tournament_size = min(2, self.archive.size())
        candidates = random.sample(self.archive.solutions, tournament_size)
        
        # Select the one with higher crowding distance (more diversity)
        return max(candidates, key=lambda x: x.crowding_distance)
    
    def get_pareto_front(self) -> List[Dict[str, Any]]:
        """
        Get the Pareto front as a list of dictionaries.
        
        Returns:
            List of solution dictionaries
        """
        pareto_solutions = []
        for i, particle in enumerate(self.archive.solutions):
            cost, reliability, env_impact = particle.fitness
            pareto_solutions.append({
                'id': i,
                'solar_panels': particle.position[0],
                'wind_turbines': particle.position[1], 
                'batteries': particle.position[2],
                'cost': cost,
                'reliability': reliability,
                'environmental_impact': env_impact
            })
        return pareto_solutions


def run_mopso_optimization(bounds: List[Tuple[int, int]],
                          evaluate_func,
                          swarm_size: int = 100,
                          max_iterations: int = 100,
                          archive_size: int = 100,
                          w: float = 0.5,
                          c1: float = 1.5,
                          c2: float = 1.5,
                          verbose: bool = True) -> Tuple[List[Particle], MOPSO]:
    """
    Convenience function to run MOPSO optimization.
    
    Args:
        bounds: Decision variable bounds
        evaluate_func: Fitness evaluation function
        swarm_size: Number of particles
        max_iterations: Number of iterations
        archive_size: Maximum archive size
        w: Inertia weight
        c1: Cognitive coefficient
        c2: Social coefficient
        verbose: Print progress
        
    Returns:
        Tuple of (pareto_solutions, mopso_instance)
    """
    mopso = MOPSO(bounds=bounds,
                  swarm_size=swarm_size,
                  archive_size=archive_size,
                  w=w, c1=c1, c2=c2)
    
    pareto_solutions = mopso.optimize(evaluate_func, max_iterations, verbose)
    
    return pareto_solutions, mopso


# Keep the old function name for backward compatibility
def run_mopso(*args, **kwargs):
    """Backward compatibility function."""
    return run_mopso_optimization(*args, **kwargs)
