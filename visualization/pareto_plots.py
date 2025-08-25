"""
Pareto front visualization for multi-objective optimization.

Contains functions for creating 2D and 3D visualizations of Pareto fronts
and objective space plots.
"""

import matplotlib.pyplot as plt
import numpy as np


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
