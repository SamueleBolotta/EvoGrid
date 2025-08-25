"""
Visualization module for EvoGrid.

Contains functions for creating plots and visualizations of optimization
results and energy system analysis.
"""

from .pareto_plots import visualize_pareto_front
from .composition_plots import visualize_solutions_composition
from .comparison_plots import compare_methods, compare_all_methods_detailed

__all__ = [
    'visualize_pareto_front',
    'visualize_solutions_composition',
    'compare_methods',
    'compare_all_methods_detailed'
]
