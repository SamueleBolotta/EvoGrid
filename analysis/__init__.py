"""
Analysis module for EvoGrid.

Contains functions for analyzing optimization results and performing
detailed solution analysis.
"""

from .results_analyzer import analyze_results
from .sensitivity_analysis import perform_sensitivity_analysis
from .solution_analysis import analyze_solution_detail

__all__ = [
    'analyze_results',
    'perform_sensitivity_analysis', 
    'analyze_solution_detail'
]
