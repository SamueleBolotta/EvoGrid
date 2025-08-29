"""
Multi-run algorithm comparison with statistical analysis.

This module provides comprehensive comparison of multi-objective optimization algorithms
including hypervolume, IGD, spacing, and other quality metrics across multiple runs.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.spatial.distance import pdist, squareform
from scipy.stats import kruskal, mannwhitneyu
import os
from typing import List, Dict, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')
from energy_system.parameters import *
# from energy_system.production import calculate_solar_energy_production, calculate_wind_energy_production
import math
from pymoo.indicators.hv import HV

# Set style for better plots
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")


class AlgorithmComparison:
    """Class for comprehensive multi-run algorithm comparison."""
    
    def __init__(self, reference_point: Optional[np.ndarray] = None):
        """
        Initialize comparison class.
        
        Args:
            reference_point: Reference point for hypervolume calculation [cost, -reliability, env_impact]
        """
        self.reference_point = reference_point
        self.results = {}
        # Store per-run metrics for proper distribution plots
        self.per_run_metrics: List[Dict] = []
        # Default to a fixed, problem-aware reference point if none provided
        if self.reference_point is None:
            try:
                self.reference_point = self.get_evo_grid_reference_point()
            except Exception:
                # Fallback: will be set adaptively inside calculate_hypervolume
                self.reference_point = None 


    def get_evo_grid_reference_point(self) -> np.ndarray:
        """
        Define a fixed reference point for hypervolume based on EvoGrid problem constraints.
        A reference point must be slightly worse than the worst-possible values for each
        objective to ensure all feasible solutions are dominated by it.
        """
        # 1. Cost Objective (Minimization)
        # The worst-case cost is bounded by the project budget. Add a 10% margin.
        max_cost = float(MAX_BUDGET) * 1.1

        # 2. Reliability Objective (Maximization -> Minimization)
        # Reliability is in [0, 1]. For minimization, we use -reliability, in range [-1, 0].
        # The worst-case value is 0. The reference point must be >= 0.
        neg_reliability_ref = 0.0

        # 3. Environmental Impact Objective (Minimization)
        # To get a stable upper bound, we calculate the max possible total emissions,
        # assuming the maximum number of components are installed.
        max_solar_panels, max_wind_turbines, max_batteries = [b[1] for b in BOUNDS]
        
        # Calculate battery replacements needed over the project lifetime
        # Using ceiling for a conservative, whole-number replacement count.
        avg_daily_cycles = 0.8 # As per objective function
        total_cycles = avg_daily_cycles * 365 * PROJECT_LIFETIME
        replacements_needed = math.ceil(total_cycles / BATTERY_CYCLE_LIFE) - 1
        
        max_solar_emissions = max_solar_panels * SOLAR_PANEL_LIFECYCLE_EMISSIONS
        max_wind_emissions = max_wind_turbines * WIND_TURBINE_LIFECYCLE_EMISSIONS
        max_battery_emissions = max_batteries * BATTERY_LIFECYCLE_EMISSIONS * (1 + replacements_needed)
        
        # Sum max emissions and add a 10% margin
        total_max_emissions = max_solar_emissions + max_wind_emissions + max_battery_emissions
        max_impact = total_max_emissions * 1.1

        # The reference point uses [cost, -reliability, impact]
        return np.array([max_cost, neg_reliability_ref, max_impact])
    

    def pareto_non_dominated(self, points: np.ndarray) -> np.ndarray:
        """Filter to non-dominated solutions (minimization assumed)."""
        if points.size == 0:
            return points
        is_dominated = np.zeros(points.shape[0], dtype=bool)
        for i, p in enumerate(points):
            if is_dominated[i]:
                continue
            dominates = np.all(points <= p, axis=1) & np.any(points < p, axis=1)
            dominates[i] = False
            if np.any(dominates):
                is_dominated[i] = True
        return points[~is_dominated]
    
    def calculate_hypervolume(self, solutions: np.ndarray) -> float:
        """
        Compute hypervolume using pymoo.

        Args:
            solutions: (n_solutions, n_objectives) with objectives [cost, reliability, env_impact].
                    Higher reliability = better (we convert it to minimization).
            reference_point: vector of worst-case objective values (must be *greater* than any solution,
                            in minimization sense). If None, compute from worst observed values + margin.

        Returns:
            float: hypervolume value
        """
        if solutions is None or len(solutions) == 0:
            return 0.0

        # convert reliability (2nd col) to minimization
        pts = solutions.copy().astype(float)
        pts[:, 1] = -pts[:, 1]

        # keep only non-dominated solutions
        nd = self.pareto_non_dominated(pts)
        if len(nd) == 0:
            return 0.0

        # define reference point if not supplied
        if self.reference_point is None:
            worst = np.max(nd, axis=0)
            ref = worst * 1.1 + 1e-9
        else:
            ref = np.asarray(self.reference_point, dtype=float)

        # create HV indicator and compute
        hv = HV(ref_point=ref)
        return float(hv(nd))
   
    
    def calculate_spacing(self, solutions: np.ndarray) -> float:
        """
        Calculate spacing metric for diversity assessment.
        
        Args:
            solutions: Array of solutions
        
        Returns:
            Spacing value (lower indicates more uniform distribution)
        """
        if len(solutions) <= 1:
            return 0.0
        
        # Calculate pairwise distances
        distances = pdist(solutions, metric='euclidean')
        if len(distances) == 0:
            return 0.0
        
        # Create distance matrix
        dist_matrix = squareform(distances)
        
        # Find minimum distance for each solution
        min_distances = []
        for i in range(len(solutions)):
            other_distances = dist_matrix[i, :]
            other_distances = other_distances[other_distances > 1e-10]  # Exclude self
            if len(other_distances) > 0:
                min_distances.append(np.min(other_distances))
        
        if len(min_distances) == 0:
            return 0.0
        
        # Spacing is standard deviation of minimum distances
        mean_distance = np.mean(min_distances)
        spacing = np.sqrt(np.mean([(d - mean_distance) ** 2 for d in min_distances]))
        
        return spacing
    
    def calculate_spread(self, solutions: np.ndarray) -> float:
        """
        Calculate spread metric for extent coverage.
        
        Args:
            solutions: Array of solutions
        
        Returns:
            Spread value (higher indicates better coverage)
        """
        if len(solutions) <= 1:
            return 0.0
        
        # Calculate range for each objective
        ranges = []
        for i in range(solutions.shape[1]):
            obj_range = np.max(solutions[:, i]) - np.min(solutions[:, i])
            ranges.append(obj_range)
        
        # Overall spread is the mean of normalized ranges
        return np.mean(ranges)
    
    def normalize_objectives(self, solutions: np.ndarray, ref_min: np.ndarray, ref_max: np.ndarray) -> np.ndarray:
        """
        Normalize objectives to [0,1] range.
        
        Args:
            solutions: Solutions to normalize
            ref_min: Minimum values for each objective
            ref_max: Maximum values for each objective
        
        Returns:
            Normalized solutions
        """
        normalized = solutions.copy()
        for i in range(solutions.shape[1]):
            if ref_max[i] - ref_min[i] > 1e-10:
                normalized[:, i] = (solutions[:, i] - ref_min[i]) / (ref_max[i] - ref_min[i])
            else:
                normalized[:, i] = 0.0
        return normalized
    
    def run_single_comparison(self, nsga_df: pd.DataFrame, mopso_df: pd.DataFrame, 
                            weighted_df: pd.DataFrame, reference_front: Optional[np.ndarray] = None) -> Dict:
        """
        Run comparison for a single run of all algorithms.
        
        Args:
            nsga_df: NSGA-II results
            mopso_df: MOPSO results  
            weighted_df: Weighted sum results
            reference_front: Optional reference Pareto front for IGD calculation
        
        Returns:
            Dictionary with all metrics
        """
        # Extract objective values
        nsga_vals = nsga_df[['cost', 'reliability', 'environmental_impact']].values
        mopso_vals = mopso_df[['cost', 'reliability', 'environmental_impact']].values
        weighted_vals = weighted_df[['cost', 'reliability', 'environmental_impact']].values
        
        # Calculate reference point if not provided
        if self.reference_point is None:
            all_solutions = np.vstack([nsga_vals, mopso_vals, weighted_vals])
            self.reference_point = np.array([
                np.max(all_solutions[:, 0]) * 1.1,  # Worst cost + 10%
                -np.min(all_solutions[:, 1]) * 0.9,  # Worst reliability (negated) - 10%
                np.max(all_solutions[:, 2]) * 1.1   # Worst env_impact + 10%
            ])
        
        # Use combined front as reference if not provided
        # if reference_front is None:
        #     reference_front = np.vstack([nsga_vals, mopso_vals, weighted_vals])
        
        # Normalize all solutions for fair comparison
        all_solutions = np.vstack([nsga_vals, mopso_vals, weighted_vals])
        ref_min = np.min(all_solutions, axis=0)
        ref_max = np.max(all_solutions, axis=0)
        
        nsga_norm = self.normalize_objectives(nsga_vals, ref_min, ref_max)
        mopso_norm = self.normalize_objectives(mopso_vals, ref_min, ref_max)
        weighted_norm = self.normalize_objectives(weighted_vals, ref_min, ref_max)
        # reference_norm = self.normalize_objectives(reference_front, ref_min, ref_max)
        
        # Calculate metrics
        metrics = {}
        
        # Hypervolume
        metrics['hypervolume'] = {
            'NSGA-II': self.calculate_hypervolume(nsga_vals),
            'MOPSO': self.calculate_hypervolume(mopso_vals),
            'Weighted Sum': self.calculate_hypervolume(weighted_vals)
        }
        
        # IGD (using normalized values)
        # metrics['igd'] = {
        #     'NSGA-II': self.calculate_igd(nsga_norm, reference_norm),
        #     'MOPSO': self.calculate_igd(mopso_norm, reference_norm),
        #     'Weighted Sum': self.calculate_igd(weighted_norm, reference_norm)
        # }
        
        # Spacing
        metrics['spacing'] = {
            'NSGA-II': self.calculate_spacing(nsga_norm),
            'MOPSO': self.calculate_spacing(mopso_norm),
            'Weighted Sum': self.calculate_spacing(weighted_norm)
        }
        
        # Spread
        metrics['spread'] = {
            'NSGA-II': self.calculate_spread(nsga_norm),
            'MOPSO': self.calculate_spread(mopso_norm),
            'Weighted Sum': self.calculate_spread(weighted_norm)
        }
        
        # Number of solutions
        metrics['n_solutions'] = {
            'NSGA-II': len(nsga_df),
            'MOPSO': len(mopso_df),
            'Weighted Sum': len(weighted_df)
        }
        
        # Persist this run's metrics for distribution plots
        self.per_run_metrics.append(metrics)
        return metrics
    
    def compile_multi_run_results(self, multi_run_metrics: List[Dict]) -> pd.DataFrame:
        """
        Compile results from multiple runs into a statistical summary.
        
        Args:
            multi_run_metrics: List of metric dictionaries from multiple runs
        
        Returns:
            DataFrame with statistical summary
        """
        algorithms = ['NSGA-II', 'MOPSO', 'Weighted Sum']
        metrics = ['hypervolume', 'spacing', 'spread', 'n_solutions']
        
        results = []
        
        for metric in metrics:
            for algorithm in algorithms:
                values = [run_metrics[metric][algorithm] for run_metrics in multi_run_metrics]
                
                # Remove invalid values
                values = [v for v in values if not (np.isnan(v) or np.isinf(v))]
                
                if len(values) > 0:
                    results.append({
                        'Algorithm': algorithm,
                        'Metric': metric,
                        'Mean': np.mean(values),
                        'Std': np.std(values),
                        'Min': np.min(values),
                        'Max': np.max(values),
                        'Median': np.median(values),
                        'N_Runs': len(values)
                    })
        
        return pd.DataFrame(results)
    
    def perform_statistical_tests(self, multi_run_metrics: List[Dict]) -> Dict:
        """
        Perform statistical significance tests between algorithms.
        
        Args:
            multi_run_metrics: List of metric dictionaries from multiple runs
        
        Returns:
            Dictionary with statistical test results
        """
        algorithms = ['NSGA-II', 'MOPSO', 'Weighted Sum']
        metrics = ['hypervolume', 'spacing', 'spread']
        
        test_results = {}
        
        for metric in metrics:
            test_results[metric] = {}
            
            # Extract values for each algorithm
            algorithm_values = {}
            for algorithm in algorithms:
                values = [run_metrics[metric][algorithm] for run_metrics in multi_run_metrics]
                values = [v for v in values if not (np.isnan(v) or np.isinf(v))]
                algorithm_values[algorithm] = values
            
            # Kruskal-Wallis test (non-parametric ANOVA)
            if all(len(values) > 1 for values in algorithm_values.values()):
                try:
                    kw_stat, kw_p = kruskal(*algorithm_values.values())
                    test_results[metric]['kruskal_wallis'] = {
                        'statistic': kw_stat,
                        'p_value': kw_p,
                        'significant': kw_p < 0.05
                    }
                except:
                    test_results[metric]['kruskal_wallis'] = None
            
            # Pairwise Mann-Whitney U tests
            test_results[metric]['pairwise'] = {}
            for i, alg1 in enumerate(algorithms):
                for j, alg2 in enumerate(algorithms[i+1:], i+1):
                    if len(algorithm_values[alg1]) > 1 and len(algorithm_values[alg2]) > 1:
                        try:
                            mw_stat, mw_p = mannwhitneyu(
                                algorithm_values[alg1], algorithm_values[alg2], 
                                alternative='two-sided'
                            )
                            test_results[metric]['pairwise'][f'{alg1}_vs_{alg2}'] = {
                                'statistic': mw_stat,
                                'p_value': mw_p,
                                'significant': mw_p < 0.05
                            }
                        except:
                            test_results[metric]['pairwise'][f'{alg1}_vs_{alg2}'] = None
        
        return test_results
    
    def create_comparison_plots(self, results_df: pd.DataFrame, save_dir: str = None, per_run_metrics: Optional[List] = None):
        """
        Create comprehensive comparison plots.

        Args:
            results_df: Statistical summary DataFrame
            save_dir: Directory to save plots
            per_run_metrics: Optional list of per-run metric dicts (one dict per run).
                            If None, falls back to self.per_run_metrics for backward compatibility.
        """
        if save_dir:
            os.makedirs(save_dir, exist_ok=True)

        # Set up the plotting style
        plt.rcParams.update({'font.size': 12})

        # 1. Bar plots with error bars for each metric
        metrics = results_df['Metric'].unique()

        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        axes = axes.flatten()

        for i, metric in enumerate(metrics):
            ax = axes[i]
            metric_data = results_df[results_df['Metric'] == metric]

            algorithms = metric_data['Algorithm'].values
            means = metric_data['Mean'].values
            stds = metric_data['Std'].values

            colors = ['#1f77b4', '#ff7f0e', '#2ca02c']  # Blue, Orange, Green
            bars = ax.bar(algorithms, means, yerr=stds, capsize=5,
                        color=colors[:len(algorithms)], alpha=0.7, edgecolor='black')

            ax.set_title(f'{metric.replace("_", " ").title()}', fontsize=14, fontweight='bold')
            ax.set_ylabel('Value')
            ax.grid(True, alpha=0.3)

            # Add value labels on bars
            for bar, mean, std in zip(bars, means, stds):
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width() / 2., height + std,
                        f'{mean:.3f}±{std:.3f}', ha='center', va='bottom', fontsize=10)

            # Rotate x-axis labels if needed
            ax.tick_params(axis='x', rotation=45)

        # Hide unused subplots
        for j in range(i + 1, len(axes)):
            axes[j].set_visible(False)

        plt.tight_layout()
        if save_dir:
            plt.savefig(os.path.join(save_dir, 'metrics_comparison.png'),
                        dpi=300, bbox_inches='tight')
        plt.show()
        
        # 2. Box plots for distribution comparison over runs
        # Use provided per_run_metrics (two-pass pipeline) or fall back to self.per_run_metrics
        if per_run_metrics is None:
            per_run_metrics = getattr(self, 'per_run_metrics', None) or []

        if not per_run_metrics:
            # No per-run data to plot
            print("No per-run metrics available for distribution plots. Skipping boxplots.")
            return

        # Define algorithms and metrics to plot
        algorithms = ['NSGA-II', 'MOPSO', 'Weighted Sum']
        dist_metrics = ['hypervolume', 'spacing', 'spread']  # keep same metrics as before
        num_plots = len(dist_metrics)
        fig, axes = plt.subplots(1, num_plots, figsize=(6 * num_plots, 6))
        if num_plots == 1:
            axes = [axes]
        colors = ['#1f77b4', '#ff7f0e', '#2ca02c']

        # Prepare data structure
        per_metric_values = {m: {a: [] for a in algorithms} for m in dist_metrics}
        for run in per_run_metrics:
            for m in dist_metrics:
                entries = run.get(m, {}) if run else {}
                for a in algorithms:
                    v = entries.get(a)
                    if v is not None:
                        per_metric_values[m][a].append(v)

        # Map titles and y-labels
        titles = {
            'hypervolume': 'Hypervolume Distribution Across Runs',
            'spacing': 'Spacing Distribution Across Runs',
            'spread': 'Spread Distribution Across Runs'
        }
        ylabels = {
            'hypervolume': 'Hypervolume (higher = better)',
            'spacing': 'Spacing (lower = more uniform)',
            'spread': 'Spread (higher = better)'
        }

        for idx, metric_name in enumerate(dist_metrics):
            ax = axes[idx]
            # build list-of-lists aligned with algorithms (so labels match boxes)
            data = []
            labels = []
            for a in algorithms:
                vals = per_metric_values[metric_name][a]
                if len(vals) == 0:
                    # ensure boxplot gets a non-empty list; mark label to indicate missing data
                    data.append([0.0])
                    labels.append(f"{a}\n(no data)")
                else:
                    data.append(vals)
                    labels.append(a)

            # If all series are empty (shouldn't happen due to earlier guard) skip
            if all(len(v) == 0 or (len(v) == 1 and v[0] == 0.0 and labels[i].endswith("(no data)")) for i, v in enumerate(data)):
                ax.text(0.5, 0.5, "No data", ha='center', va='center')
                ax.set_title(titles.get(metric_name, metric_name))
                continue

            box = ax.boxplot(data, labels=labels, patch_artist=True, showmeans=True, meanline=True)
            # Color boxes (only as many colors as there are boxes)
            for patch, color in zip(box['boxes'], colors):
                patch.set_facecolor(color)
                patch.set_alpha(0.7)
            ax.set_title(titles[metric_name], fontsize=14, fontweight='bold')
            ax.set_ylabel(ylabels[metric_name])
            ax.grid(True, alpha=0.3)
            ax.tick_params(axis='x', rotation=45)

        plt.tight_layout()
        if save_dir:
            plt.savefig(os.path.join(save_dir, 'distribution_comparison.png'),
                        dpi=300, bbox_inches='tight')
        plt.show()
        
    
    def _create_radar_chart(self, results_df: pd.DataFrame, save_dir: str = None):
        """Create radar chart for overall algorithm comparison - DISABLED."""
        # Radar chart functionality removed due to compatibility issues
        pass
    
    def print_statistical_summary(self, results_df: pd.DataFrame, test_results: Dict):
        """Print comprehensive statistical summary."""
        print("\n" + "="*100)
        print("COMPREHENSIVE ALGORITHM COMPARISON - STATISTICAL SUMMARY")
        print("="*100)
        
        # Print mean and std for each metric
        for metric in results_df['Metric'].unique():
            print(f"\n{metric.replace('_', ' ').upper()}:")
            print("-" * 50)
            metric_data = results_df[results_df['Metric'] == metric].sort_values('Mean')
            
            for _, row in metric_data.iterrows():
                print(f"  {row['Algorithm']:15}: {row['Mean']:10.4f} ± {row['Std']:8.4f} "
                      f"(min: {row['Min']:8.4f}, max: {row['Max']:8.4f})")
            
            # Add interpretation
            if metric == 'hypervolume':
                best_alg = metric_data.iloc[-1]['Algorithm']  # Highest is best
                print(f"  → Best: {best_alg} (higher is better)")
            elif metric in ['igd', 'spacing']:
                best_alg = metric_data.iloc[0]['Algorithm']   # Lowest is best
                print(f"  → Best: {best_alg} (lower is better)")
            else:
                best_alg = metric_data.iloc[-1]['Algorithm']  # Highest is best
                print(f"  → Best: {best_alg} (higher is better)")
        
        # Print statistical test results
        print("\n" + "="*80)
        print("STATISTICAL SIGNIFICANCE TESTS")
        print("="*80)
        
        for metric, tests in test_results.items():
            print(f"\n{metric.replace('_', ' ').upper()}:")
            print("-" * 30)
            
            if tests.get('kruskal_wallis'):
                kw = tests['kruskal_wallis']
                significance = "SIGNIFICANT" if kw['significant'] else "NOT SIGNIFICANT"
                print(f"  Kruskal-Wallis test: p = {kw['p_value']:.4f} ({significance})")
            
            if tests.get('pairwise'):
                print("  Pairwise comparisons (Mann-Whitney U):")
                for comparison, result in tests['pairwise'].items():
                    if result:
                        significance = "***" if result['significant'] else "   "
                        print(f"    {comparison:20}: p = {result['p_value']:.4f} {significance}")
                        

def run_multi_run_comparison(run_function, n_runs: int = 10, config_path: str = None, 
                             output_dir: str = None) -> Tuple[pd.DataFrame, Dict]:
    """
    Two-pass approach: run experiments (collect all fronts), then compute metrics
    using a single global normalization and reference point for HV comparability.
    """
    print(f"\n{'='*80}")
    print(f"STARTING MULTI-RUN COMPARISON ({n_runs} runs)")
    print(f"{'='*80}")

    comparison = AlgorithmComparison()
    per_run_raw = []  # store raw fronts for each run
    successful_runs = 0

    for run_i in range(n_runs):
        print(f"\nRun {run_i + 1}/{n_runs}...")
        try:
            nsga_df, mopso_df, weighted_df = run_function(config_path)
            # extract numpy arrays
            nsga_vals = nsga_df[['cost', 'reliability', 'environmental_impact']].values
            mopso_vals = mopso_df[['cost', 'reliability', 'environmental_impact']].values
            weighted_vals = weighted_df[['cost', 'reliability', 'environmental_impact']].values

            per_run_raw.append({
                'NSGA-II': nsga_vals,
                'MOPSO': mopso_vals,
                'Weighted Sum': weighted_vals
            })

            successful_runs += 1
            print(f"  Completed run {run_i + 1}/{n_runs}")
        except Exception as e:
            print(f"  Run {run_i + 1} failed: {e}")
            continue

    if successful_runs == 0:
        raise RuntimeError("All runs failed!")

    print(f"\nCollected data for {successful_runs} runs. Computing global reference & metrics...")

    # Build global combined array to compute normalization bounds (min/max across all runs and algos)
    all_points = []
    for run_dict in per_run_raw:
        for arr in run_dict.values():
            if arr is None or len(arr) == 0:
                continue
            all_points.append(arr)
    all_points = np.vstack(all_points)
    # Convert reliability to minimization (higher reliability better -> negate)
    all_points[:, 1] = -all_points[:, 1]

    global_min = np.min(all_points, axis=0)
    global_max = np.max(all_points, axis=0)

    # We'll normalize to [0,1] using global_min/global_max and use ref_point slightly > 1.0
    def normalize(points):
        pts = points.copy().astype(float)
        pts[:, 1] = -pts[:, 1]  # reliability -> minimization
        norm = np.zeros_like(pts)
        for i in range(pts.shape[1]):
            denom = global_max[i] - global_min[i]
            if denom > 1e-12:
                norm[:, i] = (pts[:, i] - global_min[i]) / denom
            else:
                norm[:, i] = 0.0
        return norm

    ref_point = np.ones(all_points.shape[1]) * 1.1  # slightly outside [0,1]

    comparison.reference_point = ref_point

    # compute metrics for each run (HV on normalized non-dominated fronts)
    multi_run_metrics = []
    for run_dict in per_run_raw:
        metrics = {}
        hv_dict = {}
        spacing_dict = {}
        spread_dict = {}
        n_solutions = {}

        for alg_name, arr in run_dict.items():
            if arr is None or len(arr) == 0:
                hv_val = 0.0
                nd_norm = np.zeros((0, all_points.shape[1]))
            else:
                norm_front = normalize(arr)
                nd_norm = comparison.pareto_non_dominated(norm_front)
                # compute hv with pymoo on nd_norm (already minimization space because we negated reliability)
                hv_val = comparison.calculate_hypervolume(nd_norm) if len(nd_norm) > 0 else 0.0

            hv_dict[alg_name] = hv_val
            # store counts and placeholders for other metrics (recompute those if you want)
            n_solutions[alg_name] = len(arr) if arr is not None else 0

            # if you want to compute spacing/spread on normalized nondominated front:
            spacing_dict[alg_name] = comparison.calculate_spacing(nd_norm)  # ensure function accepts numpy arrays
            spread_dict[alg_name]  = comparison.calculate_spread(nd_norm)

        metrics['hypervolume'] = hv_dict
        metrics['n_solutions'] = n_solutions
        metrics['spacing'] = spacing_dict
        metrics['spread'] = spread_dict

        multi_run_metrics.append(metrics)

    # Now you can compile and run tests using the same code as before
    results_df = comparison.compile_multi_run_results(multi_run_metrics)
    test_results = comparison.perform_statistical_tests(multi_run_metrics)

    # plotting/saving same as before
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        comparison.create_comparison_plots(results_df, output_dir, per_run_metrics=multi_run_metrics)
        results_df.to_csv(os.path.join(output_dir, 'multi_run_comparison.csv'), index=False)

    comparison.print_statistical_summary(results_df, test_results)
    return results_df, test_results
