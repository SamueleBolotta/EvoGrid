"""
Algorithm comparison visualization for multi-objective optimization.

Contains functions for comparing and visualizing results from different
optimization algorithms.
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.distance import pdist, squareform


def compare_methods(nsga_df, weighted_df, save_path=None, mopso_df=None):
    """Plot NSGA-II Pareto solutions vs. Weighted-sum solutions vs. MOPSO solutions.
    Expects DataFrames with columns: ['cost','reliability','environmental_impact'].
    """
    nsga_vals = nsga_df[['cost','reliability','environmental_impact']].values
    weighted_vals = weighted_df[['cost','reliability','environmental_impact']].values
    
    plt.figure(figsize=(12, 4))

    plt.subplot(131)
    plt.scatter(nsga_vals[:,0], nsga_vals[:,1], label='NSGA-II', alpha=0.7, s=50)
    plt.scatter(weighted_vals[:,0], weighted_vals[:,1], label='Weighted Sum', marker='x', s=100)
    if mopso_df is not None:
        mopso_vals = mopso_df[['cost','reliability','environmental_impact']].values
        plt.scatter(mopso_vals[:,0], mopso_vals[:,1], label='MOPSO', marker='^', alpha=0.7, s=50)
    plt.xlabel('Cost ($)'); plt.ylabel('Reliability'); plt.legend(); plt.grid(True, alpha=0.3)

    plt.subplot(132)
    plt.scatter(nsga_vals[:,0], nsga_vals[:,2], label='NSGA-II', alpha=0.7, s=50)
    plt.scatter(weighted_vals[:,0], weighted_vals[:,2], label='Weighted Sum', marker='x', s=100)
    if mopso_df is not None:
        plt.scatter(mopso_vals[:,0], mopso_vals[:,2], label='MOPSO', marker='^', alpha=0.7, s=50)
    plt.xlabel('Cost ($)'); plt.ylabel('Environmental Impact'); plt.legend(); plt.grid(True, alpha=0.3)

    plt.subplot(133)
    plt.scatter(nsga_vals[:,1], nsga_vals[:,2], label='NSGA-II', alpha=0.7, s=50)
    plt.scatter(weighted_vals[:,1], weighted_vals[:,2], label='Weighted Sum', marker='x', s=100)
    if mopso_df is not None:
        plt.scatter(mopso_vals[:,1], mopso_vals[:,2], label='MOPSO', marker='^', alpha=0.7, s=50)
    plt.xlabel('Reliability'); plt.ylabel('Environmental Impact'); plt.legend(); plt.grid(True, alpha=0.3)

    plt.tight_layout()
    if save_path:
        plt.savefig(f"{save_path}_comparison.png", dpi=300, bbox_inches='tight')
    plt.show()


def compare_all_methods_detailed(nsga_df, weighted_df, mopso_df, save_path=None):
    """Create detailed comparison of all three optimization methods."""
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    
    nsga_vals = nsga_df[['cost','reliability','environmental_impact']].values
    weighted_vals = weighted_df[['cost','reliability','environmental_impact']].values
    mopso_vals = mopso_df[['cost','reliability','environmental_impact']].values
    
    # Objective space comparisons (top row)
    objectives = [('Cost ($)', 'Reliability'), ('Cost ($)', 'Environmental Impact'), ('Reliability', 'Environmental Impact')]
    indices = [(0, 1), (0, 2), (1, 2)]
    
    for i, ((xlabel, ylabel), (idx1, idx2)) in enumerate(zip(objectives, indices)):
        ax = axes[0, i]
        ax.scatter(nsga_vals[:,idx1], nsga_vals[:,idx2], label='NSGA-II', alpha=0.7, s=50, c='blue')
        ax.scatter(weighted_vals[:,idx1], weighted_vals[:,idx2], label='Weighted Sum', marker='x', s=100, c='red')
        ax.scatter(mopso_vals[:,idx1], mopso_vals[:,idx2], label='MOPSO', marker='^', alpha=0.7, s=50, c='green')
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_title(f'{xlabel} vs {ylabel}')
    
    # Performance metrics (bottom row)
    
    # Hypervolume estimation (simplified)
    ax = axes[1, 0]
    methods = ['NSGA-II', 'MOPSO', 'Weighted Sum']
    
    # Simple hypervolume approximation using dominated volume
    def simple_hypervolume(solutions, ref_point):
        """Simple hypervolume calculation."""
        if len(solutions) == 0:
            return 0
        # Convert to minimization problem
        min_solutions = solutions.copy()
        min_solutions[:, 1] = -min_solutions[:, 1]  # Negate reliability for minimization
        
        volume = 0
        for sol in min_solutions:
            # Calculate dominated volume
            dom_vol = np.prod(np.maximum(0, ref_point - sol))
            volume += dom_vol
        return volume
    
    # Reference point (worst case for each objective)
    all_solutions = np.vstack([nsga_vals, mopso_vals, weighted_vals])
    ref_point = np.array([
        np.max(all_solutions[:, 0]) * 1.1,  # Worst cost
        -np.min(all_solutions[:, 1]) * 0.9,  # Worst reliability (negated)
        np.max(all_solutions[:, 2]) * 1.1   # Worst environmental impact
    ])
    
    hv_nsga = simple_hypervolume(nsga_vals, ref_point)
    hv_mopso = simple_hypervolume(mopso_vals, ref_point)
    hv_weighted = simple_hypervolume(weighted_vals, ref_point)
    
    hypervolumes = [hv_nsga, hv_mopso, hv_weighted]
    colors = ['blue', 'green', 'red']
    bars = ax.bar(methods, hypervolumes, color=colors, alpha=0.7)
    ax.set_ylabel('Hypervolume (approximation)')
    ax.set_title('Hypervolume Comparison')
    ax.grid(True, alpha=0.3)
    
    # Add values on bars
    for bar, hv in zip(bars, hypervolumes):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{hv:.2e}', ha='center', va='bottom')
    
    # Number of solutions
    ax = axes[1, 1]
    solution_counts = [len(nsga_df), len(mopso_df), len(weighted_df)]
    bars = ax.bar(methods, solution_counts, color=colors, alpha=0.7)
    ax.set_ylabel('Number of Solutions')
    ax.set_title('Solution Count Comparison')
    ax.grid(True, alpha=0.3)
    
    # Add values on bars
    for bar, count in zip(bars, solution_counts):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{count}', ha='center', va='bottom')
    
    # Diversity analysis (spacing metric)
    ax = axes[1, 2]
    
    def calculate_spacing(solutions):
        """Calculate spacing metric for diversity assessment."""
        if len(solutions) <= 1:
            return 0
        
        distances = pdist(solutions)
        if len(distances) == 0:
            return 0
        
        min_distances = []
        dist_matrix = squareform(distances)
        
        for i in range(len(solutions)):
            # Find minimum distance to other solutions
            other_dists = dist_matrix[i, :]
            other_dists = other_dists[other_dists > 0]  # Exclude self-distance
            if len(other_dists) > 0:
                min_distances.append(np.min(other_dists))
        
        if len(min_distances) == 0:
            return 0
        
        # Spacing is the standard deviation of minimum distances
        return np.std(min_distances)
    
    spacing_nsga = calculate_spacing(nsga_vals)
    spacing_mopso = calculate_spacing(mopso_vals)
    spacing_weighted = calculate_spacing(weighted_vals)
    
    spacings = [spacing_nsga, spacing_mopso, spacing_weighted]
    bars = ax.bar(methods, spacings, color=colors, alpha=0.7)
    ax.set_ylabel('Spacing (Diversity)')
    ax.set_title('Solution Diversity (Lower = More Uniform)')
    ax.grid(True, alpha=0.3)
    
    # Add values on bars
    for bar, spacing in zip(bars, spacings):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{spacing:.3f}', ha='center', va='bottom')
    
    plt.tight_layout()
    if save_path:
        plt.savefig(f"{save_path}_detailed_comparison.png", dpi=300, bbox_inches='tight')
    plt.show()
    
    # Print summary statistics
    print("\n" + "="*80)
    print("OPTIMIZATION METHODS COMPARISON SUMMARY")
    print("="*80)
    
    print(f"\nNSGA-II:")
    print(f"  • Solutions found: {len(nsga_df)}")
    print(f"  • Cost range: ${nsga_vals[:,0].min():.0f} - ${nsga_vals[:,0].max():.0f}")
    print(f"  • Reliability range: {nsga_vals[:,1].min():.3f} - {nsga_vals[:,1].max():.3f}")
    print(f"  • Env. impact range: {nsga_vals[:,2].min():.6f} - {nsga_vals[:,2].max():.6f}")
    print(f"  • Hypervolume: {hv_nsga:.2e}")
    print(f"  • Spacing (diversity): {spacing_nsga:.3f}")
    
    print(f"\nMOPSO:")
    print(f"  • Solutions found: {len(mopso_df)}")
    print(f"  • Cost range: ${mopso_vals[:,0].min():.0f} - ${mopso_vals[:,0].max():.0f}")
    print(f"  • Reliability range: {mopso_vals[:,1].min():.3f} - {mopso_vals[:,1].max():.3f}")
    print(f"  • Env. impact range: {mopso_vals[:,2].min():.6f} - {mopso_vals[:,2].max():.6f}")
    print(f"  • Hypervolume: {hv_mopso:.2e}")
    print(f"  • Spacing (diversity): {spacing_mopso:.3f}")
    
    print(f"\nWeighted Sum:")
    print(f"  • Solutions found: {len(weighted_df)}")
    print(f"  • Cost range: ${weighted_vals[:,0].min():.0f} - ${weighted_vals[:,0].max():.0f}")
    print(f"  • Reliability range: {weighted_vals[:,1].min():.3f} - {weighted_vals[:,1].max():.3f}")
    print(f"  • Env. impact range: {weighted_vals[:,2].min():.6f} - {weighted_vals[:,2].max():.6f}")
    print(f"  • Hypervolume: {hv_weighted:.2e}")
    print(f"  • Spacing (diversity): {spacing_weighted:.3f}")
    
    return {
        'hypervolumes': {'NSGA-II': hv_nsga, 'MOPSO': hv_mopso, 'Weighted Sum': hv_weighted},
        'spacings': {'NSGA-II': spacing_nsga, 'MOPSO': spacing_mopso, 'Weighted Sum': spacing_weighted},
        'solution_counts': {'NSGA-II': len(nsga_df), 'MOPSO': len(mopso_df), 'Weighted Sum': len(weighted_df)}
    }
