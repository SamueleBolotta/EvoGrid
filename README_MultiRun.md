# Multi-Run Algorithm Comparison

This document describes how to perform comprehensive comparison of the three optimization algorithms (NSGA-II, MOPSO, Weighted Sum) with statistical analysis across multiple independent runs.

## Overview

The multi-run comparison framework provides:

1. **Statistical Analysis**: Mean, standard deviation, min, max across multiple runs
2. **Quality Metrics**: 
   - Hypervolume (solution set quality)
   - IGD - Inverted Generational Distance (convergence)
   - Spacing (diversity/uniformity)
   - Spread (extent coverage)
3. **Statistical Tests**: Kruskal-Wallis and Mann-Whitney U tests for significance
4. **Comprehensive Visualization**: Bar charts, radar charts, distribution plots

## Quick Start

### Basic Multi-Run Comparison (10 runs)

```bash
python multi_run_experiment.py --config config_multirun.yaml --runs 10
```

### Custom Output Directory

```bash
python multi_run_experiment.py --config config_multirun.yaml --runs 30 --output-dir results/comparison_30runs
```

### Different Random Seed Base

```bash
python multi_run_experiment.py --config config_multirun.yaml --runs 15 --base-seed 123
```

## Understanding the Metrics

### Hypervolume
- **Higher is better**
- Measures the volume of objective space dominated by the solution set
- Combines convergence and diversity in a single metric
- Most comprehensive quality indicator

### IGD (Inverted Generational Distance)
- **Lower is better**
- Measures average distance from reference Pareto front to obtained front
- Indicates convergence quality
- Good for comparing closeness to true/ideal Pareto front

### Spacing
- **Lower is better**
- Measures uniformity of solution distribution
- Lower values indicate more evenly distributed solutions
- Important for decision maker choice diversity

### Spread
- **Higher is better**
- Measures extent of coverage in objective space
- Higher values indicate broader coverage of trade-offs

## Output Files

### Statistical Summary Files
- `multi_run_comparison.csv`: Raw statistical data (means, std, etc.)
- `statistical_summary.txt`: Human-readable summary with interpretations

### Visualization Files
- `metrics_comparison.png`: Bar charts with error bars for all metrics
- `distribution_comparison.png`: Distribution comparison plots

## Configuration

The framework uses the same YAML configuration as single runs but with optimized parameters for faster multi-run execution:

```yaml
nsga:
  pop_size: 50        # Reduced for faster runs
  num_generations: 30 # Reduced for faster runs

mopso:
  swarm_size: 50
  max_iterations: 30

weighted:
  pop_size: 50
  num_generations: 30
```

## Example Results Interpretation

### Sample Output:
```
HYPERVOLUME:
  NSGA-II        :     1.2345 ±   0.0123 (min:   1.2100, max:   1.2500)
  MOPSO          :     1.1234 ±   0.0234 (min:   1.0800, max:   1.1600)
  Weighted Sum   :     0.9876 ±   0.0345 (min:   0.9200, max:   1.0200)
  → Best: NSGA-II (higher is better)

STATISTICAL SIGNIFICANCE TESTS
IGD:
  Kruskal-Wallis test: p = 0.0023 (SIGNIFICANT)
  Pairwise comparisons (Mann-Whitney U):
    NSGA-II_vs_MOPSO      : p = 0.0156 ***
    NSGA-II_vs_Weighted Sum: p = 0.0008 ***
    MOPSO_vs_Weighted Sum : p = 0.0234 ***
```

**Interpretation:**
- NSGA-II achieved the highest hypervolume with low variance (consistent performance)
- All algorithms show statistically significant differences (p < 0.05)
- NSGA-II significantly outperforms both other algorithms

## Advanced Usage

### Programmatic Use

```python
from analysis.algorithm_comparison import run_multi_run_comparison
from multi_run_experiment import single_run_all_algorithms

# Define run function
def my_run_function(config_path):
    return single_run_all_algorithms(config_path, seed=None)

# Run comparison
results_df, test_results = run_multi_run_comparison(
    run_function=my_run_function,
    n_runs=20,
    config_path="config_multirun.yaml",
    output_dir="my_results"
)

# Access results
print(results_df)
print(test_results)
```

### Custom Metrics Analysis

```python
from analysis.algorithm_comparison import AlgorithmComparison

comparison = AlgorithmComparison()

# Calculate metrics for single run
metrics = comparison.run_single_comparison(nsga_df, mopso_df, weighted_df)

print(f"NSGA-II Hypervolume: {metrics['hypervolume']['NSGA-II']}")
print(f"MOPSO IGD: {metrics['igd']['MOPSO']}")
```

## Best Practices

### Number of Runs
- **Minimum**: 10 runs for basic statistical validity
- **Recommended**: 30 runs for robust statistical analysis
- **Publication Quality**: 50+ runs for strong statistical confidence

### Configuration Tuning
- Use smaller population sizes and generations for multi-run (faster execution)
- Ensure all algorithms use similar computational budgets
- Keep parameters consistent across runs (only vary random seeds)

### Result Interpretation
- Focus on effect sizes, not just statistical significance
- Consider practical significance (is the difference meaningful?)
- Look at variance - consistent algorithms are often preferred
- Use multiple metrics - no single metric tells the complete story

## Troubleshooting

### Memory Issues
- Reduce population sizes and generations
- Run fewer algorithms simultaneously
- Clear memory between runs

### Slow Execution
- Use the optimized config (`config_multirun.yaml`)
- Reduce number of generations/iterations
- Run on multiple cores if available

### Statistical Issues
- Ensure enough runs for valid statistics (minimum 10)
- Check for outliers that might skew results
- Use non-parametric tests (already implemented) for robustness

## Performance Expectations

With the optimized configuration (`config_multirun.yaml`):
- **Single run time**: ~30-60 seconds
- **10 runs**: ~5-10 minutes
- **30 runs**: ~15-30 minutes

Times may vary based on system performance and problem complexity.
