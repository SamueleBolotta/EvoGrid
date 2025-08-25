# EvoGrid

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

EvoGrid is a comprehensive bio-inspired framework for renewable energy system optimization using multiple evolutionary algorithms. This project demonstrates how different optimization approaches (NSGA-II, MOPSO, and weighted-sum GA) can find optimal configurations for renewable energy systems by balancing multiple competing objectives.

## Overview

This project uses genetic algorithms to design optimal renewable energy systems that balance:
- Cost minimization
- Reliability maximization
- Environmental impact minimization

By leveraging multi-objective optimization, EvoGrid identifies a range of Pareto-optimal solutions that represent different trade-offs between these competing objectives.

## Optimization Algorithms

EvoGrid implements three different optimization approaches:

### NSGA-II (Non-dominated Sorting Genetic Algorithm II)
- **Population-based search**: Maintains a population of diverse system configurations
- **Multi-objective fitness**: Evaluates solutions based on multiple criteria simultaneously
- **Pareto optimality**: Identifies solutions where improving one objective would worsen another
- **Genetic operators**: Uses crossover and mutation to explore the design space
- **Elitism**: Preserves the best solutions across generations

### MOPSO (Multi-Objective Particle Swarm Optimization)
- **Swarm intelligence**: Particles move through the solution space following personal and global best solutions
- **Archive-based**: Maintains an external archive of non-dominated solutions
- **Velocity-based**: Uses particle velocity and position updates for exploration

### Weighted-Sum GA (Baseline)
- **Single-objective**: Combines multiple objectives into a single weighted sum
- **Benchmark comparison**: Provides baseline performance for comparison with multi-objective methods

## Energy system model

The project models a renewable energy system with:

- **Solar panels**: Convert solar irradiance to electricity
- **Wind turbines**: Generate electricity from wind
- **Battery storage**: Store excess energy for use during deficit periods

The model simulates system performance using synthetic weather data (solar irradiance and wind speed) and energy demand patterns over a full year to realistically assess performance.

## Key features

- **Multiple optimization algorithms**: NSGA-II, MOPSO, and weighted-sum GA
- **Modular architecture**: Clean separation of algorithms, energy system modeling, analysis, and visualization
- **Realistic energy system simulation**: Year-long simulation with synthetic weather data
- **Comprehensive analysis**: Detailed solution analysis with energy balance, deficit analysis, and performance metrics
- **Advanced sensitivity analysis**: Parameter robustness testing for cost, capacity, and emissions parameters
- **Rich visualizations**: Pareto fronts, composition analysis, method comparisons, and sensitivity plots
- **Configurable parameters**: YAML-based configuration for easy experimentation
- **Performance metrics**: Hypervolume, spacing, and diversity analysis for algorithm comparison

## Getting started

### Prerequisites

- Python 3.9+
- Conda (recommended) or pip for package management

### Installation

```bash
# Clone the repository
git clone https://github.com/SamueleBolotta/EvoGrid.git
cd EvoGrid

# Create conda environment (recommended)
conda env create -f conda-env.yml
conda activate evogrid

# Or install with pip
pip install numpy pandas matplotlib deap scipy seaborn tqdm
```

### Running the Optimization

```bash
# Run the complete optimization suite (all three algorithms)
python main.py

# Run with custom configuration
python main.py --config config.yaml

# Run with custom output directory
python main.py --output-dir my_results

# Run with custom CSV data
python main.py --csv-path my_data.csv
```

### Configuration

Customize optimization parameters in `config.yaml`:

```yaml
# Algorithm parameters
nsga:
  pop_size: 100
  num_generations: 50
  cxpb: 0.7
  mutpb: 0.2

mopso:
  swarm_size: 100
  max_iterations: 50
  archive_size: 100

weighted:
  weights: [0.6, 0.3, 0.1]  # [cost, reliability, env_impact]

# Output settings
output:
  base_dir: "results"
```

## Results interpretation

The optimization produces comprehensive results comparing all three algorithms:

### Generated Files
- **Pareto front visualizations**: 2D/3D plots showing trade-offs between objectives
- **Algorithm comparison**: Performance metrics (hypervolume, spacing, solution count)
- **Method comparison plots**: Visual comparison of NSGA-II vs MOPSO vs Weighted-sum
- **Solution composition analysis**: Technology mix and capacity distributions
- **Detailed solution analysis**: Energy balance, deficit analysis, and performance metrics
- **Sensitivity analysis**: Parameter robustness testing with visualization
- **CSV exports**: All solutions and metrics for further analysis

### Output Structure
```
results/
├── nsga2/
│   └── YYYYMMDD_HHMMSS/
│       ├── pareto.csv
│       ├── pareto_pareto2d.png
│       ├── pareto_pareto3d.png
│       ├── methods_comparison.png
│       ├── methods_detailed_comparison.png
│       ├── composition_*.png
│       ├── sensitivity_*.png
│       ├── details/
│       │   └── *_detailed_analysis.png
│       └── config.used.yaml
```

### Example solution categories

- **Lowest cost**: 34 solar panels, 7 wind turbines, 2 batteries - $87,320 with 87.8% reliability
- **Highest reliability (cost-effective)**: 125 solar panels, 5 wind turbines, 5 batteries - $115,100 with 100% reliability
- **Lowest emissions**: 25 solar panels, 18 wind turbines, 0 batteries - $183,300 with 95.5% reliability
- **Balanced solution**: 82 solar panels, 7 wind turbines, 2 batteries - $107,480 with 99.6% reliability

## Project structure

```
EvoGrid/
├── main.py                    # Main execution script
├── config.py                  # Configuration management
├── config.yaml                # Default configuration
├── conda-env.yml              # Conda environment specification
├── algorithms/                # Optimization algorithms
│   ├── nsga2.py               #   NSGA-II implementation
│   ├── mopso.py               #   MOPSO implementation
│   └── weighted_sum.py        #   Weighted-sum GA
├── energy_system/             # Energy system modeling
│   ├── parameters.py          #   System parameters
│   ├── objective_functions.py #   Cost, reliability, env. impact
│   ├── constraints.py         #   System constraints
│   ├── simulation.py          #   Energy balance simulation
│   └── production.py          #   Solar/wind production models
├── data_generation/           # Data generation and loading
│   ├── weather.py             #   Weather data synthesis
│   ├── demand.py              #   Energy demand patterns
│   └── data_loader.py         #   Data loading utilities
├── analysis/                   # Results analysis
│   ├── results_analyzer.py     # Pareto front analysis
│   ├── sensitivity_analysis.py # Parameter sensitivity
│   └── solution_analysis.py    # Detailed solution analysis
├── visualization/           # Plotting and visualization
│   ├── pareto_plots.py      #   Pareto front visualization
│   ├── comparison_plots.py  #   Algorithm comparison plots
│   └── composition_plots.py #   Solution composition plots
├── utils.py                 # Legacy (for reference) TO REMOVE
└── results/                 # Generated outputs and visualizations
```

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## References

- Deb, K., Pratap, A., Agarwal, S., & Meyarivan, T. (2002). A fast and elitist multiobjective genetic algorithm: NSGA-II. IEEE transactions on evolutionary computation, 6(2), 182-197.
- Coello, C. A. C., Pulido, G. T., & Lechuga, M. S. (2004). Handling multiple objectives with particle swarm optimization. IEEE Transactions on evolutionary computation, 8(3), 256-279.
- DEAP (Distributed Evolutionary Algorithms in Python) documentation: https://deap.readthedocs.io/
- Das, I., & Dennis, J. E. (1998). Normal-boundary intersection: A new method for generating the Pareto surface in nonlinear multicriteria optimization problems. SIAM journal on optimization, 8(3), 631-657.
