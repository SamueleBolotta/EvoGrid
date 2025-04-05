# EvoGrid

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

EvoGrid is a bio-inspired approach to renewable energy system optimization using NSGA-II (Non-dominated Sorting Genetic Algorithm II). This project demonstrates how evolutionary algorithms can find optimal configurations for renewable energy systems by balancing multiple competing objectives.

## Overview

This project uses genetic algorithms to design optimal renewable energy systems that balance:
- Cost minimization
- Reliability maximization
- Environmental impact minimization

By leveraging multi-objective optimization, EvoGrid identifies a range of Pareto-optimal solutions that represent different trade-offs between these competing objectives.

## Bio-inspired approach

The core of EvoGrid is the NSGA-II algorithm, which mimics natural selection and evolution to find optimal solutions:

- **Population-based search**: Maintains a population of diverse system configurations
- **Multi-objective fitness**: Evaluates solutions based on multiple criteria simultaneously
- **Pareto optimality**: Identifies solutions where improving one objective would worsen another
- **Genetic operators**: Uses crossover and mutation to explore the design space
- **Elitism**: Preserves the best solutions across generations

## Energy system model

The project models a renewable energy system with:

- **Solar panels**: Convert solar irradiance to electricity
- **Wind turbines**: Generate electricity from wind
- **Battery storage**: Store excess energy for use during deficit periods

The model simulates system performance using synthetic weather data (solar irradiance and wind speed) and energy demand patterns over a full year to realistically assess performance.

## Key features

- Multi-objective optimization using NSGA-II
- Realistic energy system simulation
- Detailed analysis of Pareto-optimal solutions
- Sensitivity analysis for key parameters
- Visualization tools for result interpretation

## Getting started

### Prerequisites

- Python 3.7+
- Required packages: numpy, pandas, matplotlib, deap, scipy, seaborn

### Installation

```bash
# Clone the repository
git clone https://github.com/SamueleBolotta/EvoGrid.git
cd EvoGrid

# Create a virtual environment (optional)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install numpy pandas matplotlib deap scipy seaborn tqdm
```

### Running the Optimization

```bash
# Run the main optimization
python main.py

# Run detailed analysis on selected solutions
python utils.py
```

## Results interpretation

The optimization produces a set of Pareto-optimal solutions, each representing a different balance between cost, reliability, and environmental impact. 

Key result files include:
- Pareto front visualizations (2D and 3D)
- Detailed analysis of selected solutions
- Technology mix comparisons
- Sensitivity analysis for key parameters

### Example solution categories

- **Lowest cost**: 34 solar panels, 7 wind turbines, 2 batteries - $87,320 with 87.8% reliability
- **Highest reliability (cost-effective)**: 125 solar panels, 5 wind turbines, 5 batteries - $115,100 with 100% reliability
- **Lowest emissions**: 25 solar panels, 18 wind turbines, 0 batteries - $183,300 with 95.5% reliability
- **Balanced solution**: 82 solar panels, 7 wind turbines, 2 batteries - $107,480 with 99.6% reliability

## Project structure

- `main.py`: Core optimization algorithm and energy system modeling
- `utils.py`: Analysis and visualization utilities
- `/results`: Output visualizations and data (create this directory)

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## References

- Deb, K., Pratap, A., Agarwal, S., & Meyarivan, T. (2002). A fast and elitist multiobjective genetic algorithm: NSGA-II. IEEE transactions on evolutionary computation, 6(2), 182-197.
- DEAP (Distributed Evolutionary Algorithms in Python) documentation: https://deap.readthedocs.io/
