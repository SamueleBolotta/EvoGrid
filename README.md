# Multi-objective optimization for sustainable energy system design

## Project overview
This project implements a multi-objective optimization framework for designing a renewable energy system that balances three competing objectives:
1. Minimizing total system cost
2. Maximizing energy production reliability
3. Minimizing environmental impact based on lifecycle analysis

The optimization uses the NSGA-II algorithm to find the Pareto-optimal configurations of solar panels, wind turbines, and battery storage.

## Requirements
```
numpy
pandas
matplotlib
seaborn
deap
scipy
tqdm
```

You can install all required packages with:
```
pip install numpy pandas matplotlib seaborn deap scipy tqdm
```

## Code structure

The implementation consists of the following main components:

1. **Energy system parameters**: Defines the technical and economic parameters for solar panels, wind turbines, and batteries.

2. **Weather and demand data generation**: Creates synthetic data for solar irradiance, wind speed, and energy demand.

3. **Energy system modeling**: Simulates the performance of an energy system configuration over time.

4. **Objective functions**:
   - `calculate_total_cost`: Computes the total lifecycle cost
   - `calculate_reliability`: Evaluates the energy supply reliability
   - `calculate_environmental_impact`: Assesses the environmental impact

5. **Constraints**:
   - Land use constraint
   - Budget constraint
   - Minimum energy requirement

6. **NSGA-II implementation**: Sets up and runs the multi-objective optimization algorithm.

7. **Analysis and visualization**: Analyzes and visualizes the optimization results, including:
   - Pareto front visualization
   - Technology mix analysis
   - Simulation of selected solutions
   - Sensitivity analysis

## How to run

1. Run the entire script to perform the optimization and generate visualizations:
```
python sustainable_energy_optimization.py
```

2. The script will:
   - Generate synthetic weather and demand data
   - Run the NSGA-II optimization for 50 generations with a population size of 100
   - Analyze and visualize the Pareto-optimal solutions
   - Perform sensitivity analysis on key parameters

## Modifying the model

You can modify various aspects of the model:

- **System parameters**: Adjust costs, capacities, and environmental impacts at the top of the script
- **Optimization parameters**: Change the population size and number of generations in the `run_nsga2` function call
- **Constraints**: Modify the constraints in the `check_constraints` function
- **Objective weights**: Change the weights in the `creator.create("FitnessMulti",...)` line to prioritize different objectives

## Interpreting results

The optimization produces a set of Pareto-optimal solutions, where no solution is strictly better than another across all objectives. The visualizations help to understand:

1. **Pareto front**: Shows the trade-offs between the three objectives
2. **Solutions composition**: Visualizes the mix of technologies in each solution
3. **Performance simulation**: Shows the energy balance over time for selected solutions
4. **Sensitivity analysis**: Evaluates how changes in key parameters affect the performance of a solution

## Extending the model

The model can be extended in several ways:
- Add more renewable energy sources (e.g., hydropower, geothermal)
- Incorporate more detailed weather models
- Add more detailed financial models (e.g., loans, subsidies)
- Include grid integration aspects
- Model degradation of components over time
- Add more detailed constraints (e.g., maximum noise levels, visual impact)
