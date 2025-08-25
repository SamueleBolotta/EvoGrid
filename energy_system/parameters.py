"""
Energy system parameters for EvoGrid.

Contains all physical and economic parameters for solar panels, wind turbines,
batteries, and system constraints.
"""

# Solar panel parameters
SOLAR_PANEL_CAPACITY = 0.4  # kW per panel
SOLAR_PANEL_COST = 300      # $ per panel
SOLAR_PANEL_LAND_USE = 3    # m² per panel
SOLAR_PANEL_LIFECYCLE_EMISSIONS = 40  # kg CO2 per panel

# Wind turbine parameters
WIND_TURBINE_CAPACITY = 3.0   # kW per turbine
WIND_TURBINE_COST = 6000      # $ per turbine 
WIND_TURBINE_LAND_USE = 100   # m² per turbine
WIND_TURBINE_LIFECYCLE_EMISSIONS = 800  # kg CO2 per turbine

# Battery storage parameters
BATTERY_CAPACITY = 5.0   # kWh per battery unit
BATTERY_COST = 2000      # $ per battery unit
BATTERY_CYCLE_LIFE = 4000  # cycles before replacement
BATTERY_LIFECYCLE_EMISSIONS = 150  # kg CO2 per battery unit

# System constraints
MAX_LAND_AREA = 10000    # m²
MAX_BUDGET = 200000      # $
MIN_ENERGY_REQUIREMENT = 50  # kWh/day
PROJECT_LIFETIME = 20    # years

# Default optimization bounds
BOUNDS = [(0, 500), (0, 50), (0, 50)]  # (solar panels, wind turbines, batteries)
