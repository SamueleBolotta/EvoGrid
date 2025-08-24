import yaml


def load_config(config_path: str) -> dict:
    """Load YAML config file into a plain Python dict.

    Keep this intentionally simple so it is easy to read and extend.
    """
    with open(config_path, "r") as f:
        return yaml.safe_load(f) or {}


def apply_to_main(cfg: dict, main_mod):
    """Apply config values to globals defined in main module.

    This preserves the current project design (globals used across modules)
    while allowing experiments to override parameters cleanly.
    """
    # Energy system parameters
    sys_cfg = cfg.get("system", {})
    solar_cfg = sys_cfg.get("solar", {})
    wind_cfg = sys_cfg.get("wind", {})
    batt_cfg = sys_cfg.get("battery", {})
    cons_cfg = sys_cfg.get("constraints", {})

    if solar_cfg:
        main_mod.SOLAR_PANEL_CAPACITY = solar_cfg.get("capacity_kw", main_mod.SOLAR_PANEL_CAPACITY)
        main_mod.SOLAR_PANEL_COST = solar_cfg.get("cost", main_mod.SOLAR_PANEL_COST)
        main_mod.SOLAR_PANEL_LAND_USE = solar_cfg.get("land_m2", main_mod.SOLAR_PANEL_LAND_USE)
        main_mod.SOLAR_PANEL_LIFECYCLE_EMISSIONS = solar_cfg.get("lifecycle_emissions", main_mod.SOLAR_PANEL_LIFECYCLE_EMISSIONS)

    if wind_cfg:
        main_mod.WIND_TURBINE_CAPACITY = wind_cfg.get("capacity_kw", main_mod.WIND_TURBINE_CAPACITY)
        main_mod.WIND_TURBINE_COST = wind_cfg.get("cost", main_mod.WIND_TURBINE_COST)
        main_mod.WIND_TURBINE_LAND_USE = wind_cfg.get("land_m2", main_mod.WIND_TURBINE_LAND_USE)
        main_mod.WIND_TURBINE_LIFECYCLE_EMISSIONS = wind_cfg.get("lifecycle_emissions", main_mod.WIND_TURBINE_LIFECYCLE_EMISSIONS)

    if batt_cfg:
        main_mod.BATTERY_CAPACITY = batt_cfg.get("capacity_kwh", main_mod.BATTERY_CAPACITY)
        main_mod.BATTERY_COST = batt_cfg.get("cost", main_mod.BATTERY_COST)
        main_mod.BATTERY_CYCLE_LIFE = batt_cfg.get("cycle_life", main_mod.BATTERY_CYCLE_LIFE)
        main_mod.BATTERY_LIFECYCLE_EMISSIONS = batt_cfg.get("lifecycle_emissions", main_mod.BATTERY_LIFECYCLE_EMISSIONS)

    if cons_cfg:
        main_mod.MAX_LAND_AREA = cons_cfg.get("max_land_m2", main_mod.MAX_LAND_AREA)
        main_mod.MAX_BUDGET = cons_cfg.get("max_budget", main_mod.MAX_BUDGET)
        main_mod.MIN_ENERGY_REQUIREMENT = cons_cfg.get("min_energy_kwh_per_day", main_mod.MIN_ENERGY_REQUIREMENT)

    if "project_lifetime_years" in sys_cfg:
        main_mod.PROJECT_LIFETIME = sys_cfg.get("project_lifetime_years", main_mod.PROJECT_LIFETIME)

    # Bounds
    b = cfg.get("bounds", {})
    if b:
        main_mod.BOUNDS = [
            tuple(b.get("solar_panels", main_mod.BOUNDS[0])),
            tuple(b.get("wind_turbines", main_mod.BOUNDS[1])),
            tuple(b.get("batteries", main_mod.BOUNDS[2])),
        ]


