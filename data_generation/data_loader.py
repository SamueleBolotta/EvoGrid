"""
Data loading and processing for renewable energy systems.

Contains functions for loading environmental data from various sources
and converting between different data formats.
"""

import numpy as np
import pandas as pd
from .weather import generate_hourly_solar_irradiance, generate_hourly_wind_speed
from .demand import generate_hourly_energy_demand


def load_data_from_config(cfg: dict):
    """Load environmental data based on config dict.

    Supports synthetic generation (default) and simple CSV ingestion with
    columns: irradiance, wind_speed, demand (hourly). Aggregates to daily.
    """
    data_cfg = (cfg or {}).get("data", {})
    source = data_cfg.get("source", "synthetic")
    aggregate = data_cfg.get("aggregate_to_daily", True)

    if source == "csv":
        csv_path = data_cfg.get("csv_path")
        if not csv_path:
            raise ValueError("data.source is 'csv' but data.csv_path is not set in config")
        df = pd.read_csv(csv_path)
        irr = df["irradiance"].values
        wind = df["wind_speed"].values
        dem = df["demand"].values
        if not aggregate:
            raise ValueError("Hourly simulation not implemented; set aggregate_to_daily: true")
        daily_irr = np.array([irr[i:i+24].mean() for i in range(0, len(irr), 24)])
        daily_wind = np.array([wind[i:i+24].mean() for i in range(0, len(wind), 24)])
        daily_dem = np.array([dem[i:i+24].sum() for i in range(0, len(dem), 24)])
        return daily_irr, daily_wind, daily_dem

    # Default: synthetic
    days = int(data_cfg.get("days", 365))
    h_irr = generate_hourly_solar_irradiance(days)
    h_wind = generate_hourly_wind_speed(days)
    h_dem = generate_hourly_energy_demand(days)
    
    if aggregate:
        daily_irr = np.array([h_irr[i:i+24].mean() for i in range(0, len(h_irr), 24)])
        daily_wind = np.array([h_wind[i:i+24].mean() for i in range(0, len(h_wind), 24)])
        daily_dem = np.array([h_dem[i:i+24].sum() for i in range(0, len(h_dem), 24)])
        return daily_irr, daily_wind, daily_dem
    else:
        return h_irr, h_wind, h_dem
