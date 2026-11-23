"""Configuration loader and validator for LifeRoute using Pydantic.

Paper notation and simulation settings are loaded from a YAML configuration
file, validated, and exposed as strongly typed configuration models.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, Optional

import yaml
from pydantic import BaseModel, Field, field_validator

logger = logging.getLogger(__name__)


class GridConfig(BaseModel):
    """Configuration for synthetic grid road network (Table 4 scenario).

    Attributes:
        target_nodes: Number of nodes in the synthetic grid (e.g. 500).
        target_directed_edges: Target number of directed edges after pruning (e.g. 1200).
        cell_length_m: Nominal edge length d_e in meters (e.g. 250 m).
        free_flow_speed_kmh: Base free-flow speed v_{e,free} in km/h (e.g. 40 km/h).
    """

    target_nodes: int = Field(default=500, ge=4, description="Target number of nodes")
    target_directed_edges: int = Field(
        default=1200, ge=4, description="Target directed edge count"
    )
    cell_length_m: float = Field(
        default=250.0, gt=0.0, description="Nominal cell length d_e in meters"
    )
    free_flow_speed_kmh: float = Field(
        default=40.0, gt=0.0, description="Base free-flow speed v_{e,free} in km/h"
    )

    @field_validator("target_directed_edges")
    @classmethod
    def validate_edge_count(cls, v: int, info: Any) -> int:
        """Ensure target edge count is positive and sensible."""
        if v <= 0:
            raise ValueError("target_directed_edges must be strictly positive")
        return v


class HospitalsConfig(BaseModel):
    """Configuration for hospital assignment and simulated bed capacity.

    Attributes:
        count_grid: Number of hospital nodes to sample in the synthetic grid.
    """

    count_grid: int = Field(
        default=3, ge=1, description="Number of hospitals in synthetic grid"
    )


class AppConfig(BaseModel):
    """Global configuration for the LifeRoute research simulation.

    Follows paper notation:
    - Graph G = (V, E)
    - d_e: length in meters
    - v_{e,free}: free-flow speed in km/h
    - w_e(t) = (d_e / v_{e,free}) * C_hat_e(t)
    """

    city_name: str = Field(
        default="Coimbatore, Tamil Nadu, India",
        description="Name of city for geocoding and OSM download",
    )
    radius_m: int = Field(
        default=6000,
        gt=0,
        description="Download radius in meters around city centre",
    )
    network_type: str = Field(
        default="drive",
        description="OSM network type (e.g. 'drive')",
    )
    random_seed: int = Field(
        default=42,
        description="Deterministic PRNG seed for reproducibility",
    )
    grid: GridConfig = Field(default_factory=GridConfig)
    hospitals: HospitalsConfig = Field(default_factory=HospitalsConfig)


def find_default_config_path() -> Path:
    """Locate config.yaml across standard search directories.

    Returns:
        Path to config.yaml.
    """
    candidates = [
        Path.cwd() / "backend" / "config.yaml",
        Path.cwd() / "config.yaml",
        Path(__file__).resolve().parent.parent / "config.yaml",
        Path(__file__).resolve().parent.parent.parent / "backend" / "config.yaml",
    ]
    for candidate in candidates:
        if candidate.is_file():
            return candidate.resolve()
    return candidates[0]


def load_config(config_path: Optional[str | Path] = None) -> AppConfig:
    """Load and validate AppConfig from YAML file.

    Args:
        config_path: Optional path to config.yaml. If omitted, standard locations are searched.

    Returns:
        Validated AppConfig instance.

    Raises:
        FileNotFoundError: If configuration file cannot be found.
        ValueError: If YAML parsing or Pydantic validation fails.
    """
    if config_path is None:
        target_path = find_default_config_path()
    else:
        target_path = Path(config_path).resolve()

    if not target_path.exists():
        logger.error("Configuration file not found at %s", target_path)
        raise FileNotFoundError(f"Configuration file not found at: {target_path}")

    logger.info("Loading configuration from %s", target_path)
    try:
        with open(target_path, "r", encoding="utf-8") as f:
            raw_data = yaml.safe_load(f) or {}
    except yaml.YAMLError as exc:
        logger.error("YAML syntax error while reading %s: %s", target_path, exc)
        raise ValueError(f"Failed to parse YAML from {target_path}: {exc}") from exc

    try:
        config = AppConfig(**raw_data)
        logger.debug("Successfully validated configuration for city: %s", config.city_name)
        return config
    except Exception as exc:
        logger.error("Pydantic validation error for %s: %s", target_path, exc)
        raise

