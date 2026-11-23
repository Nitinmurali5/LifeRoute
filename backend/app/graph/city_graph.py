"""OSM-based real city road network generator (Coimbatore demo graph).

Downloads the drivable road network via OSMnx, extracts the largest strongly
connected component (SCC), parses speed limits and road classes, computes
free-flow travel times, and exports GraphML and CSV datasets.

Paper notation:
- Graph G = (V, E)
- d_e: length in meters (length_m)
- v_{e,free}: free-flow speed in km/h (free_flow_speed_kmh)
- w_e(t) = (d_e / v_{e,free}) * C_hat_e(t)
- free_flow_time_s = d_e / (v_{e,free} / 3.6)
"""

from __future__ import annotations

import csv
import logging
import re
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import networkx as nx
import osmnx as ox

from backend.app.config import AppConfig

logger = logging.getLogger(__name__)

# Public Overpass API mirrors for high availability
OVERPASS_ENDPOINTS: List[str] = [
    "https://overpass.kumi.systems/api/interpreter",
    "https://overpass-api.de/api/interpreter",
    "https://overpass.private.coffee/api/interpreter",
]

# Sensible speed defaults by highway class (km/h)
HIGHWAY_SPEED_DEFAULTS: Dict[str, float] = {
    "motorway": 80.0,
    "motorway_link": 60.0,
    "trunk": 70.0,
    "trunk_link": 50.0,
    "primary": 50.0,
    "primary_link": 40.0,
    "secondary": 40.0,
    "secondary_link": 35.0,
    "tertiary": 35.0,
    "tertiary_link": 30.0,
    "residential": 30.0,
    "living_street": 20.0,
    "unclassified": 30.0,
    "service": 20.0,
}
FALLBACK_SPEED_KMH = 30.0


def parse_highway_class(highway_val: Any) -> str:
    """Normalize OSM highway tag to a canonical string.

    Args:
        highway_val: Raw OSM highway attribute (string, list, or None).

    Returns:
        Canonical highway string.
    """
    if isinstance(highway_val, list):
        return str(highway_val[0]) if highway_val else "unclassified"
    if isinstance(highway_val, str):
        return highway_val
    return "unclassified"


def parse_speed_kmh(maxspeed_val: Any, highway_class: str) -> float:
    """Parse OSM maxspeed value or assign a sensible highway class default.

    Args:
        maxspeed_val: Raw OSM maxspeed tag (string, int, list, or None).
        highway_class: Normalized highway classification string.

    Returns:
        Speed limit in km/h as a positive float.
    """
    if maxspeed_val is not None:
        if isinstance(maxspeed_val, list):
            maxspeed_val = maxspeed_val[0] if maxspeed_val else None

        if isinstance(maxspeed_val, (int, float)):
            if maxspeed_val > 0:
                return float(maxspeed_val)

        if isinstance(maxspeed_val, str):
            clean_str = maxspeed_val.strip().lower()
            # Match numeric part
            match = re.search(r"(\d+(\.\d+)?)", clean_str)
            if match:
                speed_num = float(match.group(1))
                if "mph" in clean_str:
                    return round(speed_num * 1.60934, 1)
                if speed_num > 0:
                    return float(speed_num)

    # Fallback to highway class default
    return HIGHWAY_SPEED_DEFAULTS.get(highway_class, FALLBACK_SPEED_KMH)


def geocode_city_center(city_name: str) -> Tuple[float, float]:
    """Geocode city center coordinates using OSMnx with robust error handling.

    Args:
        city_name: Name of the city to geocode.

    Returns:
        Tuple of (latitude, longitude).

    Raises:
        RuntimeError: If geocoding fails due to network or lookup errors.
    """
    try:
        if hasattr(ox, "geocode"):
            point = ox.geocode(city_name)
        elif hasattr(ox, "geocoder") and hasattr(ox.geocoder, "geocode"):
            point = ox.geocoder.geocode(city_name)
        else:
            raise AttributeError("OSMnx does not expose a geocode method")
        return float(point[0]), float(point[1])
    except Exception as exc:
        logger.warning(
            "Geocoding failed for '%s' (%s). Checking for known coordinates.",
            city_name,
            exc,
        )
        if "coimbatore" in city_name.lower():
            return 11.00181, 76.96284
        raise RuntimeError(
            f"Failed to geocode city '{city_name}'. Please verify internet connectivity: {exc}"
        ) from exc


def download_or_load_raw_city_graph(
    city_name: str,
    radius_m: int,
    network_type: str = "drive",
    cache_dir: Optional[Path] = None,
    processed_dir: Optional[Path] = None,
) -> nx.MultiDiGraph:
    """Download drivable road network via OSMnx with local caching and mirror failover.

    Args:
        city_name: Name of the city (e.g. 'Coimbatore, Tamil Nadu, India').
        radius_m: Radius in meters around the city center.
        network_type: OSMnx network type ('drive').
        cache_dir: Path to cache directory.
        processed_dir: Optional path to processed directory for offline load.

    Returns:
        NetworkX MultiDiGraph representing the downloaded road network.

    Raises:
        RuntimeError: If download fails across all mirrors and no cache exists.
    """
    if cache_dir is None:
        cache_dir = Path.cwd() / "backend" / "data" / "cache"
    cache_dir.mkdir(parents=True, exist_ok=True)

    # Check if a previously processed city.graphml exists to allow offline re-runs
    if processed_dir is not None:
        cached_graphml = processed_dir / "city.graphml"
        if cached_graphml.exists():
            logger.info("Found existing graphml at %s; loading from disk", cached_graphml)
            try:
                return ox.load_graphml(cached_graphml)
            except Exception as load_err:
                logger.warning("Could not load %s (%s); proceeding to download", cached_graphml, load_err)

    # Configure OSMnx caching
    ox.settings.use_cache = True
    ox.settings.cache_folder = str(cache_dir)
    ox.settings.log_console = False

    center_lat, center_lon = geocode_city_center(city_name)
    logger.info("Geocoded '%s' to center: (lat: %.5f, lon: %.5f)", city_name, center_lat, center_lon)

    last_error: Optional[Exception] = None

    for ep in OVERPASS_ENDPOINTS:
        logger.info("Attempting OSM download via Overpass endpoint: %s", ep)
        ox.settings.overpass_url = ep
        ox.settings.timeout = 180

        try:
            G_raw = ox.graph_from_point(
                (center_lat, center_lon),
                dist=radius_m,
                network_type=network_type,
                simplify=True,
            )
            logger.info(
                "Raw OSM graph downloaded successfully from %s: %d nodes, %d edges",
                ep,
                G_raw.number_of_nodes(),
                G_raw.number_of_edges(),
            )
            return G_raw

        except Exception as exc:
            logger.warning(
                "Overpass mirror %s encountered error: %s. Retrying next mirror...",
                ep,
                exc,
            )
            last_error = exc
            time.sleep(2)

    error_msg = (
        f"Failed to fetch road network for '{city_name}' within radius {radius_m}m across "
        f"all configured Overpass API mirrors ({OVERPASS_ENDPOINTS}). "
        f"Ensure network connectivity to OpenStreetMap Overpass API is available. "
        f"Last error details: {last_error}"
    )
    logger.error(error_msg)
    raise RuntimeError(error_msg)


def build_city_graph(
    config: Optional[AppConfig] = None,
    output_dir: Optional[Union[Path, str]] = None,
) -> Tuple[nx.MultiDiGraph, Path, Path]:
    """Download, process, and persist the real city road network.

    Steps:
    1. Downloads drivable network around city_name within radius_m.
    2. Extracts the largest strongly connected component (SCC).
    3. Annotates edges with:
       - edge_id: stable string 'u-v-key'
       - length_m: length in meters (d_e > 0)
       - free_flow_speed_kmh: speed limit or class default (v_{e,free} > 0)
       - free_flow_time_s: d_e / (v_{e,free} / 3.6) > 0
       - highway: canonical highway string
    4. Saves city.graphml and edges.csv under output_dir (default: backend/data/processed/).

    Args:
        config: Application configuration instance.
        output_dir: Directory where processed files will be written.

    Returns:
        Tuple of (strongly_connected_graph, path_to_graphml, path_to_edges_csv).

    Raises:
        RuntimeError: If download fails or graph cannot be processed.
    """
    if config is None:
        from backend.app.config import load_config
        config = load_config()

    if output_dir is None:
        target_dir = Path.cwd() / "backend" / "data" / "processed"
    else:
        target_dir = Path(output_dir).resolve()
    target_dir.mkdir(parents=True, exist_ok=True)

    cache_dir = target_dir.parent / "cache"

    # 1. Download raw graph
    G_raw = download_or_load_raw_city_graph(
        city_name=config.city_name,
        radius_m=config.radius_m,
        network_type=config.network_type,
        cache_dir=cache_dir,
        processed_dir=target_dir,
    )

    # 2. Extract largest strongly connected component (SCC)
    logger.info("Extracting largest strongly connected component (SCC)...")
    scc_nodes = max(nx.strongly_connected_components(G_raw), key=len)
    G_scc: nx.MultiDiGraph = G_raw.subgraph(scc_nodes).copy()

    logger.info(
        "Largest SCC extracted: %d nodes (was %d), %d edges (was %d)",
        G_scc.number_of_nodes(),
        G_raw.number_of_nodes(),
        G_scc.number_of_edges(),
        G_raw.number_of_edges(),
    )

    if not nx.is_strongly_connected(G_scc):
        raise ValueError("Subgraph is unexpectedly not strongly connected")

    # 3. Add required attributes and sanitize for GraphML
    edges_records = []
    for u, v, key, data in G_scc.edges(keys=True, data=True):
        edge_id = f"{u}-{v}-{key}"
        length_m = float(data.get("length", 10.0))
        if length_m <= 0:
            length_m = 10.0

        highway_class = parse_highway_class(data.get("highway"))
        raw_maxspeed = data.get("maxspeed", None)
        speed_kmh = parse_speed_kmh(raw_maxspeed, highway_class)
        free_flow_time_s = round(length_m / (speed_kmh / 3.6), 3)

        # Update in-memory graph data
        data["edge_id"] = edge_id
        data["length_m"] = length_m
        data["free_flow_speed_kmh"] = speed_kmh
        data["free_flow_time_s"] = free_flow_time_s
        data["highway"] = highway_class

        # GraphML compatibility: sanitize lists/dicts to strings
        for attr_key, attr_val in list(data.items()):
            if isinstance(attr_val, (list, tuple, dict)):
                data[attr_key] = str(attr_val)

        edges_records.append(
            {
                "edge_id": edge_id,
                "u": u,
                "v": v,
                "key": key,
                "length_m": length_m,
                "free_flow_speed_kmh": speed_kmh,
                "free_flow_time_s": free_flow_time_s,
                "highway": highway_class,
            }
        )

    # Sanitize node attributes for GraphML
    for node, data in G_scc.nodes(data=True):
        for attr_key, attr_val in list(data.items()):
            if isinstance(attr_val, (list, tuple, dict)):
                data[attr_key] = str(attr_val)

    # 4. Save city.graphml and edges.csv
    graphml_path = target_dir / "city.graphml"
    edges_csv_path = target_dir / "edges.csv"

    logger.info("Writing GraphML to %s", graphml_path)
    try:
        ox.save_graphml(G_scc, graphml_path)
    except Exception as exc:
        logger.warning("ox.save_graphml failed (%s); falling back to nx.write_graphml", exc)
        nx.write_graphml(G_scc, graphml_path)

    logger.info("Writing edges CSV to %s", edges_csv_path)
    fieldnames = [
        "edge_id",
        "u",
        "v",
        "key",
        "length_m",
        "free_flow_speed_kmh",
        "free_flow_time_s",
        "highway",
    ]
    with open(edges_csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(edges_records)

    logger.info(
        "Successfully built city graph: %d nodes, %d edges saved to %s and %s",
        G_scc.number_of_nodes(),
        G_scc.number_of_edges(),
        graphml_path,
        edges_csv_path,
    )

    return G_scc, graphml_path, edges_csv_path
