"""Hospital extraction, nearest-node snapping, and simulated bed capacity.

Fetches real-world hospital locations via OSMnx amenity=hospital tag,
snaps each hospital facility to the nearest drivable network node,
and assigns deterministic simulated bed availability for research simulations.

NOTE: All bed availabilities are synthetic simulation parameters to evaluate
LifeRoute's time-dependent routing under capacity constraints.
They do NOT reflect real-world clinical status.
"""

from __future__ import annotations

import csv
import logging
import random
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import networkx as nx
import osmnx as ox

from backend.app.config import AppConfig

logger = logging.getLogger(__name__)

OVERPASS_ENDPOINTS: List[str] = [
    "https://overpass.kumi.systems/api/interpreter",
    "https://overpass-api.de/api/interpreter",
    "https://overpass.private.coffee/api/interpreter",
]


def snap_coordinate_to_node(G: nx.Graph, lat: float, lon: float) -> Any:
    """Snap geographic coordinate (lat, lon) to nearest node in road network.

    Args:
        G: NetworkX road graph.
        lat: Target latitude in degrees.
        lon: Target longitude in degrees.

    Returns:
        Node ID of closest network node.
    """
    try:
        nearest = ox.nearest_nodes(G, X=lon, Y=lat)
        return nearest
    except Exception as exc:
        logger.debug("ox.nearest_nodes failed (%s); using Euclidean distance fallback", exc)
        best_node = None
        min_dist_sq = float("inf")
        for node, data in G.nodes(data=True):
            node_y = data.get("y", data.get("lat"))
            node_x = data.get("x", data.get("lon"))
            if node_y is None or node_x is None:
                continue
            dist_sq = (node_y - lat) ** 2 + (node_x - lon) ** 2
            if dist_sq < min_dist_sq:
                min_dist_sq = dist_sq
                best_node = node
        if best_node is None:
            best_node = next(iter(G.nodes))
        return best_node


def fetch_osm_hospitals(
    city_name: str,
    radius_m: int,
    center_lat: Optional[float] = None,
    center_lon: Optional[float] = None,
) -> List[Dict[str, Any]]:
    """Fetch amenity=hospital features from OpenStreetMap across available mirrors.

    Args:
        city_name: Target city name for query.
        radius_m: Search radius in meters.
        center_lat: Optional center latitude.
        center_lon: Optional center longitude.

    Returns:
        List of dicts with 'name', 'lat', 'lon'.
    """
    logger.info("Querying OpenStreetMap for amenity=hospital in '%s'", city_name)

    if center_lat is None or center_lon is None:
        try:
            if hasattr(ox, "geocode"):
                center_lat, center_lon = ox.geocode(city_name)
            elif hasattr(ox, "geocoder") and hasattr(ox.geocoder, "geocode"):
                center_lat, center_lon = ox.geocoder.geocode(city_name)
        except Exception as exc:
            logger.warning("Geocoding failed for hospital search: %s", exc)
            if "coimbatore" in city_name.lower():
                center_lat, center_lon = 11.00181, 76.96284

    gdf = None
    tags = {"amenity": "hospital"}

    for ep in OVERPASS_ENDPOINTS:
        ox.settings.overpass_url = ep
        ox.settings.timeout = 120
        try:
            if center_lat is not None and center_lon is not None:
                if hasattr(ox, "features_from_point"):
                    gdf = ox.features_from_point((center_lat, center_lon), tags=tags, dist=radius_m)
                elif hasattr(ox, "geometries_from_point"):
                    gdf = ox.geometries_from_point((center_lat, center_lon), tags=tags, dist=radius_m)
            if gdf is None or gdf.empty:
                if hasattr(ox, "features_from_place"):
                    gdf = ox.features_from_place(city_name, tags=tags)
                elif hasattr(ox, "geometries_from_place"):
                    gdf = ox.geometries_from_place(city_name, tags=tags)

            if gdf is not None and not gdf.empty:
                logger.info("Retrieved hospitals from %s", ep)
                break
        except Exception as exc:
            logger.warning("Hospital query failed at %s (%s). Trying next mirror...", ep, exc)

    results: List[Dict[str, Any]] = []

    if gdf is not None and not gdf.empty:
        for idx, row in gdf.iterrows():
            geom = row.geometry
            if geom is None:
                continue

            if geom.geom_type == "Point":
                lat, lon = geom.y, geom.x
            else:
                centroid = geom.centroid
                lat, lon = centroid.y, centroid.x

            raw_name = row.get("name")
            name_str = str(raw_name) if raw_name is not None and str(raw_name) != "nan" else f"Hospital #{len(results) + 1}"

            results.append({"name": name_str, "lat": round(float(lat), 7), "lon": round(float(lon), 7)})

    logger.info("Retrieved %d hospital features from OpenStreetMap", len(results))
    return results


def extract_and_assign_hospitals(
    G: nx.Graph,
    config: Optional[AppConfig] = None,
    output_dir: Optional[Union[Path, str]] = None,
    seed: Optional[int] = None,
) -> Tuple[List[Dict[str, Any]], Path]:
    """Fetch hospitals, snap each to road network, assign simulated bed capacity, and save CSV.

    Args:
        G: Real city road network graph.
        config: Application configuration.
        output_dir: Output directory for hospitals.csv.
        seed: PRNG seed for deterministic bed availability assignment.

    Returns:
        Tuple of (list_of_hospitals, path_to_hospitals_csv).
    """
    if config is None:
        from backend.app.config import load_config
        config = load_config()

    actual_seed = seed if seed is not None else config.random_seed
    rng = random.Random(actual_seed)

    if output_dir is None:
        target_dir = Path.cwd() / "backend" / "data" / "processed"
    else:
        target_dir = Path(output_dir).resolve()
    target_dir.mkdir(parents=True, exist_ok=True)

    # 1. Fetch from OSM
    raw_hospitals = fetch_osm_hospitals(
        city_name=config.city_name,
        radius_m=config.radius_m,
    )

    # Fallback if fewer than 3 hospitals or OSM unreachable
    if len(raw_hospitals) < 3:
        logger.warning(
            "Fewer than 3 hospitals (%d) found in OSM for '%s'. Augmenting with simulated fallback hospitals.",
            len(raw_hospitals),
            config.city_name,
        )
        # Sample nodes from graph to create realistic spatial hospital distribution
        node_candidates = list(G.nodes)
        rng.shuffle(node_candidates)
        needed = 3 - len(raw_hospitals)
        fallback_names = [
            "Coimbatore Medical College Hospital",
            "G. Kuppuswamy Naidu Memorial Hospital",
            "PSG Hospital Emergency Wing",
            "Sri Ramakrishna Hospital",
            "KG Hospital & Post Graduate Medical Institute",
        ]
        for i in range(needed):
            fb_node = node_candidates[i]
            fb_data = G.nodes[fb_node]
            fb_lat = fb_data.get("y", fb_data.get("lat", 11.0168))
            fb_lon = fb_data.get("x", fb_data.get("lon", 76.9558))
            name = fallback_names[i % len(fallback_names)]
            raw_hospitals.append({"name": name, "lat": fb_lat, "lon": fb_lon})

    # 2. Snap to nearest graph nodes & assign deterministic simulated beds
    simulated_bed_patterns = [2, 0, 5, 1, 4, 0, 8, 3, 6, 0, 7]

    hospitals: List[Dict[str, Any]] = []
    seen_nodes = set()

    for idx, raw_h in enumerate(raw_hospitals):
        nearest_node = snap_coordinate_to_node(G, raw_h["lat"], raw_h["lon"])
        if nearest_node not in G:
            logger.warning("Snapped node %s not found in graph; skipping", nearest_node)
            continue

        hospital_id = f"HOSP_{idx + 1:03d}"
        beds = simulated_bed_patterns[idx % len(simulated_bed_patterns)]
        at_capacity = bool(beds == 0)

        hospital_entry = {
            "hospital_id": hospital_id,
            "name": f"{raw_h['name']} (Simulated)",
            "lat": raw_h["lat"],
            "lon": raw_h["lon"],
            "nearest_node": nearest_node,
            "beds_available": beds,
            "at_capacity": at_capacity,
        }
        hospitals.append(hospital_entry)
        seen_nodes.add(nearest_node)

    if len(hospitals) < 3:
        logger.warning(
            "Final hospital count (%d) is still under the recommended minimum of 3.",
            len(hospitals),
        )

    # 3. Save hospitals.csv
    csv_path = target_dir / "hospitals.csv"
    fieldnames = [
        "hospital_id",
        "name",
        "lat",
        "lon",
        "nearest_node",
        "beds_available",
        "at_capacity",
    ]
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(hospitals)

    logger.info("Saved %d hospitals to %s", len(hospitals), csv_path)
    return hospitals, csv_path
