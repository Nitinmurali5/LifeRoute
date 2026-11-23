"""Synthetic road grid network generator (Paper Table 4 Scenario).

Builds a reproducible, strongly connected synthetic road grid network
simulating the urban road topology for LifeRoute research.
Notation:
- Graph G = (V, E)
- d_e: length in meters (length_m)
- v_{e,free}: free-flow speed in km/h (free_flow_speed_kmh)
- w_e(t) = (d_e / v_{e,free}) * C_hat_e(t)
- free_flow_time_s: d_e / (v_{e,free} / 3.6)
"""

from __future__ import annotations

import json
import logging
import math
import random
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import networkx as nx

from backend.app.config import AppConfig

logger = logging.getLogger(__name__)

# Center of Coimbatore, Tamil Nadu, India for coordinate anchoring
DEFAULT_CENTER_LAT = 11.0168445
DEFAULT_CENTER_LON = 76.9558321
METERS_PER_DEGREE_LAT = 111320.0


def _meters_to_lat_lon(
    dx_m: float,
    dy_m: float,
    center_lat: float = DEFAULT_CENTER_LAT,
    center_lon: float = DEFAULT_CENTER_LON,
) -> Tuple[float, float]:
    """Convert meter offsets (dx, dy) from center into geographic (lat, lon).

    Args:
        dx_m: East-west offset in meters.
        dy_m: North-south offset in meters.
        center_lat: Origin latitude in degrees.
        center_lon: Origin longitude in degrees.

    Returns:
        Tuple of (latitude, longitude) in degrees.
    """
    lat = center_lat + (dy_m / METERS_PER_DEGREE_LAT)
    meters_per_degree_lon = METERS_PER_DEGREE_LAT * math.cos(math.radians(center_lat))
    lon = center_lon + (dx_m / meters_per_degree_lon)
    return round(lat, 7), round(lon, 7)


def build_synthetic_grid_graph(
    config: Optional[AppConfig] = None,
    rows: int = 25,
    cols: int = 20,
    seed: Optional[int] = None,
    center_lat: float = DEFAULT_CENTER_LAT,
    center_lon: float = DEFAULT_CENTER_LON,
) -> Tuple[nx.DiGraph, int, List[Dict[str, Any]]]:
    """Construct the paper's Table 4 synthetic directed road grid network.

    1. Builds a 2D grid of size rows x cols (~500 nodes).
    2. Initializes bidirectional directed edges.
    3. Prunes edges randomly (seeded) down to target_directed_edges (~1200)
       while strictly preserving strong connectivity.
    4. Populates edge attributes: edge_id, length_m, free_flow_speed_kmh,
       free_flow_time_s, highway.
    5. Populates node attributes: x, y, lat, lon.
    6. Designates fixed incident node S and simulated hospital nodes with capacity.

    Args:
        config: Application configuration instance.
        rows: Number of grid rows (default 25).
        cols: Number of grid columns (default 20).
        seed: Random PRNG seed. If omitted, uses config.random_seed or 42.
        center_lat: Reference center latitude for map projection.
        center_lon: Reference center longitude for map projection.

    Returns:
        Tuple of (graph, incident_node_id, hospitals_list).

    Raises:
        ValueError: If graph parameters violate connectivity or constraints.
    """
    if config is not None:
        target_nodes = config.grid.target_nodes
        target_edges = config.grid.target_directed_edges
        cell_length_m = config.grid.cell_length_m
        base_speed = config.grid.free_flow_speed_kmh
        num_hospitals = config.hospitals.count_grid
        actual_seed = seed if seed is not None else config.random_seed
    else:
        target_nodes = rows * cols
        target_edges = 1200
        cell_length_m = 250.0
        base_speed = 40.0
        num_hospitals = 3
        actual_seed = seed if seed is not None else 42

    rng = random.Random(actual_seed)
    logger.info(
        "Generating synthetic grid (rows=%d, cols=%d, target_nodes=%d, target_edges=%d, seed=%d)",
        rows,
        cols,
        rows * cols,
        target_edges,
        actual_seed,
    )

    G = nx.DiGraph()

    # Center origin in meters for aesthetic centering
    grid_width_m = (cols - 1) * cell_length_m
    grid_height_m = (rows - 1) * cell_length_m
    origin_x_m = -grid_width_m / 2.0
    origin_y_m = -grid_height_m / 2.0

    # 1. Create nodes
    for r in range(rows):
        for c in range(cols):
            node_id = r * cols + c
            x_m = origin_x_m + c * cell_length_m
            y_m = origin_y_m + r * cell_length_m
            lat, lon = _meters_to_lat_lon(x_m, y_m, center_lat, center_lon)

            G.add_node(
                node_id,
                x=round(x_m, 2),
                y=round(y_m, 2),
                lat=lat,
                lon=lon,
                row=r,
                col=c,
            )

    # 2. Add initial bidirectional directed edges with hierarchical road classes
    initial_edges: List[Tuple[int, int]] = []
    for r in range(rows):
        for c in range(cols):
            u = r * cols + c
            # Horizontal neighbor (c + 1)
            if c + 1 < cols:
                v = r * cols + (c + 1)
                initial_edges.append((u, v))
                initial_edges.append((v, u))
            # Vertical neighbor (r + 1)
            if r + 1 < rows:
                v = (r + 1) * cols + c
                initial_edges.append((u, v))
                initial_edges.append((v, u))

    for u, v in initial_edges:
        u_row = G.nodes[u]["row"]
        u_col = G.nodes[u]["col"]
        v_row = G.nodes[v]["row"]
        v_col = G.nodes[v]["col"]

        # Hierarchical road classes:
        # Avenues (every 5 blocks) -> primary (50 km/h)
        # Collectors (every 2 blocks) -> secondary (40 km/h)
        # Local streets -> residential (30 km/h)
        if u_row % 5 == 0 and v_row % 5 == 0 or u_col % 5 == 0 and v_col % 5 == 0:
            highway = "primary"
            speed_kmh = 50.0
        elif u_row % 2 == 0 and v_row % 2 == 0 or u_col % 2 == 0 and v_col % 2 == 0:
            highway = "secondary"
            speed_kmh = 40.0
        else:
            highway = "residential"
            speed_kmh = 30.0

        # Small seeded variation in length (+- 4%)
        length_jitter = rng.uniform(0.96, 1.04)
        length_m = round(cell_length_m * length_jitter, 2)
        # Free-flow time in seconds: d_e / (v_{e,free} / 3.6)
        free_flow_time_s = round(length_m / (speed_kmh / 3.6), 3)

        G.add_edge(
            u,
            v,
            edge_id=f"{u}-{v}",
            length_m=length_m,
            free_flow_speed_kmh=speed_kmh,
            free_flow_time_s=free_flow_time_s,
            highway=highway,
        )

    # 3. Prune edges randomly down to target_edges while strictly maintaining strong connectivity
    candidate_edges = list(G.edges())
    rng.shuffle(candidate_edges)

    edges_removed = 0
    for u, v in candidate_edges:
        if G.number_of_edges() <= target_edges:
            break

        edge_data = G.get_edge_data(u, v)
        G.remove_edge(u, v)

        # In a strongly connected graph, removing (u, v) preserves strong connectivity
        # if and only if there remains an alternate directed path from u to v.
        if nx.has_path(G, u, v):
            edges_removed += 1
        else:
            # Revert edge
            G.add_edge(u, v, **edge_data)

    logger.info(
        "Pruning completed: removed %d edges. Final edges: %d (target: %d)",
        edges_removed,
        G.number_of_edges(),
        target_edges,
    )

    if not nx.is_strongly_connected(G):
        raise ValueError("Generated grid graph is not strongly connected!")

    # 4. Pick fixed incident node S and simulated hospital nodes
    # Incident node S picked near the lower quadrant
    incident_node = int(cols * (rows // 4) + (cols // 4))

    # Candidate nodes for hospitals: pick distant nodes to simulate diverse travel distances
    candidate_hospital_nodes = [
        int(cols * (3 * rows // 4) + (3 * cols // 4)),  # Top-right
        int(cols * (3 * rows // 4) + (cols // 4)),       # Top-left
        int(cols * (rows // 2) + (cols // 2)),           # Center
        int(cols * (rows // 4) + (3 * cols // 4)),       # Bottom-right
    ]

    hospital_bed_scenarios = [2, 0, 5, 3]  # Deterministic simulated bed counts
    hospital_names = [
        "City Central Hospital",
        "Memorial Emergency Center",
        "Apex Multi-Specialty Clinic",
        "Northgate Trauma Hospital",
    ]

    hospitals: List[Dict[str, Any]] = []
    selected_nodes: List[int] = []

    for i in range(min(num_hospitals, len(candidate_hospital_nodes))):
        h_node = candidate_hospital_nodes[i]
        beds = hospital_bed_scenarios[i % len(hospital_bed_scenarios)]
        selected_nodes.append(h_node)

        h_data = {
            "hospital_id": f"HOSP_{i+1:02d}",
            "name": f"{hospital_names[i % len(hospital_names)]} (Simulated)",
            "nearest_node": h_node,
            "node_id": h_node,
            "lat": G.nodes[h_node]["lat"],
            "lon": G.nodes[h_node]["lon"],
            "beds_available": beds,
            "at_capacity": bool(beds == 0),
        }
        hospitals.append(h_data)

    # Mark node attributes on graph
    for n in G.nodes():
        G.nodes[n]["is_incident"] = (n == incident_node)
        G.nodes[n]["is_hospital"] = (n in selected_nodes)

    logger.info(
        "Incident node: %d, Hospitals: %s",
        incident_node,
        [h["hospital_id"] for h in hospitals],
    )
    return G, incident_node, hospitals


def save_grid_scenario_json(
    G: nx.DiGraph,
    incident_node: int,
    hospitals: List[Dict[str, Any]],
    output_path: Path | str,
    metadata: Optional[Dict[str, Any]] = None,
) -> Path:
    """Serialize the synthetic grid scenario to JSON.

    Args:
        G: Directed grid road network.
        incident_node: Designated incident node ID.
        hospitals: List of hospital metadata dicts.
        output_path: Target JSON file path.
        metadata: Optional extra metadata dictionary.

    Returns:
        Path to saved JSON file.
    """
    path = Path(output_path).resolve()
    path.parent.mkdir(parents=True, exist_ok=True)

    nodes_data = []
    for node_id, data in G.nodes(data=True):
        nodes_data.append(
            {
                "node_id": node_id,
                "x": data["x"],
                "y": data["y"],
                "lat": data["lat"],
                "lon": data["lon"],
                "is_incident": data.get("is_incident", False),
                "is_hospital": data.get("is_hospital", False),
            }
        )

    edges_data = []
    for u, v, data in G.edges(data=True):
        edges_data.append(
            {
                "edge_id": data["edge_id"],
                "u": u,
                "v": v,
                "length_m": data["length_m"],
                "free_flow_speed_kmh": data["free_flow_speed_kmh"],
                "free_flow_time_s": data["free_flow_time_s"],
                "highway": data["highway"],
            }
        )

    payload = {
        "metadata": {
            "scenario": "Paper Table 4 Synthetic Road Grid",
            "paper_reference": "LifeRoute Research Simulation (IEEE Proposed System)",
            "notation": {
                "d_e": "edge length in meters (length_m)",
                "v_e_free": "free-flow speed in km/h (free_flow_speed_kmh)",
                "weight_formula": "w_e(t) = (d_e / v_e_free) * C_hat_e(t)",
                "free_flow_time_s": "d_e / (v_e_free / 3.6)",
            },
            "num_nodes": G.number_of_nodes(),
            "num_edges": G.number_of_edges(),
            **(metadata or {}),
        },
        "incident_node": incident_node,
        "hospitals": hospitals,
        "nodes": nodes_data,
        "edges": edges_data,
    }

    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)

    logger.info("Saved grid scenario JSON to %s", path)
    return path

