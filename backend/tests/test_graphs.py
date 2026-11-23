"""Graph verification tests for LifeRoute (Phase 1).

Tests:
1. Both graphs (Synthetic Grid and Real City) are strongly connected.
2. Grid has roughly 500 nodes (e.g. 500) and 1,100 to 1,300 directed edges.
3. Every edge has edge_id, length_m > 0, free_flow_speed_kmh > 0, free_flow_time_s > 0.
4. Every hospital has a valid nearest_node present in its graph.
5. Same seed produces identical grid output (determinism).
"""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Any, Dict, List, Tuple

import networkx as nx
import pytest

from backend.app.config import AppConfig, load_config
from backend.app.graph.city_graph import build_city_graph
from backend.app.graph.grid_graph import build_synthetic_grid_graph
from backend.app.graph.hospitals import extract_and_assign_hospitals


@pytest.fixture(scope="session")
def app_config() -> AppConfig:
    """Load default application configuration for testing."""
    return load_config()


@pytest.fixture(scope="session")
def grid_data(app_config: AppConfig) -> Tuple[nx.DiGraph, int, List[Dict[str, Any]]]:
    """Generate synthetic grid graph once for test session."""
    return build_synthetic_grid_graph(config=app_config, seed=app_config.random_seed)


@pytest.fixture(scope="session")
def city_data(app_config: AppConfig) -> Tuple[nx.MultiDiGraph, List[Dict[str, Any]]]:
    """Load pre-built city graph or build if not present."""
    data_dir = Path(__file__).resolve().parent.parent / "data" / "processed"
    
    # Check Bengaluru graph first, then fallback to city.graphml
    bengaluru_graphml = data_dir / "bengaluru_network.graphml"
    bengaluru_hospitals = data_dir / "bengaluru_hospitals.csv"
    graphml_path = data_dir / "city.graphml"
    hospitals_csv_path = data_dir / "hospitals.csv"

    if bengaluru_graphml.exists() and bengaluru_hospitals.exists():
        G_city = nx.read_graphml(bengaluru_graphml)
        hospitals = []
        with open(bengaluru_hospitals, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                beds = int(row.get("beds_available", 0))
                hospitals.append(
                    {
                        "hospital_id": row.get("id", ""),
                        "name": row.get("name", ""),
                        "lat": float(row.get("lat", 0.0)),
                        "lon": float(row.get("lon", 0.0)),
                        "nearest_node": row.get("node_id", ""),
                        "beds_available": beds,
                        "at_capacity": beds == 0,
                    }
                )
        return G_city, hospitals
    elif graphml_path.exists() and hospitals_csv_path.exists():
        # Load existing processed graph
        G_city = nx.read_graphml(graphml_path)
        # Parse hospitals CSV
        hospitals = []
        with open(hospitals_csv_path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                hospitals.append(
                    {
                        "hospital_id": row["hospital_id"],
                        "name": row["name"],
                        "lat": float(row["lat"]),
                        "lon": float(row["lon"]),
                        "nearest_node": int(row["nearest_node"]) if row["nearest_node"].isdigit() else row["nearest_node"],
                        "beds_available": int(row["beds_available"]),
                        "at_capacity": row["at_capacity"].lower() in ("true", "1"),
                    }
                )
        return G_city, hospitals
    else:
        # Build live
        G_city, _, _ = build_city_graph(config=app_config, output_dir=data_dir)
        hospitals, _ = extract_and_assign_hospitals(G=G_city, config=app_config, output_dir=data_dir)
        return G_city, hospitals


# --- 1. Synthetic Grid Tests ---

def test_grid_graph_strong_connectivity(grid_data: Tuple[nx.DiGraph, int, List[Dict[str, Any]]]) -> None:
    """Verify that the synthetic grid graph is strongly connected."""
    G, _, _ = grid_data
    assert nx.is_strongly_connected(G), "Synthetic grid graph must be strongly connected"


def test_grid_graph_node_and_edge_counts(grid_data: Tuple[nx.DiGraph, int, List[Dict[str, Any]]]) -> None:
    """Verify grid has roughly 500 nodes and 1,100 to 1,300 directed edges."""
    G, _, _ = grid_data
    num_nodes = G.number_of_nodes()
    num_edges = G.number_of_edges()

    assert 480 <= num_nodes <= 520, f"Expected ~500 nodes, got {num_nodes}"
    assert 1100 <= num_edges <= 1300, f"Expected 1100-1300 directed edges, got {num_edges}"


def test_grid_graph_edge_attributes(grid_data: Tuple[nx.DiGraph, int, List[Dict[str, Any]]]) -> None:
    """Verify every edge has edge_id, length_m > 0, free_flow_speed_kmh > 0, free_flow_time_s > 0."""
    G, _, _ = grid_data
    for u, v, data in G.edges(data=True):
        assert "edge_id" in data and isinstance(data["edge_id"], str) and len(data["edge_id"]) > 0
        assert "length_m" in data and float(data["length_m"]) > 0
        assert "free_flow_speed_kmh" in data and float(data["free_flow_speed_kmh"]) > 0
        assert "free_flow_time_s" in data and float(data["free_flow_time_s"]) > 0
        assert "highway" in data and isinstance(data["highway"], str)


def test_grid_hospitals_validity(grid_data: Tuple[nx.DiGraph, int, List[Dict[str, Any]]]) -> None:
    """Verify incident node S and all hospitals have valid nodes and bed attributes."""
    G, incident_node, hospitals = grid_data
    assert incident_node in G, f"Incident node {incident_node} must be in graph"
    assert len(hospitals) >= 3, f"Expected at least 3 hospitals, got {len(hospitals)}"

    for h in hospitals:
        hn = h["nearest_node"]
        assert hn in G, f"Hospital nearest_node {hn} not found in graph"
        assert h["beds_available"] >= 0, f"Invalid bed count: {h['beds_available']}"
        assert h["at_capacity"] == (h["beds_available"] == 0)


def test_grid_graph_seed_determinism() -> None:
    """Verify that running with identical seed produces identical grid output."""
    cfg = AppConfig()
    G1, inc1, hosp1 = build_synthetic_grid_graph(config=cfg, seed=42)
    G2, inc2, hosp2 = build_synthetic_grid_graph(config=cfg, seed=42)

    assert set(G1.nodes()) == set(G2.nodes())
    assert set(G1.edges()) == set(G2.edges())
    assert inc1 == inc2
    assert hosp1 == hosp2

    for u, v in G1.edges():
        assert G1[u][v]["edge_id"] == G2[u][v]["edge_id"]
        assert G1[u][v]["length_m"] == G2[u][v]["length_m"]
        assert G1[u][v]["free_flow_speed_kmh"] == G2[u][v]["free_flow_speed_kmh"]
        assert G1[u][v]["free_flow_time_s"] == G2[u][v]["free_flow_time_s"]


# --- 2. Real City Graph Tests ---

def test_city_graph_strong_connectivity(city_data: Tuple[nx.MultiDiGraph, List[Dict[str, Any]]]) -> None:
    """Verify that the real city graph is strongly connected."""
    G_city, _ = city_data
    assert nx.is_strongly_connected(G_city), "City graph SCC component must be strongly connected"


def test_city_graph_edge_attributes(city_data: Tuple[nx.MultiDiGraph, List[Dict[str, Any]]]) -> None:
    """Verify every city edge has edge_id, length_m > 0, free_flow_speed_kmh > 0, free_flow_time_s > 0."""
    G_city, _ = city_data
    is_multi = G_city.is_multigraph()

    if is_multi:
        edge_iter = G_city.edges(keys=True, data=True)
    else:
        edge_iter = ((u, v, 0, d) for u, v, d in G_city.edges(data=True))

    for u, v, k, data in edge_iter:
        assert "edge_id" in data and isinstance(data["edge_id"], str)
        length_m = float(data.get("length_m", data.get("length", 0)))
        assert length_m > 0, f"Edge ({u}, {v}) has non-positive length: {length_m}"

        speed = float(data.get("free_flow_speed_kmh", 0))
        assert speed > 0, f"Edge ({u}, {v}) has non-positive speed: {speed}"

        time_s = float(data.get("free_flow_time_s", 0))
        assert time_s > 0, f"Edge ({u}, {v}) has non-positive free-flow time: {time_s}"


def test_city_hospitals_validity(city_data: Tuple[nx.MultiDiGraph, List[Dict[str, Any]]]) -> None:
    """Verify every city hospital has a valid nearest_node present in G_city."""
    G_city, hospitals = city_data
    assert len(hospitals) >= 3, f"Expected at least 3 hospitals, got {len(hospitals)}"

    # In GraphML node IDs are strings; handle both int and str lookups
    node_set = set(G_city.nodes())
    node_str_set = {str(n) for n in G_city.nodes()}

    for h in hospitals:
        hn = h["nearest_node"]
        assert (hn in node_set) or (str(hn) in node_str_set), (
            f"City hospital nearest_node '{hn}' not found in city graph"
        )
        assert h["beds_available"] >= 0
        assert h["at_capacity"] == (h["beds_available"] == 0)

