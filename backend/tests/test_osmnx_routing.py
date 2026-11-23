"""Unit tests for LifeRoute Step 4 OSMnx Graph Construction and Dynamic Routing."""

import json
from datetime import datetime
from pathlib import Path
import networkx as nx
import osmnx as ox
import pytest

from backend.scripts.run_osmnx_routing import (
    load_or_build_graph,
    map_speeds_to_edges,
    run_shortest_path,
    ROAD_TYPE_DEFAULT_SPEEDS,
)
from backend.app.model.traffic_model import TrafficPredictor

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
PROCESSED_DIR = PROJECT_ROOT / "backend" / "data" / "processed"


@pytest.fixture(scope="module")
def osm_graph():
    """Load the Central Bengaluru OSM graph."""
    graphml_path = PROCESSED_DIR / "bengaluru_osm_graph.graphml"
    assert graphml_path.exists(), "bengaluru_osm_graph.graphml must exist"
    return ox.load_graphml(graphml_path)


@pytest.fixture(scope="module")
def predictor():
    """Load trained traffic prediction model if available."""
    model_dir = PROCESSED_DIR / "model"
    if (model_dir / "traffic_model.json").exists():
        return TrafficPredictor.load(model_dir)
    return None


class TestStep4OSMNxRouting:
    """Verifies OSMnx graph integrity, speed mapping, and routing consistency."""

    def test_osm_graph_size_and_attributes(self, osm_graph):
        assert len(osm_graph.nodes) >= 1000, "OSM graph must have at least 1,000 nodes"
        assert len(osm_graph.edges) >= 2000, "OSM graph must have at least 2,000 edges"

        # Check node attributes
        sample_node = next(iter(osm_graph.nodes(data=True)))[1]
        assert "x" in sample_node and "y" in sample_node, "Nodes must have lat/lon coordinates"

        # Check edge attributes
        sample_edge = next(iter(osm_graph.edges(data=True)))[2]
        assert "length" in sample_edge, "Edges must have length attribute"
        assert "highway" in sample_edge, "Edges must have highway attribute"

    def test_speed_mapping_and_share(self, osm_graph, predictor):
        departure = datetime(2026, 11, 18, 18, 30)
        G_mapped, audit = map_speeds_to_edges(osm_graph, predictor, departure)

        assert audit["total_edges"] == len(osm_graph.edges)
        assert audit["predicted_edges"] > 0, "Must map at least some edges to ML predicted speeds"
        assert audit["fallback_edges"] > 0, "Must fall back to road-type defaults for unmonitored roads"
        assert audit["predicted_share_pct"] + audit["fallback_share_pct"] == pytest.approx(100.0, 0.01)

        # Check that all edge travel times are positive
        for u, v, k, d in G_mapped.edges(keys=True, data=True):
            assert d["travel_time_seconds"] > 0.0, "Travel time must be strictly positive"
            assert d["speed_kmh"] >= 3.0, "Speed must be >= 3.0 km/h"

    def test_dijkstra_astar_identical_traversal(self, osm_graph, predictor):
        largest_scc = max(nx.strongly_connected_components(osm_graph), key=len)
        G_scc = osm_graph.subgraph(largest_scc).copy()
        departure = datetime(2026, 11, 18, 18, 30)
        G_weighted, _ = map_speeds_to_edges(G_scc, predictor, departure)

        nodes_sorted = sorted(list(G_weighted.nodes(data=True)), key=lambda n: n[1]["y"] + n[1]["x"])
        orig = nodes_sorted[0][0]
        dest = nodes_sorted[-1][0]

        res_dijkstra = run_shortest_path(G_weighted, orig, dest, weight_attr="travel_time_seconds", algorithm="dijkstra")
        res_astar = run_shortest_path(G_weighted, orig, dest, weight_attr="travel_time_seconds", algorithm="astar")

        assert abs(res_dijkstra["travel_time_minutes"] - res_astar["travel_time_minutes"]) < 1e-4, "Dijkstra and A* must return identical optimal travel times"
        assert res_dijkstra["path_nodes"] == res_astar["path_nodes"], "Dijkstra and A* must yield identical paths"
        assert res_astar["total_distance_km"] > 0.0
        assert res_astar["travel_time_minutes"] > 0.0
