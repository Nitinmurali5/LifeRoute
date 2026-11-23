"""Unit tests for LifeRoute traffic prediction model, Bengaluru graph, and routing engine."""

import math
from datetime import datetime
from pathlib import Path
import pytest
import networkx as nx
import pandas as pd

from backend.app.graph.bengaluru_network import build_bengaluru_network, BENGALURU_HOSPITALS
from backend.app.model.traffic_model import TrafficModelTrainer, TrafficPredictor
from backend.app.routing.router import LifeRouteRouter, RoutingMode

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent


@pytest.fixture(scope="module")
def model_and_predictor():
    """Load or train a lightweight test predictor."""
    model_dir = PROJECT_ROOT / "backend" / "data" / "processed" / "model"
    if (model_dir / "traffic_model.json").exists():
        return TrafficPredictor.load(model_dir)
    
    # If not already on disk, train small instance
    trainer = TrafficModelTrainer(n_estimators=10, max_depth=3)
    dummy_data = pd.DataFrame({
        "date": ["2026-11-01"] * 100,
        "time": [f"{i%24:02d}:00" for i in range(100)],
        "route_code": ["R1"] * 50 + ["R2"] * 50,
        "duration": [20] * 100,
        "distance": [10.0] * 100,
    })
    cleaned = trainer.clean_raw_data(dummy_data)
    trainer.train_and_evaluate(cleaned, test_ratio=0.2)
    trainer.save(model_dir)
    return TrafficPredictor.load(model_dir)


@pytest.fixture(scope="module")
def road_network(model_and_predictor):
    """Build Bengaluru network with model free-flow speeds."""
    G, hospitals = build_bengaluru_network(free_flow_speeds=model_and_predictor.free_flow_speeds)
    return G, hospitals


class TestTrafficModel:
    """Tests for model cleaning, feature engineering, and inference."""

    def test_clean_raw_data_filters_artifacts(self):
        raw = pd.DataFrame({
            "date": ["2026-10-01", "2026-10-01", "2026-10-01", "2026-10-01"],
            "time": ["10:00", "11:00", "12:00", "13:00"],
            "route_code": ["R1", "R1", "R1", "R1"],
            "duration": [2, 0, 30, 20],  # 2 min for 34km is 1020 km/h (outlier), 0 duration is invalid
            "distance": [34.0, 10.0, 10.0, 0.0],  # 0 distance is invalid
        })
        cleaned = TrafficModelTrainer.clean_raw_data(raw)
        assert len(cleaned) == 1
        assert cleaned.iloc[0]["duration"] == 30
        assert cleaned.iloc[0]["distance"] == 10.0
        assert math.isclose(cleaned.iloc[0]["speed_kmh"], 20.0)

    def test_feature_engineering_bounds(self):
        trainer = TrafficModelTrainer()
        df = pd.DataFrame({
            "date": ["2026-11-18"],
            "time": ["18:30"],
            "route_code": ["ROUTE_A"],
            "duration": [25],
            "distance": [10.0],
            "speed_kmh": [24.0],
            "timestamp": [pd.Timestamp("2026-11-18 18:30:00")],
        })
        df_feat, feature_cols = trainer.engineer_features(df, is_training=True)
        assert "sin_hour" in feature_cols
        assert "cos_hour" in feature_cols
        assert "route_ROUTE_A" in feature_cols
        assert -1.0 <= df_feat["sin_hour"].iloc[0] <= 1.0
        assert -1.0 <= df_feat["cos_hour"].iloc[0] <= 1.0

    def test_predictor_congestion_factor_ge_one(self, model_and_predictor):
        dt = datetime(2026, 11, 18, 18, 30)
        route = model_and_predictor.route_codes[0]
        speed = model_and_predictor.predict_speed(route, dt)
        c_factor = model_and_predictor.predict_congestion_factor(route, dt)

        assert speed > 0.0
        assert c_factor >= 1.0  # Paper Eq. 1: C_hat_e(t) >= 1.0


class TestBengaluruRoadNetwork:
    """Tests for road network structure and connectivity."""

    def test_network_strong_connectivity(self, road_network):
        G, _ = road_network
        assert nx.is_strongly_connected(G), "Bengaluru road network must be strongly connected"

    def test_edge_attributes_validity(self, road_network):
        G, _ = road_network
        for u, v, data in G.edges(data=True):
            assert data["length_m"] > 0, f"Edge ({u}, {v}) has non-positive length"
            assert data["free_flow_speed_kmh"] > 0, f"Edge ({u}, {v}) has non-positive speed"
            assert data["free_flow_time_s"] > 0, f"Edge ({u}, {v}) has non-positive traversal time"

    def test_hospitals_snapped_properly(self, road_network):
        G, hospitals = road_network
        assert len(hospitals) == len(BENGALURU_HOSPITALS)
        for h in hospitals:
            assert h["node_id"] in G.nodes, f"Hospital {h['name']} snapped to invalid node"
            assert h["snap_distance_m"] >= 0.0


class TestRouter:
    """Tests for time-dependent Dijkstra/A* routing and hospital selection."""

    def test_dijkstra_astar_consistency(self, road_network, model_and_predictor):
        G, _ = road_network
        router = LifeRouteRouter(G, predictor=model_and_predictor)
        orig = "N_BENNIGANAHALLI"
        dest = "N_TOWN_HALL"
        dep = datetime(2026, 11, 18, 18, 30)

        route_dijkstra = router.find_shortest_path(orig, dest, dep, mode=RoutingMode.DYNAMIC_PROACTIVE, use_astar=False)
        route_astar = router.find_shortest_path(orig, dest, dep, mode=RoutingMode.DYNAMIC_PROACTIVE, use_astar=True)

        assert math.isclose(route_dijkstra.travel_time_seconds, route_astar.travel_time_seconds, rel_tol=1e-4)
        assert route_dijkstra.path_nodes == route_astar.path_nodes

    def test_dynamic_traversal_time_ge_free_flow(self, road_network, model_and_predictor):
        G, _ = road_network
        router = LifeRouteRouter(G, predictor=model_and_predictor)
        orig = "N_BROOKFIELD"
        dest = "N_WIND_TUNNEL"
        dep = datetime(2026, 11, 18, 18, 30)

        route = router.find_shortest_path(orig, dest, dep, mode=RoutingMode.DYNAMIC_PROACTIVE)
        assert route.travel_time_seconds >= route.free_flow_time_seconds
        assert route.congestion_delay_seconds >= 0.0

    def test_capacity_aware_hospital_dispatch_skips_full_hospital(self, road_network, model_and_predictor):
        G, hospitals = road_network
        router = LifeRouteRouter(G, predictor=model_and_predictor)

        # Place incident right at Bowring Hospital node (which has 0 available beds)
        incident_node = "N_SHIVAJINAGAR"
        dep = datetime(2026, 11, 18, 18, 30)

        best_hosp, hosp_route = router.find_best_hospital(
            incident_node, hospitals, departure_time=dep, mode=RoutingMode.DYNAMIC_PROACTIVE, require_capacity=True
        )

        assert best_hosp is not None
        assert best_hosp["beds_available"] > 0, "Router must not dispatch to an at-capacity hospital"
        assert best_hosp["id"] != "HOSP_BOWRING", "Must skip Bowring Hospital because beds_available == 0"
