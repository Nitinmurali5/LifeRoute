"""Unit tests for LifeRoute V2I Green Wave traffic signal preemption."""

import math
from datetime import datetime
import pytest
import networkx as nx

from backend.app.routing.router import (
    DEFAULT_BENGALURU_SIGNALS,
    LifeRouteRouter,
    RoutingMode,
    SignalizedIntersection,
)


class TestV2IGreenWavePreemption:
    """Test suite for V2I DSRC / C-V2X traffic signal preemption."""

    def test_signalized_intersection_expected_delay_formula(self):
        """Verify Paper formula: d_j = (1 - gamma_j) * (C_j / 2) without preemption, 0 with preemption."""
        sig = SignalizedIntersection(
            id="SIG_TEST",
            name="Test Intersection",
            latitude=12.9716,
            longitude=77.5946,
            cycle_time_seconds=90.0,
            green_ratio=0.35,
        )

        # Unpreempted delay: (1 - 0.35) * (90 / 2) = 0.65 * 45 = 29.25 seconds
        assert math.isclose(sig.expected_delay_unpreempted(), 29.25, rel_tol=1e-3)
        # Preempted delay: 0 seconds
        assert sig.expected_delay_preempted() == 0.0
        # Time saved = 29.25 seconds
        assert math.isclose(sig.time_saved_seconds(), 29.25, rel_tol=1e-3)

    def test_default_bengaluru_signals_present_and_valid(self):
        """Verify default Bengaluru signal registry contains key arterial intersections."""
        assert len(DEFAULT_BENGALURU_SIGNALS) >= 8
        required_keys = [
            "SIG_INDIRANAGAR_100FT",
            "SIG_DOMLUR_FLYOVER",
            "SIG_EJIPURA",
            "SIG_TRINITY_CIRCLE",
            "SIG_ADUGODI",
        ]
        for k in required_keys:
            assert k in DEFAULT_BENGALURU_SIGNALS
            sig = DEFAULT_BENGALURU_SIGNALS[k]
            assert 12.90 <= sig.latitude <= 13.05
            assert 77.55 <= sig.longitude <= 77.70
            assert sig.cycle_time_seconds >= 60.0
            assert 0.20 <= sig.green_ratio <= 0.60
            assert sig.expected_delay_unpreempted() > 0.0

    def test_evaluate_v2i_along_path(self):
        """Test evaluation of V2I Green Wave along a path touching signal locations."""
        G = nx.DiGraph()
        G.add_node("A", lat=12.96171, lon=77.64174)  # Domlur Flyover signal location
        G.add_node("B", lat=12.93902, lon=77.63277)  # Ejipura signal location
        G.add_edge("A", "B", length_m=2800.0, free_flow_speed_kmh=40.0)

        router = LifeRouteRouter(G)
        coords = [(12.96171, 77.64174), (12.93902, 77.63277)]

        # Preempted mode
        eval_preempted = router.evaluate_v2i_green_wave(coords, v2i_enabled=True)
        assert eval_preempted["signals_count"] >= 2
        assert eval_preempted["total_signal_delay_seconds"] == 0.0
        assert eval_preempted["time_saved_seconds"] > 0.0

        # Unpreempted mode
        eval_unpreempted = router.evaluate_v2i_green_wave(coords, v2i_enabled=False)
        assert eval_unpreempted["signals_count"] >= 2
        assert eval_unpreempted["total_signal_delay_seconds"] > 0.0
        assert math.isclose(
            eval_unpreempted["total_signal_delay_seconds"],
            eval_preempted["time_saved_seconds"],
            rel_tol=1e-3,
        )

    def test_routing_with_and_without_v2i(self):
        """Test find_shortest_path reflects green wave savings when toggled."""
        G = nx.DiGraph()
        G.add_node("A", lat=12.96171, lon=77.64174)  # Domlur
        G.add_node("B", lat=12.93902, lon=77.63277)  # Ejipura
        G.add_edge("A", "B", length_m=2800.0, free_flow_speed_kmh=40.0)

        router = LifeRouteRouter(G)
        dep = datetime(2026, 11, 18, 18, 30)

        route_preempted = router.find_shortest_path(
            "A", "B", dep, mode=RoutingMode.STATIC_FREE_FLOW, v2i_preemption=True
        )
        route_unpreempted = router.find_shortest_path(
            "A", "B", dep, mode=RoutingMode.STATIC_FREE_FLOW, v2i_preemption=False
        )

        assert route_preempted is not None
        assert route_unpreempted is not None
        assert route_preempted.v2i_preemption_active is True
        assert route_unpreempted.v2i_preemption_active is False

        # Travel time with preemption must be strictly less than without preemption
        assert route_preempted.travel_time_seconds < route_unpreempted.travel_time_seconds
        assert route_unpreempted.signal_delay_seconds > 0.0
        assert route_preempted.signal_delay_seconds == 0.0
