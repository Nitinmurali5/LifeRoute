"""Unit tests for LifeRoute Step 5 Simulated Scenario Benchmark."""

import json
from pathlib import Path
import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
PROCESSED_DIR = PROJECT_ROOT / "backend" / "data" / "processed"


class TestStep5Simulation:
    """Verifies that the simulated scenario is reproducible and adheres to constraints."""

    def test_simulation_summary_exists_and_valid(self):
        json_path = PROCESSED_DIR / "simulated_scenario_reproducible.json"
        assert json_path.exists(), "simulated_scenario_reproducible.json must exist"

        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        meta = data["scenario_metadata"]
        assert meta["is_simulation"] is True
        assert meta["source_node"] == "N_BENNIGANAHALLI"
        assert meta["destination_node"] == "N_KORAMANGALA_100FT"
        assert len(meta["injected_jam"]["edges"]) == 2
        assert meta["injected_jam"]["jam_speed_kmh"] == 4.0

    def test_three_routes_metrics_and_savings(self):
        json_path = PROCESSED_DIR / "simulated_scenario_reproducible.json"
        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        routes = data["routes"]
        r_dist = routes["shortest_distance"]
        r_react = routes["reactive_current_speed"]
        r_life = routes["liferoute_proactive"]

        # Traversal time: LifeRoute must be faster than both under the jammed scenario
        assert r_life["travel_time_minutes"] < r_dist["travel_time_minutes"], "LifeRoute must be faster than shortest distance"
        assert r_life["travel_time_minutes"] < r_react["travel_time_minutes"], "LifeRoute must be faster than reactive"

        # Distance: shortest distance must have lowest distance
        assert r_dist["total_distance_km"] < r_life["total_distance_km"], "Shortest distance route must have lower physical distance"

        # Time savings
        savings = data["time_savings"]
        assert savings["vs_shortest_distance"]["minutes_saved"] > 30.0, "Must save substantial time vs gridlocked route"
        assert savings["vs_reactive"]["minutes_saved"] > 0.0

    def test_scenario_artifacts_exist(self):
        png_path = PROCESSED_DIR / "simulated_scenario_routes.png"
        html_path = PROJECT_ROOT / "preview_simulated_scenario.html"
        assert png_path.exists(), "simulated_scenario_routes.png must exist"
        assert html_path.exists(), "preview_simulated_scenario.html must exist"
