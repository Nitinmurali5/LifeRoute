"""LifeRoute Flask Backend Application Package.

Phase 1 foundation: provides application factory create_app(),
CORS support, and baseline health check endpoint.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, Optional

from flask import Flask, jsonify
from flask_cors import CORS

from backend.app.config import AppConfig, load_config

logger = logging.getLogger(__name__)


def create_app(config: Optional[AppConfig] = None) -> Flask:
    """Create and configure the LifeRoute Flask application instance.

    Args:
        config: Optional AppConfig instance. If omitted, loaded from config.yaml.

    Returns:
        Configured Flask application.
    """
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    app = Flask(__name__)
    CORS(app)

    # Load application configuration
    if config is None:
        try:
            app_cfg = load_config()
        except Exception as exc:
            logger.warning("Could not load config.yaml (%s); using default AppConfig", exc)
            app_cfg = AppConfig()
    else:
        app_cfg = config

    app.config["LIFEROUTE_CONFIG"] = app_cfg

    # Initialize graph and predictor lazily
    _router_cache = {}

    def get_router():
        if "router" not in _router_cache:
            from backend.app.graph.bengaluru_network import build_bengaluru_network
            from backend.app.model.traffic_model import TrafficPredictor
            from backend.app.routing.router import LifeRouteRouter
            from pathlib import Path

            model_dir = Path.cwd() / "backend" / "data" / "processed" / "model"
            predictor = TrafficPredictor.load(model_dir) if (model_dir / "traffic_model.json").exists() else None
            free_flow = predictor.free_flow_speeds if predictor else None
            G, hospitals = build_bengaluru_network(free_flow_speeds=free_flow)
            router = LifeRouteRouter(G, predictor=predictor)
            _router_cache["router"] = router
            _router_cache["hospitals"] = hospitals
        return _router_cache["router"], _router_cache["hospitals"]

    @app.route("/health", methods=["GET"])
    def health_check() -> Any:
        """Health check endpoint."""
        return jsonify({"status": "ok"}), 200

    @app.route("/api/hospitals", methods=["GET"])
    def get_hospitals() -> Any:
        """Return list of monitored emergency hospitals with live capacity."""
        try:
            _, hospitals = get_router()
            return jsonify({"status": "success", "count": len(hospitals), "hospitals": hospitals}), 200
        except Exception as exc:
            return jsonify({"status": "error", "message": str(exc)}), 500

    @app.route("/api/route", methods=["POST", "GET"])
    def compute_route() -> Any:
        """Calculate static or dynamic proactive shortest path."""
        from flask import request
        from datetime import datetime
        from backend.app.routing.router import RoutingMode

        data = request.get_json(silent=True) or request.args.to_dict()
        origin = data.get("origin", "N_BENNIGANAHALLI")
        destination = data.get("destination", "N_SILK_BOARD")
        dep_str = data.get("departure_time")
        mode_str = data.get("mode", "dynamic_proactive").lower()
        v2i_param = data.get("v2i_preemption", True)
        v2i_preemption = False if str(v2i_param).lower() in ["false", "0", "off"] else True

        try:
            departure_time = datetime.fromisoformat(dep_str) if dep_str else datetime(2026, 11, 18, 18, 30)
            mode = RoutingMode.DYNAMIC_PROACTIVE if "dynamic" in mode_str else RoutingMode.STATIC_FREE_FLOW
            router, _ = get_router()
            result = router.find_shortest_path(
                origin,
                destination,
                departure_time=departure_time,
                mode=mode,
                v2i_preemption=v2i_preemption,
            )
            if not result:
                return jsonify({"status": "error", "message": "No path found"}), 404

            coords = [[router.graph.nodes[n]["lat"], router.graph.nodes[n]["lon"]] for n in result.path_nodes if n in router.graph]
            return jsonify({
                "status": "success",
                "mode": result.mode.value,
                "origin": origin,
                "destination": destination,
                "departure_time": result.departure_time.isoformat(),
                "arrival_time": result.arrival_time.isoformat(),
                "travel_time_minutes": result.travel_time_minutes,
                "travel_time_seconds": result.travel_time_seconds,
                "distance_km": result.distance_km,
                "average_speed_kmh": result.average_speed_kmh,
                "path_nodes": result.path_nodes,
                "coordinates": coords,
                "corridors_traversed": result.corridors_traversed,
                "v2i": {
                    "preemption_active": result.v2i_preemption_active,
                    "intersections_count": result.intersections_count,
                    "signal_delay_seconds": result.signal_delay_seconds,
                    "time_saved_seconds": result.v2i_time_saved_seconds,
                    "intersections": result.intersections_detail,
                },
            }), 200
        except Exception as exc:
            return jsonify({"status": "error", "message": str(exc)}), 400

    @app.route("/api/v2i/signals", methods=["GET"])
    def get_v2i_signals() -> Any:
        """Return all monitored signalized intersections with V2I preemption parameters."""
        try:
            router, _ = get_router()
            signals = [s.to_dict() for s in router.signals.values()]
            return jsonify({"status": "success", "count": len(signals), "signals": signals}), 200
        except Exception as exc:
            return jsonify({"status": "error", "message": str(exc)}), 500

    @app.route("/api/v2i/evaluate", methods=["POST", "GET"])
    def evaluate_v2i() -> Any:
        """Evaluate V2I preemption along a given list of coordinates."""
        from flask import request
        data = request.get_json(silent=True) or request.args.to_dict()
        coords = data.get("coordinates", [])
        v2i_param = data.get("v2i_enabled", True)
        v2i_enabled = False if str(v2i_param).lower() in ["false", "0", "off"] else True

        try:
            router, _ = get_router()
            res = router.evaluate_v2i_green_wave(coords, v2i_enabled=v2i_enabled)
            return jsonify({"status": "success", "data": res}), 200
        except Exception as exc:
            return jsonify({"status": "error", "message": str(exc)}), 400

    @app.route("/api/dispatch", methods=["POST", "GET"])
    def capacity_aware_dispatch() -> Any:
        """Find best hospital and optimal route considering bed availability."""
        from flask import request
        from datetime import datetime
        from backend.app.routing.router import RoutingMode

        data = request.get_json(silent=True) or request.args.to_dict()
        incident = data.get("incident_node", "N_BENNIGANAHALLI")
        dep_str = data.get("departure_time")

        try:
            departure_time = datetime.fromisoformat(dep_str) if dep_str else datetime(2026, 11, 18, 18, 30)
            router, hospitals = get_router()
            best = router.find_best_hospital(incident, hospitals, departure_time=departure_time, mode=RoutingMode.DYNAMIC_PROACTIVE, require_capacity=True)
            if not best:
                return jsonify({"status": "error", "message": "No capable hospital available"}), 404

            best_hosp, result = best
            coords = [[router.graph.nodes[n]["lat"], router.graph.nodes[n]["lon"]] for n in result.path_nodes if n in router.graph]
            return jsonify({
                "status": "success",
                "incident_node": incident,
                "selected_hospital": best_hosp,
                "route": {
                    "travel_time_minutes": result.travel_time_minutes,
                    "distance_km": result.distance_km,
                    "path_nodes": result.path_nodes,
                    "coordinates": coords,
                    "corridors_traversed": result.corridors_traversed,
                }
            }), 200
        except Exception as exc:
            return jsonify({"status": "error", "message": str(exc)}), 400

    @app.route("/", methods=["GET"])
    def index() -> Any:
        """Serve the visual dashboard UI."""
        from flask import send_from_directory
        from pathlib import Path
        frontend_dir = Path(__file__).resolve().parent.parent.parent / "frontend"
        if (frontend_dir / "index.html").exists():
            return send_from_directory(frontend_dir, "index.html")
        return jsonify({"message": "LifeRoute API Server is running. Frontend index.html not found in frontend/."}), 200

    @app.route("/api/nodes", methods=["GET"])
    def get_nodes() -> Any:
        """Return all road network nodes with coordinates and labels."""
        try:
            router, _ = get_router()
            nodes_data = []
            for node_id, data in router.graph.nodes(data=True):
                nodes_data.append({
                    "id": node_id,
                    "label": data.get("label", str(node_id)),
                    "lat": data.get("lat", 0.0),
                    "lon": data.get("lon", 0.0),
                })
            nodes_data.sort(key=lambda x: x["label"])
            return jsonify({"status": "success", "count": len(nodes_data), "nodes": nodes_data}), 200
        except Exception as exc:
            return jsonify({"status": "error", "message": str(exc)}), 500

    @app.route("/api/simulation", methods=["GET"])
    def get_simulation() -> Any:
        """Return the precomputed 3-way simulation scenario with full node coordinates."""
        import json
        from pathlib import Path
        sim_file = Path.cwd() / "backend" / "data" / "processed" / "simulated_scenario_reproducible.json"
        if sim_file.exists():
            with open(sim_file, "r") as f:
                data = json.load(f)
            # Enrich paths with lat/lon coordinates
            router, _ = get_router()
            G = router.graph
            for rk in ["shortest_distance", "reactive_current_speed", "liferoute_proactive"]:
                if rk in data.get("routes", {}):
                    path = data["routes"][rk].get("path_nodes", [])
                    coords = [[G.nodes[n]["lat"], G.nodes[n]["lon"]] for n in path if n in G]
                    data["routes"][rk]["coordinates"] = coords
            return jsonify({"status": "success", "data": data}), 200
        return jsonify({"status": "error", "message": "Simulation file not found"}), 404

    @app.route("/api/benchmarks", methods=["GET"])
    def get_benchmarks() -> Any:
        """Return the benchmark evaluation summary."""
        import json
        from pathlib import Path
        bench_file = Path.cwd() / "backend" / "data" / "processed" / "benchmark_evaluation_summary.json"
        if bench_file.exists():
            with open(bench_file, "r") as f:
                data = json.load(f)
            return jsonify({"status": "success", "data": data}), 200
        return jsonify({"status": "error", "message": "Benchmark file not found"}), 404

    @app.route("/api/model-metrics", methods=["GET"])
    def get_model_metrics() -> Any:
        """Return the ML speed prediction model benchmark table."""
        import json
        from pathlib import Path
        metrics_file = Path.cwd() / "backend" / "data" / "processed" / "model_benchmark_table.json"
        if metrics_file.exists():
            with open(metrics_file, "r") as f:
                data = json.load(f)
            return jsonify({"status": "success", "data": data}), 200
        return jsonify({"status": "error", "message": "Metrics file not found"}), 404

    @app.route("/api/corridors", methods=["GET"])
    def get_corridors() -> Any:
        """Return list of monitored corridors with free-flow speeds and predicted factors."""
        try:
            router, _ = get_router()
            predictor = router.predictor
            corridors = []
            if predictor and hasattr(predictor, "free_flow_speeds"):
                for cid, v_free in predictor.free_flow_speeds.items():
                    pred_speed = predictor.predict_speed(cid, hour=18, day_of_week=2)
                    c_e = round(v_free / max(pred_speed, 1.0), 2)
                    corridors.append({
                        "corridor_id": cid,
                        "free_flow_speed_kmh": round(v_free, 1),
                        "evening_peak_speed_kmh": round(pred_speed, 1),
                        "congestion_factor": c_e,
                    })
            return jsonify({"status": "success", "count": len(corridors), "corridors": corridors}), 200
        except Exception as exc:
            return jsonify({"status": "error", "message": str(exc)}), 500

    logger.info(
        "LifeRoute Flask app initialized (city: %s, radius: %dm)",
        app_cfg.city_name,
        app_cfg.radius_m,
    )
    return app

