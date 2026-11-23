"""Comprehensive evaluation and benchmark script for LifeRoute.

Compares Static Shortest Path Routing vs. LifeRoute Dynamic Proactive Routing
across multiple emergency scenarios, departure hours, and OD pairs in Bengaluru.
Generates an interactive Folium visualization: preview_bengaluru_routes.html.
"""

from __future__ import annotations

import json
import logging
import math
import os
import sys
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Set MPLCONFIGDIR to local directory
os.environ["MPLCONFIGDIR"] = str(PROJECT_ROOT / "cache" / "mpl")
os.makedirs(PROJECT_ROOT / "cache" / "mpl", exist_ok=True)

import folium
import networkx as nx
import pandas as pd

from backend.app.graph.bengaluru_network import build_bengaluru_network
from backend.app.model.traffic_model import TrafficPredictor
from backend.app.routing.router import LifeRouteRouter, RoutingMode

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("evaluate_liferoute")


def compute_actual_experienced_travel_time(
    router: LifeRouteRouter,
    path_edges: List[tuple],
    departure_time: datetime,
) -> float:
    """Calculate the true travel time experienced along a path during dynamic traffic conditions."""
    curr_time = departure_time
    total_time_s = 0.0
    for u, v, k in path_edges:
        edge_data = router.graph[u][v]
        w_e, _, _ = router.compute_edge_weight(
            u, v, edge_data, curr_time, mode=RoutingMode.DYNAMIC_PROACTIVE
        )
        total_time_s += w_e
        curr_time += timedelta(seconds=w_e)
    return total_time_s


def run_benchmark_scenarios(
    router: LifeRouteRouter,
    hospitals: List[Dict[str, Any]],
) -> Dict[str, Any]:
    """Run comprehensive comparative evaluations across multiple OD pairs and departure times."""
    test_scenarios = [
        {
            "name": "Evening Peak: East Outer Ring to Silk Board Interchange",
            "origin": "N_BENNIGANAHALLI",
            "destination": "N_SILK_BOARD",
            "departure": datetime(2026, 11, 18, 18, 30),
            "period": "Evening Peak (18:30)",
        },
        {
            "name": "Evening Peak: Brookfield Tech Corridor to Central Hospitals",
            "origin": "N_BROOKFIELD",
            "destination": "N_TOWN_HALL",
            "departure": datetime(2026, 11, 18, 18, 45),
            "period": "Evening Peak (18:45)",
        },
        {
            "name": "Morning Peak: North Outer Corridor (Hebbal) to Koramangala Hub",
            "origin": "N_HEBBAL",
            "destination": "N_KORAMANGALA_100FT",
            "departure": datetime(2026, 11, 16, 9, 15),
            "period": "Morning Peak (09:15)",
        },
        {
            "name": "Off-Peak Midnight: Mysore Road to Shivajinagar",
            "origin": "N_RR_NAGAR",
            "destination": "N_SHIVAJINAGAR",
            "departure": datetime(2026, 11, 18, 2, 0),
            "period": "Off-Peak Midnight (02:00)",
        },
    ]

    results = []

    for sc in test_scenarios:
        orig = sc["origin"]
        dest = sc["destination"]
        dep = sc["departure"]

        # 1. Static Route (Standard GPS free-flow shortest path)
        static_route = router.find_shortest_path(
            orig, dest, departure_time=dep, mode=RoutingMode.STATIC_FREE_FLOW
        )
        # Experienced travel time if taking static path during live traffic
        experienced_static_time_s = compute_actual_experienced_travel_time(
            router, static_route.path_edges, dep
        )

        # 2. LifeRoute Dynamic Proactive Route
        dynamic_route = router.find_shortest_path(
            orig, dest, departure_time=dep, mode=RoutingMode.DYNAMIC_PROACTIVE
        )

        time_saved_min = (experienced_static_time_s - dynamic_route.travel_time_seconds) / 60.0
        pct_improvement = (
            (time_saved_min / (experienced_static_time_s / 60.0)) * 100.0
            if experienced_static_time_s > 0
            else 0.0
        )

        res_entry = {
            "scenario": sc["name"],
            "period": sc["period"],
            "origin": orig,
            "destination": dest,
            "static_path": static_route.path_nodes,
            "static_distance_km": static_route.distance_km,
            "static_nominal_time_min": static_route.travel_time_minutes,
            "static_experienced_time_min": round(experienced_static_time_s / 60.0, 2),
            "dynamic_path": dynamic_route.path_nodes,
            "dynamic_distance_km": dynamic_route.distance_km,
            "dynamic_travel_time_min": dynamic_route.travel_time_minutes,
            "time_saved_minutes": round(time_saved_min, 2),
            "improvement_percent": round(pct_improvement, 1),
            "bypassed_congested_corridors": list(
                set(static_route.corridors_traversed) - set(dynamic_route.corridors_traversed)
            ),
        }
        results.append(res_entry)

    # Hospital Capacity Dispatch Test
    incident_loc = "N_BENNIGANAHALLI"
    rush_dt = datetime(2026, 11, 18, 18, 30)
    best_hosp, hosp_route = router.find_best_hospital(
        incident_loc, hospitals, departure_time=rush_dt, mode=RoutingMode.DYNAMIC_PROACTIVE, require_capacity=True
    )

    hospital_dispatch_summary = {
        "incident_location": incident_loc,
        "departure_time": str(rush_dt),
        "selected_hospital": best_hosp["name"],
        "beds_available": best_hosp["beds_available"],
        "icu_available": best_hosp["icu_available"],
        "trauma_center": best_hosp["trauma_center"],
        "route_travel_time_min": hosp_route.travel_time_minutes,
        "route_distance_km": hosp_route.distance_km,
        "route_nodes": hosp_route.path_nodes,
    }

    return {
        "scenarios": results,
        "hospital_dispatch": hospital_dispatch_summary,
    }


def generate_interactive_map(
    router: LifeRouteRouter,
    hospitals: List[Dict[str, Any]],
    output_html_path: Path,
) -> None:
    """Generate high-resolution Folium map displaying road network, hospitals, and route comparison."""
    center_lat, center_lon = 12.9600, 77.6200
    m = folium.Map(
        location=[center_lat, center_lon],
        zoom_start=12,
        tiles="OpenStreetMap",
        control_scale=True,
    )

    G = router.graph

    # Draw all road network edges
    drawn_edges = set()
    for u, v, data in G.edges(data=True):
        edge_key = tuple(sorted([u, v]))
        if edge_key in drawn_edges:
            continue
        drawn_edges.add(edge_key)

        u_data = G.nodes[u]
        v_data = G.nodes[v]
        p1 = (u_data["lat"], u_data["lon"])
        p2 = (v_data["lat"], v_data["lon"])

        cid = data.get("corridor_id")
        cname = data.get("corridor_name", "")
        v_free = data.get("free_flow_speed_kmh", 40.0)

        if cid:
            # Monitored corridor
            color = "#3388ff"
            weight = 4
            opacity = 0.7
            tooltip = f"Corridor: {cname} ({cid}) | v_free: {v_free} km/h"
        else:
            color = "#999999"
            weight = 2
            opacity = 0.5
            tooltip = f"Arterial Road ({data.get('highway', 'road')}) | v_free: {v_free} km/h"

        folium.PolyLine(
            locations=[p1, p2],
            color=color,
            weight=weight,
            opacity=opacity,
            tooltip=tooltip,
        ).add_to(m)

    # Add network nodes
    for n, d in G.nodes(data=True):
        folium.CircleMarker(
            location=[d["lat"], d["lon"]],
            radius=4,
            color="#444444",
            fill=True,
            fill_color="#ffffff",
            fill_opacity=0.9,
            popup=f"<b>Node:</b> {n}<br>{d.get('label', '')}",
        ).add_to(m)

    # Plot Hospitals
    for h in hospitals:
        beds = h.get("beds_available", 0)
        is_avail = beds > 0
        icon_color = "green" if is_avail else "red"
        status_text = f"<span style='color:green;font-weight:bold;'>Available ({beds} beds, {h.get('icu_available', 0)} ICU)</span>" if is_avail else "<span style='color:red;font-weight:bold;'>AT CAPACITY (0 beds)</span>"

        popup_html = f"""
        <div style="font-family:sans-serif;width:220px;">
            <h4 style="margin:0 0 5px 0;">{h['name']}</h4>
            <b>Status:</b> {status_text}<br>
            <b>Trauma Center:</b> {'Yes' if h.get('trauma_center') else 'No'}<br>
            <b>Snapped Node:</b> {h.get('node_id')}<br>
        </div>
        """
        folium.Marker(
            location=[h["lat"], h["lon"]],
            icon=folium.Icon(color=icon_color, icon="plus-sign", prefix="glyphicon"),
            popup=folium.Popup(popup_html, max_width=300),
            tooltip=f"Hospital: {h['name']} ({beds} beds)",
        ).add_to(m)

    # Demonstration Route: Benniganahalli to Silk Board at Evening Rush Hour (18:30)
    orig = "N_BENNIGANAHALLI"
    dest = "N_SILK_BOARD"
    rush_dt = datetime(2026, 11, 18, 18, 30)

    static_r = router.find_shortest_path(orig, dest, departure_time=rush_dt, mode=RoutingMode.STATIC_FREE_FLOW)
    dyn_r = router.find_shortest_path(orig, dest, departure_time=rush_dt, mode=RoutingMode.DYNAMIC_PROACTIVE)

    # Static Route (dashed red/orange line)
    static_coords = [(G.nodes[n]["lat"], G.nodes[n]["lon"]) for n in static_r.path_nodes]
    folium.PolyLine(
        locations=static_coords,
        color="#d9534f",
        weight=6,
        dash_array="8, 8",
        opacity=0.85,
        tooltip=f"Static Route (Choked via ORR/Inner Ring): {compute_actual_experienced_travel_time(router, static_r.path_edges, rush_dt)/60:.1f} min",
    ).add_to(m)

    # Dynamic Proactive Route (solid vibrant green line)
    dyn_coords = [(G.nodes[n]["lat"], G.nodes[n]["lon"]) for n in dyn_r.path_nodes]
    folium.PolyLine(
        locations=dyn_coords,
        color="#28a745",
        weight=6,
        opacity=0.95,
        tooltip=f"LifeRoute Proactive Route (Bypasses Congestion): {dyn_r.travel_time_minutes} min ({dyn_r.distance_km} km)",
    ).add_to(m)

    # Incident marker S
    orig_node_data = G.nodes[orig]
    folium.Marker(
        location=[orig_node_data["lat"], orig_node_data["lon"]],
        icon=folium.Icon(color="darkpurple", icon="warning-sign", prefix="glyphicon"),
        popup=f"<b>EMERGENCY INCIDENT ORIGIN S</b><br>{orig_node_data.get('label')}",
        tooltip="Incident Location S",
    ).add_to(m)

    # Target marker
    dest_node_data = G.nodes[dest]
    folium.Marker(
        location=[dest_node_data["lat"], dest_node_data["lon"]],
        icon=folium.Icon(color="blue", icon="flag", prefix="glyphicon"),
        popup=f"<b>DESTINATION HUB</b><br>{dest_node_data.get('label')}",
        tooltip="Destination Hub",
    ).add_to(m)

    m.save(str(output_html_path))
    logger.info("Saved interactive route visualization to %s", output_html_path)


def main():
    model_dir = PROJECT_ROOT / "backend" / "data" / "processed" / "model"
    output_dir = PROJECT_ROOT / "backend" / "data" / "processed"
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Loading trained traffic prediction model from %s...", model_dir)
    predictor = TrafficPredictor.load(model_dir)

    logger.info("Building Bengaluru road network with monitored corridors...")
    G, hospitals = build_bengaluru_network(free_flow_speeds=predictor.free_flow_speeds)

    router = LifeRouteRouter(G, predictor=predictor)

    logger.info("Running comparative benchmark scenarios across peak & off-peak conditions...")
    benchmark_results = run_benchmark_scenarios(router, hospitals)

    print("\n" + "=" * 90)
    print("LIFEROUTE EMERGENCY DISPATCH ROUTING BENCHMARK (Bengaluru Road Network)")
    print("=" * 90)
    for res in benchmark_results["scenarios"]:
        print(f"Scenario:    {res['scenario']}")
        print(f"Time Period: {res['period']}")
        print(f"  [Static Baseline] Distance: {res['static_distance_km']} km | Nominal: {res['static_nominal_time_min']} min | ACTUAL Time: {res['static_experienced_time_min']} min")
        print(f"  [LifeRoute AI]   Distance: {res['dynamic_distance_km']} km | PREDICTED Time: {res['dynamic_travel_time_min']} min")
        print(f"  >>> GAIN: Saved {res['time_saved_minutes']} min ({res['improvement_percent']}% faster) by avoiding {res['bypassed_congested_corridors']} <<<\n")

    h_disp = benchmark_results["hospital_dispatch"]
    print("=" * 90)
    print("CAPACITY-AWARE EMERGENCY HOSPITAL DISPATCH TEST")
    print("=" * 90)
    print(f"Incident:          {h_disp['incident_location']} at {h_disp['departure_time']}")
    print(f"Selected Facility: {h_disp['selected_hospital']}")
    print(f"Status:            {h_disp['beds_available']} available beds (ICU: {h_disp['icu_available']}, Trauma: {h_disp['trauma_center']})")
    print(f"Optimal Traversal: {h_disp['route_travel_time_min']} minutes ({h_disp['route_distance_km']} km)")
    print(f"Route:             {' -> '.join(h_disp['route_nodes'])}")
    print("=" * 90 + "\n")

    # Save summary JSON
    summary_file = output_dir / "benchmark_evaluation_summary.json"
    with open(summary_file, "w", encoding="utf-8") as f:
        json.dump(benchmark_results, f, indent=2)
    logger.info("Saved benchmark evaluation summary to %s", summary_file)

    # Save hospital CSV
    hosp_df = pd.DataFrame(hospitals)
    hosp_csv = output_dir / "bengaluru_hospitals.csv"
    hosp_df.to_csv(hosp_csv, index=False)
    logger.info("Saved hospitals dataset to %s", hosp_csv)

    # Save graphml (convert None attribute values to empty strings for GraphML compliance)
    graphml_path = output_dir / "bengaluru_network.graphml"
    G_export = G.copy()
    for _, _, d in G_export.edges(data=True):
        for k, v in list(d.items()):
            if v is None:
                d[k] = ""
    nx.write_graphml(G_export, str(graphml_path))
    logger.info("Saved road network GraphML to %s", graphml_path)

    # Save edges CSV
    edge_records = []
    for u, v, d in G.edges(data=True):
        edge_records.append({
            "u": u,
            "v": v,
            "length_m": d.get("length_m"),
            "free_flow_speed_kmh": d.get("free_flow_speed_kmh"),
            "free_flow_time_s": d.get("free_flow_time_s"),
            "highway": d.get("highway"),
            "corridor_id": d.get("corridor_id"),
            "corridor_name": d.get("corridor_name"),
        })
    edges_df = pd.DataFrame(edge_records)
    edges_csv = output_dir / "bengaluru_edges.csv"
    edges_df.to_csv(edges_csv, index=False)
    logger.info("Saved edges dataset to %s", edges_csv)

    # Generate interactive visualization
    html_preview = PROJECT_ROOT / "preview_bengaluru_routes.html"
    generate_interactive_map(router, hospitals, html_preview)


if __name__ == "__main__":
    main()
