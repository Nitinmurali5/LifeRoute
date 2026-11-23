"""Master graph build script for LifeRoute (Phase 1).

Executes end-to-end graph construction:
1. Synthetic road grid graph (Paper Table 4 scenario) -> JSON
2. Real-world city road network (OSM download) -> GraphML & edges.csv
3. Snapped hospital locations with simulated bed availability -> hospitals.csv
4. Visualizations:
   - preview_grid.png (matplotlib)
   - preview_city.html (folium)
5. Prints comprehensive terminal summary of network metrics.

Paper notation:
- G = (V, E)
- d_e: length in meters (length_m)
- v_{e,free}: free-flow speed in km/h (free_flow_speed_kmh)
- w_e(t) = (d_e / v_{e,free}) * C_hat_e(t)
- free_flow_time_s = d_e / (v_{e,free} / 3.6)
"""

from __future__ import annotations

import collections
import logging
import os
import shutil
import sys
from pathlib import Path
from typing import Any, Dict, List

# Ensure project root is on sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import folium
import matplotlib
matplotlib.use("Agg")  # Non-interactive backend
import matplotlib.pyplot as plt
import networkx as nx

from backend.app.config import AppConfig, load_config
from backend.app.graph.city_graph import build_city_graph
from backend.app.graph.grid_graph import (
    build_synthetic_grid_graph,
    save_grid_scenario_json,
)
from backend.app.graph.hospitals import extract_and_assign_hospitals

# Setup logging (only warnings/errors during script unless requested, info logged to stream)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("build_graphs")


def generate_grid_preview(
    G: nx.DiGraph,
    incident_node: int,
    hospitals: List[Dict[str, Any]],
    output_path: Path,
) -> None:
    """Generate high-resolution PNG visualization of the synthetic road grid.

    Args:
        G: Synthetic grid graph.
        incident_node: ID of incident node S.
        hospitals: List of hospital metadata dicts.
        output_path: Destination path for PNG image.
    """
    logger.info("Generating grid visualization: %s", output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(12, 10), dpi=300)
    fig.patch.set_facecolor("#f8f9fa")
    ax.set_facecolor("#ffffff")

    # Node positions (x, y)
    pos = {n: (data["x"], data["y"]) for n, data in G.nodes(data=True)}

    # Group edges by road class for distinct styling
    class_styles = {
        "primary": {"color": "#1f77b4", "width": 1.8, "alpha": 0.85, "label": "Primary (50 km/h)"},
        "secondary": {"color": "#ff7f0e", "width": 1.2, "alpha": 0.75, "label": "Secondary (40 km/h)"},
        "residential": {"color": "#b0bec5", "width": 0.8, "alpha": 0.55, "label": "Residential (30 km/h)"},
    }

    edge_groups: Dict[str, List[tuple]] = collections.defaultdict(list)
    for u, v, data in G.edges(data=True):
        hw = data.get("highway", "residential")
        edge_groups[hw].append((u, v))

    for hw, edges in edge_groups.items():
        style = class_styles.get(hw, class_styles["residential"])
        nx.draw_networkx_edges(
            G,
            pos,
            edgelist=edges,
            ax=ax,
            edge_color=style["color"],
            width=style["width"],
            alpha=style["alpha"],
            arrows=True,
            arrowsize=6,
            arrowstyle="-|>",
        )

    # Base nodes
    nx.draw_networkx_nodes(
        G,
        pos,
        ax=ax,
        node_size=18,
        node_color="#90a4ae",
        alpha=0.6,
    )

    # Incident node S
    inc_x, inc_y = pos[incident_node]
    ax.scatter(
        inc_x,
        inc_y,
        s=260,
        c="#d32f2f",
        marker="*",
        edgecolors="black",
        linewidths=1.2,
        zorder=10,
        label=f"Incident Node S (#{incident_node})",
    )

    # Hospital nodes
    hosp_nodes = [h["nearest_node"] for h in hospitals]
    for h in hospitals:
        hn = h["nearest_node"]
        hx, hy = pos[hn]
        beds = h["beds_available"]
        at_cap = h["at_capacity"]

        h_color = "#d32f2f" if at_cap else "#2e7d32"
        h_label = f"{h['hospital_id']}: {beds} beds ({'FULL' if at_cap else 'AVAIL'})"

        ax.scatter(
            hx,
            hy,
            s=220,
            c=h_color,
            marker="s",
            edgecolors="black",
            linewidths=1.2,
            zorder=9,
        )
        ax.text(
            hx + 40,
            hy + 40,
            h_label,
            fontsize=8,
            fontweight="bold",
            color="#212121",
            bbox=dict(boxstyle="round,pad=0.2", facecolor="#ffffff", alpha=0.85, edgecolor="#9e9e9e"),
            zorder=11,
        )

    # Custom legend items
    for hw, style in class_styles.items():
        ax.plot([], [], color=style["color"], linewidth=style["width"], label=style["label"])

    ax.plot([], [], marker="s", color="#2e7d32", linestyle="None", markersize=9, label="Hospital (Beds Available)")
    ax.plot([], [], marker="s", color="#d32f2f", linestyle="None", markersize=9, label="Hospital (At Capacity - 0 Beds)")

    ax.set_title(
        f"LifeRoute Synthetic Road Grid (Paper Table 4 Scenario)\n"
        f"Nodes: {G.number_of_nodes()} | Directed Edges: {G.number_of_edges()} | Strongly Connected: {nx.is_strongly_connected(G)}",
        fontsize=13,
        fontweight="bold",
        pad=15,
    )
    ax.legend(loc="upper right", framealpha=0.95, fontsize=9)
    ax.axis("off")
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    logger.info("Grid visualization saved to %s", output_path)


def generate_city_preview(
    G: nx.MultiDiGraph,
    hospitals: List[Dict[str, Any]],
    output_path: Path,
    config: AppConfig,
) -> None:
    """Generate interactive Folium HTML map for the city road network and hospitals.

    Args:
        G: City road network MultiDiGraph.
        hospitals: List of hospitals with coordinates and bed capacities.
        output_path: Target path for HTML file.
        config: AppConfig instance.
    """
    logger.info("Generating city interactive Folium preview: %s", output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Calculate center coordinate from nodes
    lats = [d["y"] for _, d in G.nodes(data=True) if "y" in d]
    lons = [d["x"] for _, d in G.nodes(data=True) if "x" in d]
    center_lat = sum(lats) / len(lats) if lats else 11.0168
    center_lon = sum(lons) / len(lons) if lons else 76.9558

    m = folium.Map(
        location=[center_lat, center_lon],
        zoom_start=13,
        tiles="CartoDB positron",
        control_scale=True,
    )

    # Layer for Road Network (focus on arterial & collector roads for fast rendering)
    road_layer = folium.FeatureGroup(name="Drivable Road Network", show=True)

    # Road color mapping
    road_colors = {
        "motorway": "#d32f2f",
        "trunk": "#e64a19",
        "primary": "#1976d2",
        "secondary": "#388e3c",
        "tertiary": "#fbc02d",
        "residential": "#9e9e9e",
        "unclassified": "#bdbdbd",
    }

    # Add edges as polylines
    # To keep the HTML file lean and responsive, we render arterial/collector roads
    # and a representative sample of residential roads
    rendered_edges = 0
    for u, v, k, data in G.edges(keys=True, data=True):
        hw = data.get("highway", "residential")
        # Render major roads or sampled residential
        if hw in ["motorway", "trunk", "primary", "secondary", "tertiary"] or (rendered_edges % 3 == 0):
            u_node = G.nodes[u]
            v_node = G.nodes[v]
            if "y" in u_node and "x" in u_node and "y" in v_node and "x" in v_node:
                coords = [(u_node["y"], u_node["x"]), (v_node["y"], v_node["x"])]
                color = road_colors.get(hw, "#757575")
                folium.PolyLine(
                    locations=coords,
                    color=color,
                    weight=2.5 if hw in ["primary", "trunk", "motorway"] else 1.2,
                    opacity=0.7,
                    tooltip=f"Road: {hw} | Speed: {data.get('free_flow_speed_kmh', 30)} km/h | Length: {data.get('length_m', 0):.1f}m",
                ).add_to(road_layer)
                rendered_edges += 1

    road_layer.add_to(m)

    # Hospital markers layer
    hosp_layer = folium.FeatureGroup(name="Emergency Hospital Facilities (Simulated)", show=True)

    for h in hospitals:
        lat = h["lat"]
        lon = h["lon"]
        beds = h["beds_available"]
        at_cap = h["at_capacity"]
        h_id = h["hospital_id"]
        name = h["name"]
        nearest_node = h["nearest_node"]

        status_text = "FULL (At Capacity)" if at_cap else f"AVAILABLE ({beds} beds)"
        badge_color = "red" if at_cap else "green"

        popup_html = f"""
        <div style="font-family: Arial, sans-serif; font-size: 12px; width: 230px;">
            <h4 style="margin: 0 0 6px 0; color: #1976d2;">{name}</h4>
            <p style="margin: 3px 0;"><b>ID:</b> {h_id}</p>
            <p style="margin: 3px 0;"><b>Nearest Road Node:</b> {nearest_node}</p>
            <p style="margin: 3px 0;"><b>Coordinates:</b> {lat:.5f}, {lon:.5f}</p>
            <hr style="margin: 6px 0; border: 0; border-top: 1px solid #e0e0e0;"/>
            <p style="margin: 3px 0; font-weight: bold; color: {'#d32f2f' if at_cap else '#2e7d32'};">
                Bed Status: {status_text}
            </p>
            <small style="color: #757575;">*Simulated parameter for research</small>
        </div>
        """

        folium.CircleMarker(
            location=[lat, lon],
            radius=9,
            color="#212121",
            weight=1.5,
            fill=True,
            fill_color="#d32f2f" if at_cap else "#2e7d32",
            fill_opacity=0.9,
            popup=folium.Popup(popup_html, max_width=260),
            tooltip=f"{name} - {status_text}",
        ).add_to(hosp_layer)

    hosp_layer.add_to(m)
    folium.LayerControl().add_to(m)

    m.save(str(output_path))
    logger.info("Interactive city preview saved to %s", output_path)


def compute_network_metrics(G: nx.Graph | nx.DiGraph | nx.MultiDiGraph) -> Dict[str, Any]:
    """Compute aggregate road network metrics following paper notation.

    Args:
        G: Road network graph.

    Returns:
        Dictionary of network summary statistics.
    """
    total_length_m = 0.0
    classes: collections.Counter = collections.Counter()

    is_multigraph = G.is_multigraph()

    if is_multigraph:
        for u, v, k, d in G.edges(keys=True, data=True):
            length = float(d.get("length_m", d.get("length", 0.0)))
            total_length_m += length
            hw = d.get("highway", "unclassified")
            classes[hw] += 1
    else:
        for u, v, d in G.edges(data=True):
            length = float(d.get("length_m", d.get("length", 0.0)))
            total_length_m += length
            hw = d.get("highway", "unclassified")
            classes[hw] += 1

    return {
        "num_nodes": G.number_of_nodes(),
        "num_edges": G.number_of_edges(),
        "total_road_km": round(total_length_m / 1000.0, 2),
        "classes": dict(classes.most_common()),
        "is_strongly_connected": nx.is_strongly_connected(G),
    }


def print_formatted_summary(
    grid_metrics: Dict[str, Any],
    grid_hospitals: List[Dict[str, Any]],
    city_metrics: Dict[str, Any],
    city_hospitals: List[Dict[str, Any]],
    config: AppConfig,
    paths: Dict[str, Path],
) -> None:
    """Print the final comprehensive formatted summary required by Phase 1.

    Args:
        grid_metrics: Summary statistics for synthetic grid.
        grid_hospitals: Hospitals in synthetic grid.
        city_metrics: Summary statistics for real city graph.
        city_hospitals: Hospitals in real city graph.
        config: Application configuration.
        paths: Paths of created artifacts.
    """
    grid_beds_avail = sum(h["beds_available"] for h in grid_hospitals)
    city_beds_avail = sum(h["beds_available"] for h in city_hospitals)
    city_beds_cap = sum(1 for h in city_hospitals if h["at_capacity"])

    sep = "=" * 74
    subsep = "-" * 74

    print(sep)
    print("   LIFEROUTE - AI-POWERED EMERGENCY AMBULANCE NAVIGATION SYSTEM")
    print("   Phase 1 of 9: Foundation & Network Graph Build Summary")
    print(sep)
    print(f" Research Simulation Context: IEEE Proposed-System Paper")
    print(f" Configured City:     {config.city_name}")
    print(f" Download Radius:     {config.radius_m:,} meters")
    print(f" Deterministic Seed:  {config.random_seed}")
    print(subsep)

    print(" 1. SYNTHETIC ROAD GRID NETWORK (Table 4 Scenario)")
    print(f"    - Nodes:                  {grid_metrics['num_nodes']:,}")
    print(f"    - Directed Edges:         {grid_metrics['num_edges']:,}")
    print(f"    - Strongly Connected:     {grid_metrics['is_strongly_connected']}")
    print(f"    - Total Road Length:      {grid_metrics['total_road_km']:.2f} km")
    print("    - Edges per Road Class:")
    for hw, count in grid_metrics["classes"].items():
        print(f"        * {hw:<14} : {count:>5} edges")
    print(f"    - Hospital Facilities:    {len(grid_hospitals)}")
    for h in grid_hospitals:
        cap_str = "AT CAPACITY (0 beds)" if h["at_capacity"] else f"{h['beds_available']} beds available"
        print(f"        * [{h['hospital_id']}] Node #{h['nearest_node']:<4} -> {cap_str}")
    print(f"    - Total Simulated Beds:   {grid_beds_avail} available across all hospitals")

    print(subsep)
    print(f" 2. REAL CITY ROAD NETWORK ({config.city_name})")
    print(f"    - Nodes (SCC Component):  {city_metrics['num_nodes']:,}")
    print(f"    - Directed Edges:         {city_metrics['num_edges']:,}")
    print(f"    - Strongly Connected:     {city_metrics['is_strongly_connected']}")
    print(f"    - Total Road Length:      {city_metrics['total_road_km']:.2f} km")
    print("    - Edges per Road Class (Top Classes):")
    for hw, count in list(city_metrics["classes"].items())[:8]:
        print(f"        * {hw:<18} : {count:>6} edges")
    print(f"    - Hospital Facilities:    {len(city_hospitals)} found & snapped to network")
    print(f"    - Simulated Beds:         {city_beds_avail} available ({city_beds_cap} at capacity)")

    print(subsep)
    print(" 3. GENERATED ARTIFACTS")
    for name, path in paths.items():
        print(f"    - {name:<22}: {path}")

    print(sep)
    print(" [OK] Phase 1 graph foundation build completed successfully.")
    print(sep)


def main() -> None:
    """Run full graph construction pipeline."""
    config = load_config()

    # Define paths
    base_data_dir = PROJECT_ROOT / "backend" / "data" / "processed"
    base_data_dir.mkdir(parents=True, exist_ok=True)

    grid_json_path = base_data_dir / "grid_scenario.json"
    grid_png_path = base_data_dir / "preview_grid.png"
    city_html_path = base_data_dir / "preview_city.html"

    # Also place preview copies at project root for easy user inspection
    root_grid_png_path = PROJECT_ROOT / "preview_grid.png"
    root_city_html_path = PROJECT_ROOT / "preview_city.html"

    # 1. Build Synthetic Grid
    logger.info("Building synthetic road grid...")
    grid_G, incident_node, grid_hospitals = build_synthetic_grid_graph(config=config)
    save_grid_scenario_json(
        G=grid_G,
        incident_node=incident_node,
        hospitals=grid_hospitals,
        output_path=grid_json_path,
    )
    generate_grid_preview(grid_G, incident_node, grid_hospitals, grid_png_path)
    shutil.copyfile(grid_png_path, root_grid_png_path)
    grid_metrics = compute_network_metrics(grid_G)

    # 2. Build City Graph
    logger.info("Building real city road network...")
    city_G, graphml_path, edges_csv_path = build_city_graph(config=config, output_dir=base_data_dir)
    city_hospitals, hospitals_csv_path = extract_and_assign_hospitals(
        G=city_G,
        config=config,
        output_dir=base_data_dir,
    )
    generate_city_preview(city_G, city_hospitals, city_html_path, config)
    shutil.copyfile(city_html_path, root_city_html_path)
    city_metrics = compute_network_metrics(city_G)

    # 3. Print Final Summary
    artifacts = {
        "Grid Scenario (JSON)": grid_json_path,
        "City GraphML": graphml_path,
        "City Edges (CSV)": edges_csv_path,
        "Hospitals (CSV)": hospitals_csv_path,
        "Grid Preview (PNG)": root_grid_png_path,
        "City Preview (HTML)": root_city_html_path,
    }

    print_formatted_summary(
        grid_metrics=grid_metrics,
        grid_hospitals=grid_hospitals,
        city_metrics=city_metrics,
        city_hospitals=city_hospitals,
        config=config,
        paths=artifacts,
    )


if __name__ == "__main__":
    main()

