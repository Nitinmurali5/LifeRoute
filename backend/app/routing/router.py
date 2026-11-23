"""Time-dependent shortest path router for LifeRoute emergency ambulance dispatch.

Implements modified Dijkstra and A* routing over urban road graphs weighted by
machine-learning predicted travel times and dynamic congestion factors.

Paper Notation & Equations:
- Graph G = (V, E)
- d_e: length in meters (length_m)
- v_{e,free}: free-flow speed in km/h (free_flow_speed_kmh)
- t_{e,free} = d_e / (v_{e,free} / 3.6) [seconds]
- w_e(t) = (d_e / v_{e,free}) * C_hat_e(t) [seconds] (Paper Eq. 1)
  where C_hat_e(t) >= 1.0 is the predicted dynamic congestion factor.
- V2I Traffic Signal Preemption:
  d_j(t) = 0                       if V2I Green Wave engaged
         = (1 - gamma_j) * (C_j/2) without preemption
"""

from __future__ import annotations

import heapq
import logging
import math
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Set, Tuple, Union

import networkx as nx

from backend.app.model.traffic_model import TrafficPredictor

logger = logging.getLogger(__name__)


def haversine_distance_m(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Compute great-circle distance in meters between two lat/lon coordinates."""
    r = 6371000.0  # Earth radius in meters
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlambda = math.radians(lon2 - lon1)
    a = (
        math.sin(dphi / 2.0) ** 2
        + math.cos(phi1) * math.cos(phi2) * math.sin(dlambda / 2.0) ** 2
    )
    c = 2.0 * math.atan2(math.sqrt(a), math.sqrt(1.0 - a))
    return r * c


@dataclass
class SignalizedIntersection:
    """Represents an intelligent traffic signal controller with V2I DSRC / C-V2X preemption.

    Paper Eq. Formulation:
      Intersection Delay d_j(t) = 0                       if V2I Green Wave engaged
                                = (1 - gamma_j) * (C_j / 2) without preemption
    """
    id: str
    name: str
    latitude: float
    longitude: float
    cycle_time_seconds: float = 90.0      # C_j: signal cycle time (s)
    green_ratio: float = 0.35             # gamma_j: effective green ratio (35% default)
    geofence_radius_meters: float = 350.0  # DSRC / C-V2X virtual trigger horizon
    node_id: Optional[Any] = None

    def expected_delay_unpreempted(self) -> float:
        """Expected delay without V2I preemption: d_j = (1 - gamma_j) * (C_j / 2)."""
        return round((1.0 - self.green_ratio) * (self.cycle_time_seconds / 2.0), 2)

    def expected_delay_preempted(self) -> float:
        """Expected delay with V2I Green Wave preemption: d_j = 0."""
        return 0.0

    def time_saved_seconds(self) -> float:
        return self.expected_delay_unpreempted() - self.expected_delay_preempted()

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "name": self.name,
            "latitude": self.latitude,
            "longitude": self.longitude,
            "cycle_time_seconds": self.cycle_time_seconds,
            "green_ratio": self.green_ratio,
            "geofence_radius_meters": self.geofence_radius_meters,
            "expected_delay_unpreempted_s": self.expected_delay_unpreempted(),
            "expected_delay_preempted_s": self.expected_delay_preempted(),
            "time_saved_s": self.time_saved_seconds(),
        }


DEFAULT_BENGALURU_SIGNALS: Dict[str, SignalizedIntersection] = {
    "SIG_INDIRANAGAR_100FT": SignalizedIntersection(
        id="SIG_INDIRANAGAR_100FT",
        name="Indiranagar 100ft Rd / CMH Junction",
        latitude=12.97874,
        longitude=77.64663,
        cycle_time_seconds=90.0,
        green_ratio=0.35,
        geofence_radius_meters=350.0,
        node_id="N_INDIRANAGAR",
    ),
    "SIG_DOMLUR_FLYOVER": SignalizedIntersection(
        id="SIG_DOMLUR_FLYOVER",
        name="Domlur Flyover / Intermediate Ring Rd",
        latitude=12.96171,
        longitude=77.64174,
        cycle_time_seconds=100.0,
        green_ratio=0.30,
        geofence_radius_meters=350.0,
        node_id="N_DOMLUR",
    ),
    "SIG_EJIPURA": SignalizedIntersection(
        id="SIG_EJIPURA",
        name="Ejipura Junction / Inner Ring Road",
        latitude=12.93902,
        longitude=77.63277,
        cycle_time_seconds=90.0,
        green_ratio=0.35,
        geofence_radius_meters=350.0,
        node_id="N_KORAMANGALA_NGV",
    ),
    "SIG_KORAMANGALA_80FT": SignalizedIntersection(
        id="SIG_KORAMANGALA_80FT",
        name="Sony World / Koramangala 80ft Road",
        latitude=12.93493,
        longitude=77.62417,
        cycle_time_seconds=85.0,
        green_ratio=0.40,
        geofence_radius_meters=350.0,
        node_id="N_KORAMANGALA_80FT",
    ),
    "SIG_OLD_MADRAS_RD": SignalizedIntersection(
        id="SIG_OLD_MADRAS_RD",
        name="Old Madras Road / Swami Vivekananda Signal",
        latitude=12.99029,
        longitude=77.64948,
        cycle_time_seconds=95.0,
        green_ratio=0.35,
        geofence_radius_meters=350.0,
        node_id="N_ULSOOR",
    ),
    "SIG_TRINITY_CIRCLE": SignalizedIntersection(
        id="SIG_TRINITY_CIRCLE",
        name="Trinity Circle / MG Road",
        latitude=12.97448,
        longitude=77.62457,
        cycle_time_seconds=120.0,
        green_ratio=0.30,
        geofence_radius_meters=350.0,
        node_id="N_TRINITY_CIRCLE",
    ),
    "SIG_RICHMOND_CIRCLE": SignalizedIntersection(
        id="SIG_RICHMOND_CIRCLE",
        name="Richmond Circle / Hosur Road",
        latitude=12.96188,
        longitude=77.60275,
        cycle_time_seconds=110.0,
        green_ratio=0.32,
        geofence_radius_meters=350.0,
        node_id="N_RICHMOND_CIRCLE",
    ),
    "SIG_ADUGODI": SignalizedIntersection(
        id="SIG_ADUGODI",
        name="Adugodi Junction / Hosur Road",
        latitude=12.94472,
        longitude=77.60745,
        cycle_time_seconds=90.0,
        green_ratio=0.35,
        geofence_radius_meters=350.0,
        node_id="N_ADUGODI",
    ),
    "SIG_FORUM_DAIRY": SignalizedIntersection(
        id="SIG_FORUM_DAIRY",
        name="Forum Mall / Dairy Circle Approach",
        latitude=12.93595,
        longitude=77.61527,
        cycle_time_seconds=95.0,
        green_ratio=0.35,
        geofence_radius_meters=350.0,
        node_id="N_DAIRY_CIRCLE",
    ),
    "SIG_MADIWALA_CHECKPOST": SignalizedIntersection(
        id="SIG_MADIWALA_CHECKPOST",
        name="Madiwala Checkpost / St. John's Entrance",
        latitude=12.92908,
        longitude=77.62152,
        cycle_time_seconds=90.0,
        green_ratio=0.35,
        geofence_radius_meters=350.0,
        node_id="N_ST_JOHNS_EMERGENCY",
    ),
}


class RoutingMode(str, Enum):
    """Routing modes for comparative evaluation."""

    STATIC_FREE_FLOW = "static_free_flow"  # Naive baseline: assumes unhindered free-flow speeds
    DYNAMIC_PROACTIVE = "dynamic_proactive"  # LifeRoute: proactive routing using predicted congestion weights


@dataclass
class RouteResult:
    """Encapsulates the complete trajectory, duration, and metrics of a calculated route."""

    mode: RoutingMode
    origin_node: Any
    destination_node: Any
    departure_time: datetime
    arrival_time: datetime
    travel_time_seconds: float
    free_flow_time_seconds: float
    length_meters: float
    average_speed_kmh: float
    congestion_delay_seconds: float
    path_nodes: List[Any]
    path_edges: List[Tuple[Any, Any, int]]
    corridors_traversed: List[str]
    node_coords: List[Tuple[float, float]] = field(default_factory=list)
    # V2I Green Wave Attributes
    intersections_count: int = 0
    signal_delay_seconds: float = 0.0
    v2i_preemption_active: bool = True
    v2i_time_saved_seconds: float = 0.0
    intersections_detail: List[Dict[str, Any]] = field(default_factory=list)

    @property
    def travel_time_minutes(self) -> float:
        return round(self.travel_time_seconds / 60.0, 2)

    @property
    def distance_km(self) -> float:
        return round(self.length_meters / 1000.0, 2)

    @property
    def signal_delay_minutes(self) -> float:
        return round(self.signal_delay_seconds / 60.0, 2)

    @property
    def v2i_time_saved_minutes(self) -> float:
        return round(self.v2i_time_saved_seconds / 60.0, 2)


class LifeRouteRouter:
    """Graph router supporting static baseline and time-dependent proactive routing."""

    def __init__(
        self,
        graph: nx.DiGraph,
        predictor: Optional[TrafficPredictor] = None,
        max_speed_kmh: float = 80.0,
        signals: Optional[Dict[str, SignalizedIntersection]] = None,
    ) -> None:
        """Initialize router with a road network and optional traffic predictor.

        Args:
            graph: Strongly connected directed graph (NetworkX DiGraph or MultiDiGraph).
            predictor: Trained TrafficPredictor instance for dynamic congestion estimation.
            max_speed_kmh: Maximum network speed for admissible A* heuristic.
            signals: Optional dictionary of SignalizedIntersection instances.
        """
        self.graph = graph
        self.predictor = predictor
        self.max_speed_kmh = max(max_speed_kmh, 40.0)
        self.max_speed_ms = self.max_speed_kmh / 3.6
        self.signals = signals if signals is not None else DEFAULT_BENGALURU_SIGNALS

    def evaluate_v2i_green_wave(
        self,
        coords: List[Tuple[float, float]],
        v2i_enabled: bool = True,
        proximity_threshold_m: float = 120.0,
    ) -> Dict[str, Any]:
        """Evaluate V2I Green Wave preemption along a sequence of route coordinates.

        Identifies all signalized intersections within proximity_threshold_m of the trajectory.
        Paper Formula:
          d_j(t) = 0                       if V2I Green Wave engaged
                 = (1 - gamma_j) * (C_j/2) without preemption

        Args:
            coords: List of (lat, lon) tuples forming the path.
            v2i_enabled: If True, traffic signals are preempted to Green (0s delay).
            proximity_threshold_m: Distance within which a signal is deemed traversed.

        Returns:
            Dictionary with traversed intersections, total delay, and time saved.
        """
        if not coords or not self.signals:
            return {
                "traversed_signals": [],
                "signals_count": 0,
                "total_signal_delay_seconds": 0.0,
                "time_saved_seconds": 0.0,
                "v2i_enabled": v2i_enabled,
            }

        traversed = []
        total_delay = 0.0
        total_saved = 0.0

        for sig_id, sig in self.signals.items():
            min_dist = float("inf")
            for pt in coords:
                d = haversine_distance_m(pt[0], pt[1], sig.latitude, sig.longitude)
                if d < min_dist:
                    min_dist = d
                if min_dist <= proximity_threshold_m:
                    break

            if min_dist <= proximity_threshold_m:
                unpreempted_delay = sig.expected_delay_unpreempted()
                delay_s = 0.0 if v2i_enabled else unpreempted_delay
                saved_s = unpreempted_delay if v2i_enabled else 0.0

                total_delay += delay_s
                total_saved += saved_s

                traversed.append({
                    "id": sig.id,
                    "name": sig.name,
                    "latitude": sig.latitude,
                    "longitude": sig.longitude,
                    "cycle_time_s": sig.cycle_time_seconds,
                    "green_ratio": sig.green_ratio,
                    "delay_seconds": round(delay_s, 2),
                    "time_saved_seconds": round(saved_s, 2),
                    "status": "PREEMPTED_GREEN" if v2i_enabled else "RED_PHASE_DELAY",
                    "proximity_m": round(min_dist, 1),
                })

        return {
            "traversed_signals": traversed,
            "signals_count": len(traversed),
            "total_signal_delay_seconds": round(total_delay, 2),
            "time_saved_seconds": round(total_saved, 2),
            "v2i_enabled": v2i_enabled,
        }

    def compute_edge_weight(
        self,
        u: Any,
        v: Any,
        edge_data: Dict[str, Any],
        arrival_time: datetime,
        mode: RoutingMode,
    ) -> Tuple[float, float, float]:
        """Compute the dynamic traversal time w_e(t) in seconds according to Paper Eq. 1.

        Returns:
            Tuple of (traversal_time_s, length_m, free_flow_time_s).
        """
        # Segment length d_e in meters
        d_e = float(edge_data.get("length_m", edge_data.get("length", 100.0)))
        if d_e <= 0:
            d_e = 1.0

        # Free-flow speed v_{e,free} in km/h
        v_free = float(edge_data.get("free_flow_speed_kmh", 40.0))
        if v_free <= 0:
            v_free = 30.0

        # Free-flow time in seconds: t_{e,free} = d_e / (v_{e,free} / 3.6)
        t_free = d_e / (v_free / 3.6)

        if mode == RoutingMode.STATIC_FREE_FLOW or self.predictor is None:
            # Baseline static routing: w_e = t_{e,free}
            return t_free, d_e, t_free

        # Dynamic Proactive Mode: w_e(t) = (d_e / v_{e,free}) * C_hat_e(t)
        corridor_id = edge_data.get("corridor_id")
        if corridor_id and corridor_id in self.predictor.route_codes:
            # Predict dynamic congestion factor C_hat_e(t) >= 1.0
            c_factor = self.predictor.predict_congestion_factor(corridor_id, arrival_time)
        else:
            # Edge is not part of a monitored corridor: use minor street heuristic
            # or default moderate congestion factor based on time of day
            hour = arrival_time.hour
            is_peak = (8 <= hour <= 11) or (17 <= hour <= 21)
            c_factor = 1.25 if is_peak else 1.05

        w_e = t_free * max(1.0, c_factor)
        return w_e, d_e, t_free

    def _haversine_distance(self, u: Any, v: Any) -> float:
        """Compute great-circle distance between two nodes in meters."""
        u_data = self.graph.nodes[u]
        v_data = self.graph.nodes[v]

        lat1 = u_data.get("y", u_data.get("lat", 0.0))
        lon1 = u_data.get("x", u_data.get("lon", 0.0))
        lat2 = v_data.get("y", v_data.get("lat", 0.0))
        lon2 = v_data.get("x", v_data.get("lon", 0.0))

        # If cartesian coordinates are present without lat/lon
        if "lat" not in u_data and "y" in u_data and abs(lat1) > 90.0:
            return math.hypot(lat2 - lat1, lon2 - lon1)

        r = 6371000.0  # Earth radius in meters
        phi1, phi2 = math.radians(lat1), math.radians(lat2)
        dphi = math.radians(lat2 - lat1)
        dlambda = math.radians(lon2 - lon1)

        a = (
            math.sin(dphi / 2.0) ** 2
            + math.cos(phi1) * math.cos(phi2) * math.sin(dlambda / 2.0) ** 2
        )
        c = 2.0 * math.atan2(math.sqrt(a), math.sqrt(1.0 - a))
        return r * c

    def _admissible_heuristic(self, node: Any, target: Any) -> float:
        """Admissible and consistent A* heuristic: h(u) = distance(u, target) / v_max."""
        dist_m = self._haversine_distance(node, target)
        return dist_m / self.max_speed_ms

    def find_shortest_path(
        self,
        origin: Any,
        destination: Any,
        departure_time: datetime,
        mode: RoutingMode = RoutingMode.DYNAMIC_PROACTIVE,
        use_astar: bool = True,
        v2i_preemption: bool = True,
    ) -> Optional[RouteResult]:
        """Compute time-dependent shortest path using Dijkstra or A*.

        Args:
            origin: Source node ID.
            destination: Target node ID.
            departure_time: Departure timestamp.
            mode: RoutingMode (STATIC_FREE_FLOW or DYNAMIC_PROACTIVE).
            use_astar: If True, uses A* with admissible Euclidean heuristic; else Dijkstra.
            v2i_preemption: If True, engages V2I Green Wave preemption (0s signal delay).
                            If False, accumulates expected red-phase signal delay.

        Returns:
            RouteResult object with full route details, or None if unreachable.
        """
        if origin not in self.graph:
            raise ValueError(f"Origin node {origin} not in graph")
        if destination not in self.graph:
            raise ValueError(f"Destination node {destination} not in graph")

        if origin == destination:
            return RouteResult(
                mode=mode,
                origin_node=origin,
                destination_node=destination,
                departure_time=departure_time,
                arrival_time=departure_time,
                travel_time_seconds=0.0,
                free_flow_time_seconds=0.0,
                length_meters=0.0,
                average_speed_kmh=0.0,
                congestion_delay_seconds=0.0,
                path_nodes=[origin],
                path_edges=[],
                corridors_traversed=[],
                intersections_count=0,
                signal_delay_seconds=0.0,
                v2i_preemption_active=v2i_preemption,
                v2i_time_saved_seconds=0.0,
                intersections_detail=[],
            )

        # Priority queue stores tuples: (priority, current_time_s, current_node)
        # where current_time_s is elapsed traversal time from origin
        dist: Dict[Any, float] = {origin: 0.0}
        pred: Dict[Any, Tuple[Any, int, Dict[str, Any]]] = {}  # v -> (u, key, edge_data)

        # Counter for tie-breaking in priority queue
        counter = 0
        h0 = self._admissible_heuristic(origin, destination) if use_astar else 0.0
        pq: List[Tuple[float, int, float, Any]] = [(h0, counter, 0.0, origin)]
        visited: Set[Any] = set()

        while pq:
            f_score, _, elapsed_s, u = heapq.heappop(pq)

            if u in visited:
                continue
            visited.add(u)

            if u == destination:
                break

            current_arrival = departure_time + timedelta(seconds=elapsed_s)

            # Traverse outgoing edges
            # Supports both DiGraph and MultiDiGraph
            out_edges = self.graph[u]
            for v, edge_info in out_edges.items():
                if v in visited:
                    continue

                # If MultiDiGraph, edge_info is dict of keys -> data dict
                # If DiGraph, edge_info is data dict
                if isinstance(list(edge_info.keys())[0], int) and isinstance(
                    list(edge_info.values())[0], dict
                ):
                    candidates = [(k, d) for k, d in edge_info.items()]
                else:
                    candidates = [(0, edge_info)]

                for key, data in candidates:
                    w_e, d_e, _ = self.compute_edge_weight(
                        u, v, data, current_arrival, mode
                    )
                    new_elapsed = elapsed_s + w_e

                    if v not in dist or new_elapsed < dist[v]:
                        dist[v] = new_elapsed
                        pred[v] = (u, key, data)
                        h = (
                            self._admissible_heuristic(v, destination)
                            if use_astar
                            else 0.0
                        )
                        counter += 1
                        heapq.heappush(pq, (new_elapsed + h, counter, new_elapsed, v))

        if destination not in dist:
            logger.warning(
                "No path exists between node %s and node %s", origin, destination
            )
            return None

        # Reconstruct path
        path_nodes: List[Any] = [destination]
        path_edges: List[Tuple[Any, Any, int]] = []
        curr = destination
        total_length_m = 0.0
        total_free_flow_s = 0.0
        corridors: List[str] = []

        while curr != origin:
            u_prev, key, edge_data = pred[curr]
            path_nodes.append(u_prev)
            path_edges.append((u_prev, curr, key))

            d_e = float(edge_data.get("length_m", edge_data.get("length", 100.0)))
            v_free = float(edge_data.get("free_flow_speed_kmh", 40.0))
            total_length_m += d_e
            total_free_flow_s += d_e / (v_free / 3.6)

            cid = edge_data.get("corridor_id")
            if cid and cid not in corridors:
                corridors.append(cid)

            curr = u_prev

        path_nodes.reverse()
        path_edges.reverse()

        total_time_s = dist[destination]
        arrival_time = departure_time + timedelta(seconds=total_time_s)
        congestion_delay_s = max(0.0, total_time_s - total_free_flow_s)

        # Extract coordinates if available
        coords: List[Tuple[float, float]] = []
        for n in path_nodes:
            nd = self.graph.nodes[n]
            lat = nd.get("y", nd.get("lat"))
            lon = nd.get("x", nd.get("lon"))
            if lat is not None and lon is not None:
                coords.append((float(lat), float(lon)))

        # V2I Green Wave preemption evaluation along path coordinates
        v2i_metrics = self.evaluate_v2i_green_wave(coords, v2i_enabled=v2i_preemption)
        signal_delay_s = v2i_metrics["total_signal_delay_seconds"]
        v2i_time_saved_s = v2i_metrics["time_saved_seconds"]

        # When V2I is disabled, add accumulated red-light wait times
        final_total_time_s = total_time_s + signal_delay_s
        final_arrival_time = departure_time + timedelta(seconds=final_total_time_s)
        final_congestion_delay_s = max(0.0, congestion_delay_s + signal_delay_s)
        final_avg_speed_kmh = (
            (total_length_m / final_total_time_s) * 3.6 if final_total_time_s > 0 else 0.0
        )

        return RouteResult(
            mode=mode,
            origin_node=origin,
            destination_node=destination,
            departure_time=departure_time,
            arrival_time=final_arrival_time,
            travel_time_seconds=round(final_total_time_s, 2),
            free_flow_time_seconds=round(total_free_flow_s, 2),
            length_meters=round(total_length_m, 2),
            average_speed_kmh=round(final_avg_speed_kmh, 2),
            congestion_delay_seconds=round(final_congestion_delay_s, 2),
            path_nodes=path_nodes,
            path_edges=path_edges,
            corridors_traversed=corridors,
            node_coords=coords,
            intersections_count=v2i_metrics["signals_count"],
            signal_delay_seconds=round(signal_delay_s, 2),
            v2i_preemption_active=v2i_preemption,
            v2i_time_saved_seconds=round(v2i_time_saved_s, 2),
            intersections_detail=v2i_metrics["traversed_signals"],
        )

    def find_best_hospital(
        self,
        incident_node: Any,
        hospitals: List[Dict[str, Any]],
        departure_time: datetime,
        mode: RoutingMode = RoutingMode.DYNAMIC_PROACTIVE,
        require_capacity: bool = True,
        v2i_preemption: bool = True,
    ) -> Optional[Tuple[Dict[str, Any], RouteResult]]:
        """Evaluate candidate hospitals and return the optimal facility by travel time.

        Args:
            incident_node: Location of emergency incident S.
            hospitals: List of hospital dicts with keys 'node_id', 'name', 'beds_available', etc.
            departure_time: Time of ambulance dispatch.
            mode: Routing mode.
            require_capacity: If True, filters out hospitals marked with beds_available <= 0.
            v2i_preemption: If True, enables V2I Green Wave traffic signal preemption.

        Returns:
            Tuple of (chosen_hospital_dict, RouteResult), or None if none reachable.
        """
        best_hospital: Optional[Dict[str, Any]] = None
        best_result: Optional[RouteResult] = None
        min_travel_time = float("inf")

        for hosp in hospitals:
            hosp_node = hosp.get("node_id", hosp.get("id"))
            if hosp_node is None or hosp_node not in self.graph:
                continue

            if require_capacity and hosp.get("beds_available", 1) <= 0:
                logger.info(
                    "Skipping hospital %s (%s) due to zero bed availability",
                    hosp.get("name"),
                    hosp_node,
                )
                continue

            res = self.find_shortest_path(
                origin=incident_node,
                destination=hosp_node,
                departure_time=departure_time,
                mode=mode,
                use_astar=True,
                v2i_preemption=v2i_preemption,
            )

            if res is not None and res.travel_time_seconds < min_travel_time:
                min_travel_time = res.travel_time_seconds
                best_travel_time = res.travel_time_seconds
                best_hospital = hosp
                best_result = res

        if best_hospital is not None and best_result is not None:
            return best_hospital, best_result
        return None
