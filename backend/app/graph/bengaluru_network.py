"""Bengaluru Road Network Builder for LifeRoute.

Constructs the urban road network of Bengaluru encompassing:
1. The 13 DataMeet monitored arterial corridors with physical GPS coordinates,
   measured distances, and model-predicted dynamic congestion factors.
2. Major arterial interchanges (Silk Board, Richmond Circle, Town Hall, Domlur, Hebbal, Marathahalli).
3. Primary multi-specialty hospitals in Bengaluru with coordinates, snapped nodes, and bed capacity.

Paper Notation & Formulation:
- Graph G = (V, E)
- d_e: length in meters (length_m)
- v_{e,free}: free-flow speed in km/h (free_flow_speed_kmh)
- t_{e,free} = d_e / (v_{e,free} / 3.6) [seconds]
- w_e(t) = (d_e / v_{e,free}) * C_hat_e(t) [seconds]
"""

from __future__ import annotations

import logging
import math
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import networkx as nx
import pandas as pd

logger = logging.getLogger(__name__)

# Real multi-specialty emergency hospitals in Bengaluru
BENGALURU_HOSPITALS: List[Dict[str, Any]] = [
    {
        "id": "HOSP_MANIPAL",
        "name": "Manipal Hospital (Old Airport Road)",
        "lat": 12.9576,
        "lon": 77.6534,
        "total_beds": 600,
        "beds_available": 14,
        "icu_available": 3,
        "trauma_center": True,
    },
    {
        "id": "HOSP_BOWRING",
        "name": "Bowring and Lady Curzon Hospital (Shivajinagar)",
        "lat": 12.9832,
        "lon": 77.6041,
        "total_beds": 450,
        "beds_available": 0,  # At capacity (tests capacity-aware dispatch)
        "icu_available": 0,
        "trauma_center": True,
    },
    {
        "id": "HOSP_VICTORIA",
        "name": "Victoria Hospital (K.R. Market / Fort)",
        "lat": 12.9634,
        "lon": 77.5746,
        "total_beds": 750,
        "beds_available": 22,
        "icu_available": 5,
        "trauma_center": True,
    },
    {
        "id": "HOSP_ST_MARTHAS",
        "name": "St. Martha's Hospital (Nrupathunga Road)",
        "lat": 12.9731,
        "lon": 77.5898,
        "total_beds": 350,
        "beds_available": 8,
        "icu_available": 2,
        "trauma_center": True,
    },
    {
        "id": "HOSP_ST_JOHNS",
        "name": "St. John's Medical College Hospital (Koramangala)",
        "lat": 12.9312,
        "lon": 77.6205,
        "total_beds": 800,
        "beds_available": 18,
        "icu_available": 4,
        "trauma_center": True,
    },
    {
        "id": "HOSP_CHINMAYA",
        "name": "Chinmaya Mission Hospital (Indiranagar)",
        "lat": 12.9774,
        "lon": 77.6441,
        "total_beds": 250,
        "beds_available": 5,
        "icu_available": 1,
        "trauma_center": False,
    },
    {
        "id": "HOSP_FORTIS",
        "name": "Fortis Hospital (Cunningham Road)",
        "lat": 12.9877,
        "lon": 77.5954,
        "total_beds": 150,
        "beds_available": 2,
        "icu_available": 1,
        "trauma_center": True,
    },
    {
        "id": "HOSP_APOLLO",
        "name": "Apollo Hospitals (Jayanagar / Bannerghatta)",
        "lat": 12.9165,
        "lon": 77.5968,
        "total_beds": 250,
        "beds_available": 11,
        "icu_available": 3,
        "trauma_center": True,
    },
    {
        "id": "HOSP_NIMHANS",
        "name": "NIMHANS Neuro-Emergency Center (Hosur Road)",
        "lat": 12.9392,
        "lon": 77.5956,
        "total_beds": 500,
        "beds_available": 6,
        "icu_available": 2,
        "trauma_center": True,
    },
    {
        "id": "HOSP_ASTER_CMI",
        "name": "Aster CMI Hospital (Hebbal Flyover)",
        "lat": 13.0573,
        "lon": 77.5915,
        "total_beds": 500,
        "beds_available": 16,
        "icu_available": 4,
        "trauma_center": True,
    },
    {
        "id": "HOSP_COLUMBIA",
        "name": "Manipal Hospital Whitefield (IT Corridor)",
        "lat": 12.9592,
        "lon": 77.7478,
        "total_beds": 300,
        "beds_available": 9,
        "icu_available": 2,
        "trauma_center": True,
    },
    {
        "id": "HOSP_KC_GENERAL",
        "name": "KC General Hospital (Malleswaram)",
        "lat": 12.9984,
        "lon": 77.5712,
        "total_beds": 350,
        "beds_available": 3,
        "icu_available": 1,
        "trauma_center": False,
    },
    {
        "id": "HOSP_BGS_GLENEAGLES",
        "name": "BGS Gleneagles Global Hospital (Kengeri)",
        "lat": 12.9056,
        "lon": 77.4982,
        "total_beds": 450,
        "beds_available": 12,
        "icu_available": 3,
        "trauma_center": True,
    },
    {
        "id": "HOSP_NARAYANA_HRUD",
        "name": "Narayana Institute of Cardiac Sciences (Electronic City)",
        "lat": 12.8252,
        "lon": 77.6914,
        "total_beds": 1000,
        "beds_available": 25,
        "icu_available": 6,
        "trauma_center": True,
    },
]


def haversine_m(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Calculate great-circle distance between two points in meters."""
    r = 6371000.0
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlambda = math.radians(lon2 - lon1)
    a = (
        math.sin(dphi / 2.0) ** 2
        + math.cos(phi1) * math.cos(phi2) * math.sin(dlambda / 2.0) ** 2
    )
    c = 2.0 * math.atan2(math.sqrt(a), math.sqrt(1.0 - a))
    return r * c


def build_bengaluru_network(
    routes_csv_path: Optional[Path] = None,
    free_flow_speeds: Optional[Dict[str, float]] = None,
) -> Tuple[nx.DiGraph, List[Dict[str, Any]]]:
    """Build strongly connected road network of Bengaluru with monitored corridors and hospitals.

    Returns:
        Tuple of (NetworkX DiGraph, hospitals_metadata_list).
    """
    G = nx.DiGraph()

    # Default fallback free-flow speeds if model metadata is not passed
    defaults_v_free = {
        "XJG4+7J|5PX4+HQ": 57.3,
        "XHJ7+MG|WJM6+VC": 37.4,
        "WHR9+R6|XJGF+6J": 31.9,
        "WH5F+26|WJ8X+F5W": 43.3,
        "XJPW+92|WJP4+FF": 37.9,
        "XMW9+G8|WMJR+V4": 45.7,
        "VJRQ+2M|RMJJ+F4": 44.5,
        "WGG8+G5|XH7P+G6": 37.1,
        "2HM2+P8|XJV5+RG": 35.3,
        "2HVW+G8|XJXR+WG": 38.8,
        "XPC7+72|XM33+J3": 40.7,
        "WP44+W8|WJFH+XQ": 35.6,
        "XG48+2W|VHWC+C3": 35.1,
    }
    v_free_map = free_flow_speeds or defaults_v_free

    # Key network nodes: intersections, landmarks, and corridor waypoints
    nodes: Dict[str, Tuple[float, float, str]] = {
        # Central Hubs
        "N_MG_ROAD": (12.9757, 77.6066, "MG Road Metro / Trinity"),
        "N_RICHMOND_CIRCLE": (12.9665, 77.5992, "Richmond Circle"),
        "N_TOWN_HALL": (12.9638, 77.5856, "Town Hall / Hudson Circle"),
        "N_MAJESTIC": (12.9774, 77.5713, "Kempegowda Majestic Hub"),
        "N_LULU_MALL": (12.9817, 77.5638, "Lulu Mall / Rajajinagar"),
        "N_KORAMANGALA_NEXUS": (12.9347, 77.6111, "Nexus Mall Koramangala"),
        "N_BASAVANAGUDI": (12.9421, 77.5681, "Big Bull Temple Basavanagudi"),
        "N_HALASURU": (12.9756, 77.6241, "Someshwara Temple Halasuru"),
        "N_JP_NAGAR": (12.9076, 77.5731, "JP Nagar Metro Station"),
        "N_SILK_BOARD": (12.9175, 77.6233, "Central Silk Board Junction"),
        "N_HSR_LAYOUT": (12.9162, 77.6480, "HSR Layout Sector 1"),
        "N_INDIRANAGAR_SV": (12.9859, 77.6451, "Swami Vivekananda Metro"),
        "N_CHRIST_UNIV": (12.9362, 77.6062, "Christ University Hosur Rd"),
        "N_BENNIGANAHALLI": (12.9963, 77.6683, "Benniganahalli Metro"),
        "N_TECH_VILLAGE": (12.9322, 77.6903, "Embassy TechVillage ORR"),
        "N_KUDLU_GATE": (12.8901, 77.6392, "Kudlu Gate Hosur Road"),
        "N_BIOCON": (12.8312, 77.6803, "Biocon Campus Electronic City"),
        "N_RR_NAGAR": (12.9263, 77.5154, "Rajarajeshwari Nagar"),
        "N_JP_PARK": (13.0343, 77.5508, "JP Park Mathikere"),
        "N_COLES_PARK": (12.9946, 77.6088, "Coles Park Fraser Town"),
        "N_HEBBAL": (13.0438, 77.5958, "Hebbal Flyover Junction"),
        "N_SMVT": (12.9998, 77.6413, "SMVT Baiyappanahalli"),
        "N_BROOKFIELD": (12.9707, 77.7126, "Brookfield / Whitefield"),
        "N_WIND_TUNNEL": (12.9541, 77.6527, "Wind Tunnel Road / HAL"),
        "N_KARMELARAM": (12.9073, 77.7058, "Karmelaram Railway Station"),
        "N_ST_FRANCIS": (12.9249, 77.6294, "St. Francis College Koramangala"),
        "N_NLSIU": (12.9551, 77.5173, "NLSIU Nagarbhavi"),
        "N_YELACHENAHALLI": (12.8961, 77.5702, "Yelachenahalli Metro"),
        # Intermediate Interchanges & Connectors
        "N_DOMLUR": (12.9610, 77.6387, "Domlur Intermediate Flyover"),
        "N_MARATHAHALLI": (12.9560, 77.7011, "Marathahalli Bridge"),
        "N_SHIVAJINAGAR": (12.9860, 77.6030, "Shivajinagar Bus Station"),
        "N_LALBAGH": (12.9507, 77.5848, "Lalbagh Botanical Garden West"),
        "N_JAYANAGAR_4TH": (12.9298, 77.5832, "Jayanagar 4th Block"),
        "N_DAIRY_CIRCLE": (12.9380, 77.6000, "Dairy Circle Flyover"),
        "N_MALLESWARAM": (13.0030, 77.5700, "Malleswaram 8th Cross"),
        "N_KR_PURAM": (13.0010, 77.6950, "KR Puram Hanging Bridge"),
        "N_CUNNINGHAM": (12.9880, 77.5960, "Cunningham Road Junction"),
        "N_SHOOLAY": (12.9690, 77.6080, "Shoolay Circle / Hosur Rd Entry"),
        "N_KORAMANGALA_100FT": (12.9340, 77.6250, "Koramangala 100 Feet Rd"),
        "N_AIRPORT_KIAL": (13.1989, 77.7069, "Kempegowda Int Airport KIAL"),
    }

    # Add nodes to graph
    for node_id, (lat, lon, label) in nodes.items():
        G.add_node(
            node_id,
            lat=lat,
            lon=lon,
            y=lat,
            x=lon,
            label=label,
        )

    def add_edge_bidir(
        u: str,
        v: str,
        corridor_id: Optional[str] = None,
        corridor_name: Optional[str] = None,
        custom_speed: Optional[float] = None,
        highway: str = "primary",
    ) -> None:
        """Add bidirectional directed edges with accurate physical distance and speeds."""
        lat1, lon1, _ = nodes[u]
        lat2, lon2, _ = nodes[v]
        dist_m = max(50.0, haversine_m(lat1, lon1, lat2, lon2))

        # Speed limit
        if custom_speed is not None:
            v_free = custom_speed
        elif corridor_id and corridor_id in v_free_map:
            v_free = v_free_map[corridor_id]
        else:
            v_free = 45.0 if "trunk" in highway or "motorway" in highway else 35.0

        t_free = dist_m / (v_free / 3.6)

        edge_attrs = {
            "length_m": round(dist_m, 1),
            "length": round(dist_m, 1),
            "free_flow_speed_kmh": round(v_free, 1),
            "free_flow_time_s": round(t_free, 2),
            "highway": highway,
            "corridor_id": corridor_id,
            "corridor_name": corridor_name,
        }

        # Forward and backward
        G.add_edge(u, v, edge_id=f"{u}-{v}", **edge_attrs)
        G.add_edge(v, u, edge_id=f"{v}-{u}", **edge_attrs)

    # 1. MONITORED CORRIDOR SEGMENTS (DataMeet 13 routes)
    # Airport Expy: MG Road -> Hebbal -> KIAL
    add_edge_bidir("N_MG_ROAD", "N_HEBBAL", "XJG4+7J|5PX4+HQ", "Airport Expy", highway="trunk")
    add_edge_bidir("N_HEBBAL", "N_AIRPORT_KIAL", "XJG4+7J|5PX4+HQ", "Airport Expy", highway="motorway")

    # Central Diagonal 1: Lulu Mall -> Majestic -> Town Hall -> Dairy Circle -> Koramangala
    add_edge_bidir("N_LULU_MALL", "N_MAJESTIC", "XHJ7+MG|WJM6+VC", "Central Diagonal 1", highway="primary")
    add_edge_bidir("N_MAJESTIC", "N_TOWN_HALL", "XHJ7+MG|WJM6+VC", "Central Diagonal 1", highway="primary")
    add_edge_bidir("N_TOWN_HALL", "N_DAIRY_CIRCLE", "XHJ7+MG|WJM6+VC", "Central Diagonal 1", highway="primary")
    add_edge_bidir("N_DAIRY_CIRCLE", "N_KORAMANGALA_NEXUS", "XHJ7+MG|WJM6+VC", "Central Diagonal 1", highway="primary")

    # Central Diagonal 2: Basavanagudi -> Lalbagh -> Richmond Circle -> Halasuru
    add_edge_bidir("N_BASAVANAGUDI", "N_LALBAGH", "WHR9+R6|XJGF+6J", "Central Diagonal 2", highway="primary")
    add_edge_bidir("N_LALBAGH", "N_RICHMOND_CIRCLE", "WHR9+R6|XJGF+6J", "Central Diagonal 2", highway="primary")
    add_edge_bidir("N_RICHMOND_CIRCLE", "N_HALASURU", "WHR9+R6|XJGF+6J", "Central Diagonal 2", highway="primary")

    # Double Decker Flyover: JP Nagar -> Silk Board -> HSR Layout
    add_edge_bidir("N_JP_NAGAR", "N_SILK_BOARD", "WH5F+26|WJ8X+F5W", "Double Decker Flyover", highway="primary")
    add_edge_bidir("N_SILK_BOARD", "N_HSR_LAYOUT", "WH5F+26|WJ8X+F5W", "Double Decker Flyover", highway="trunk")

    # East Inner Ring: SV Road -> Domlur -> Koramangala 100ft -> Christ Univ
    add_edge_bidir("N_INDIRANAGAR_SV", "N_DOMLUR", "XJPW+92|WJP4+FF", "East Inner Ring", highway="primary")
    add_edge_bidir("N_DOMLUR", "N_KORAMANGALA_100FT", "XJPW+92|WJP4+FF", "East Inner Ring", highway="primary")
    add_edge_bidir("N_KORAMANGALA_100FT", "N_CHRIST_UNIV", "XJPW+92|WJP4+FF", "East Inner Ring", highway="primary")

    # East Outer Ring: Benniganahalli -> KR Puram -> Marathahalli -> Tech Village
    add_edge_bidir("N_BENNIGANAHALLI", "N_KR_PURAM", "XMW9+G8|WMJR+V4", "East Outer Ring", highway="trunk")
    add_edge_bidir("N_KR_PURAM", "N_MARATHAHALLI", "XMW9+G8|WMJR+V4", "East Outer Ring", highway="trunk")
    add_edge_bidir("N_MARATHAHALLI", "N_TECH_VILLAGE", "XMW9+G8|WMJR+V4", "East Outer Ring", highway="trunk")
    add_edge_bidir("N_TECH_VILLAGE", "N_SILK_BOARD", "XMW9+G8|WMJR+V4", "East Outer Ring (ORR South)", highway="trunk")

    # Hosur Road: Silk Board -> Kudlu Gate -> Biocon
    add_edge_bidir("N_SILK_BOARD", "N_KUDLU_GATE", "VJRQ+2M|RMJJ+F4", "Hosur Road", highway="trunk")
    add_edge_bidir("N_KUDLU_GATE", "N_BIOCON", "VJRQ+2M|RMJJ+F4", "Hosur Road", highway="trunk")

    # Mysore Road: RR Nagar -> Town Hall
    add_edge_bidir("N_RR_NAGAR", "N_TOWN_HALL", "WGG8+G5|XH7P+G6", "Mysore Road", highway="primary")

    # North Inner Ring: JP Park -> Malleswaram -> Coles Park
    add_edge_bidir("N_JP_PARK", "N_MALLESWARAM", "2HM2+P8|XJV5+RG", "North Inner Ring", highway="primary")
    add_edge_bidir("N_MALLESWARAM", "N_COLES_PARK", "2HM2+P8|XJV5+RG", "North Inner Ring", highway="primary")

    # North Outer Ring: Hebbal -> SMVT
    add_edge_bidir("N_HEBBAL", "N_SMVT", "2HVW+G8|XJXR+WG", "North Outer Ring", highway="primary")

    # Old Airport Road: Brookfield -> Marathahalli -> Wind Tunnel / Domlur
    add_edge_bidir("N_BROOKFIELD", "N_MARATHAHALLI", "XPC7+72|XM33+J3", "Old Airport Road", highway="primary")
    add_edge_bidir("N_MARATHAHALLI", "N_WIND_TUNNEL", "XPC7+72|XM33+J3", "Old Airport Road", highway="primary")
    add_edge_bidir("N_WIND_TUNNEL", "N_DOMLUR", "XPC7+72|XM33+J3", "Old Airport Road", highway="primary")

    # Sarjapur Road: Karmelaram -> St. Francis / Koramangala
    add_edge_bidir("N_KARMELARAM", "N_ST_FRANCIS", "WP44+W8|WJFH+XQ", "Sarjapur Road", highway="primary")
    add_edge_bidir("N_ST_FRANCIS", "N_KORAMANGALA_100FT", "WP44+W8|WJFH+XQ", "Sarjapur Road", highway="primary")

    # South Outer Ring: NLSIU -> JP Nagar -> Yelachenahalli
    add_edge_bidir("N_NLSIU", "N_JP_NAGAR", "XG48+2W|VHWC+C3", "South Outer Ring", highway="primary")
    add_edge_bidir("N_JP_NAGAR", "N_YELACHENAHALLI", "XG48+2W|VHWC+C3", "South Outer Ring", highway="primary")

    # 2. KEY ARTERIAL CROSS-CONNECTORS (ensuring multi-path options for router)
    # MG Road to Central Hubs
    add_edge_bidir("N_MG_ROAD", "N_RICHMOND_CIRCLE", highway="secondary", custom_speed=35.0)
    add_edge_bidir("N_MG_ROAD", "N_HALASURU", highway="secondary", custom_speed=40.0)
    add_edge_bidir("N_MG_ROAD", "N_SHIVAJINAGAR", highway="secondary", custom_speed=30.0)
    add_edge_bidir("N_MG_ROAD", "N_SHOOLAY", highway="secondary", custom_speed=35.0)
    add_edge_bidir("N_SHOOLAY", "N_DAIRY_CIRCLE", highway="primary", custom_speed=40.0)

    # Richmond Circle to Town Hall & Lalbagh
    add_edge_bidir("N_RICHMOND_CIRCLE", "N_TOWN_HALL", highway="primary", custom_speed=35.0)
    add_edge_bidir("N_RICHMOND_CIRCLE", "N_LALBAGH", highway="secondary", custom_speed=35.0)

    # Lalbagh to Jayanagar & Dairy Circle
    add_edge_bidir("N_LALBAGH", "N_JAYANAGAR_4TH", highway="primary", custom_speed=40.0)
    add_edge_bidir("N_JAYANAGAR_4TH", "N_JP_NAGAR", highway="primary", custom_speed=40.0)
    add_edge_bidir("N_JAYANAGAR_4TH", "N_DAIRY_CIRCLE", highway="secondary", custom_speed=35.0)

    # Dairy Circle to Christ Univ & Silk Board
    add_edge_bidir("N_DAIRY_CIRCLE", "N_CHRIST_UNIV", highway="primary", custom_speed=40.0)
    add_edge_bidir("N_CHRIST_UNIV", "N_SILK_BOARD", highway="trunk", custom_speed=50.0)

    # Koramangala Nexus to 100ft & Silk Board
    add_edge_bidir("N_KORAMANGALA_NEXUS", "N_KORAMANGALA_100FT", highway="secondary", custom_speed=30.0)
    add_edge_bidir("N_KORAMANGALA_NEXUS", "N_SILK_BOARD", highway="primary", custom_speed=40.0)

    # Domlur to Halasuru & Indiranagar
    add_edge_bidir("N_DOMLUR", "N_HALASURU", highway="primary", custom_speed=40.0)
    add_edge_bidir("N_HALASURU", "N_INDIRANAGAR_SV", highway="primary", custom_speed=35.0)
    add_edge_bidir("N_BENNIGANAHALLI", "N_INDIRANAGAR_SV", highway="primary", custom_speed=40.0)

    # Cunningham Road & Coles Park to Majestic / Shivajinagar
    add_edge_bidir("N_CUNNINGHAM", "N_SHIVAJINAGAR", highway="secondary", custom_speed=30.0)
    add_edge_bidir("N_CUNNINGHAM", "N_MAJESTIC", highway="primary", custom_speed=35.0)
    add_edge_bidir("N_CUNNINGHAM", "N_COLES_PARK", highway="secondary", custom_speed=30.0)
    add_edge_bidir("N_SHIVAJINAGAR", "N_COLES_PARK", highway="secondary", custom_speed=30.0)
    add_edge_bidir("N_MAJESTIC", "N_MALLESWARAM", highway="primary", custom_speed=35.0)

    # Hebbal to Coles Park & Malleswaram
    add_edge_bidir("N_HEBBAL", "N_COLES_PARK", highway="primary", custom_speed=45.0)
    add_edge_bidir("N_HEBBAL", "N_MALLESWARAM", highway="primary", custom_speed=45.0)

    # 3. SNAP HOSPITALS TO NEAREST NETWORK NODES
    hospital_data: List[Dict[str, Any]] = []
    for hosp in BENGALURU_HOSPITALS:
        h_lat, h_lon = hosp["lat"], hosp["lon"]
        # Find closest graph node
        best_node = None
        min_d = float("inf")
        for n, (n_lat, n_lon, _) in nodes.items():
            d = haversine_m(h_lat, h_lon, n_lat, n_lon)
            if d < min_d:
                min_d = d
                best_node = n

        hosp_record = dict(hosp)
        hosp_record["node_id"] = best_node
        hosp_record["snap_distance_m"] = round(min_d, 1)
        hospital_data.append(hosp_record)

    logger.info(
        "Constructed Bengaluru road network: %d nodes, %d directed edges, %d hospitals",
        G.number_of_nodes(),
        G.number_of_edges(),
        len(hospital_data),
    )
    return G, hospital_data
