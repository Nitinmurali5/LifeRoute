"""LifeRoute graph representation package.

Implements graph construction for:
1. Synthetic road grid (Table 4 scenario): grid_graph.py
2. Real-world OSMnx city road network: city_graph.py
3. Hospital snapping and simulated bed capacity: hospitals.py
"""

from backend.app.graph.grid_graph import build_synthetic_grid_graph
from backend.app.graph.city_graph import build_city_graph
from backend.app.graph.hospitals import extract_and_assign_hospitals

__all__ = [
    "build_synthetic_grid_graph",
    "build_city_graph",
    "extract_and_assign_hospitals",
]

