"""
Camera zone management.
Zones are polygons defined per camera in data/zones/zones.json.
Each zone has a name, polygon vertices (normalised 0-1), and a type.

Example zones.json:
{
  "cam_01": [
    {"name": "bed",         "type": "bed",        "poly": [[0.1,0.2],[0.5,0.2],[0.5,0.7],[0.1,0.7]]},
    {"name": "restricted",  "type": "restricted",  "poly": [[0.8,0.0],[1.0,0.0],[1.0,0.3],[0.8,0.3]]}
  ]
}
"""
import json
import numpy as np
from pathlib import Path
from config import RESTRICTED_ZONE_FILE


def _point_in_poly(x: float, y: float, poly: list[list[float]]) -> bool:
    """Ray-casting algorithm on normalised coords."""
    n = len(poly)
    inside = False
    j = n - 1
    for i in range(n):
        xi, yi = poly[i]
        xj, yj = poly[j]
        if ((yi > y) != (yj > y)) and (x < (xj - xi) * (y - yi) / (yj - yi) + xi):
            inside = not inside
        j = i
    return inside


class ZoneManager:
    def __init__(self, camera_id: str):
        self.camera_id = camera_id
        self.zones: list[dict] = []
        if RESTRICTED_ZONE_FILE.exists():
            data = json.loads(RESTRICTED_ZONE_FILE.read_text())
            self.zones = data.get(camera_id, [])

    def zone_for_point(
        self,
        x_norm: float,
        y_norm: float,
    ) -> str:
        """Return zone type for a normalised (x, y) coordinate."""
        for zone in self.zones:
            if _point_in_poly(x_norm, y_norm, zone["poly"]):
                return zone.get("type", "default")
        return "default"

    def is_restricted(self, x_norm: float, y_norm: float) -> bool:
        for zone in self.zones:
            if zone.get("type") == "restricted":
                if _point_in_poly(x_norm, y_norm, zone["poly"]):
                    return True
        return False
