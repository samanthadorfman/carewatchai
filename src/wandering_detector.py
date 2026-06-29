"""
Wandering detector — flags repeated zone crossing during restricted hours.
A resident is considered wandering if they cross between zones more than
CROSSING_THRESHOLD times within WINDOW_FRAMES frames during nighttime hours.
"""
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime
from config import RESTRICTED_HOURS


@dataclass
class WanderingDetector:
    track_id: int
    _zone_history: deque = field(default_factory=lambda: deque(maxlen=300))
    _crossing_count: int = field(default=0, repr=False)
    alert_fired: bool = False

    CROSSING_THRESHOLD: int = 4   # zone changes in window before alert
    RESET_AFTER_ALERT_FRAMES: int = 900  # 30s before can re-alert

    def update(self, zone: str) -> bool:
        if not self._is_nighttime():
            return False

        prev = self._zone_history[-1] if self._zone_history else None
        self._zone_history.append(zone)

        if prev is not None and zone != prev:
            self._crossing_count += 1

        if self._crossing_count >= self.CROSSING_THRESHOLD and not self.alert_fired:
            self.alert_fired = True
            return True

        return False

    def reset(self) -> None:
        self._crossing_count = 0
        self.alert_fired = False

    @staticmethod
    def _is_nighttime() -> bool:
        hour = datetime.now().hour
        start_h, end_h = RESTRICTED_HOURS
        if start_h > end_h:
            return hour >= start_h or hour < end_h
        return start_h <= hour < end_h
