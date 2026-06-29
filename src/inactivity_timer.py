"""
Per-track inactivity timer — same pattern as FlockMetric mortality detection,
extended with zone-aware thresholds.
"""
from dataclasses import dataclass, field
from config import INACTIVITY_MOTION_THRESHOLD_PX, INACTIVITY_THRESHOLDS


@dataclass
class InactivityTimer:
    track_id: int
    _still_frames: int = field(default=0, repr=False)
    _last_centroid: tuple[float, float] | None = field(default=None, repr=False)
    alert_fired: bool = False

    def update(
        self,
        centroid: tuple[float, float] | None,
        zone: str = "default",
    ) -> bool:
        """
        Feed current centroid. Returns True when inactivity alert fires.
        zone: key matching INACTIVITY_THRESHOLDS (floor / chair / bed / default).
        """
        if centroid is None:
            return False

        if self._last_centroid is not None:
            dx = centroid[0] - self._last_centroid[0]
            dy = centroid[1] - self._last_centroid[1]
            dist = (dx ** 2 + dy ** 2) ** 0.5
            if dist < INACTIVITY_MOTION_THRESHOLD_PX:
                self._still_frames += 1
            else:
                self._still_frames = 0
                self.alert_fired = False   # movement resets
        self._last_centroid = centroid

        threshold = INACTIVITY_THRESHOLDS.get(zone, INACTIVITY_THRESHOLDS["default"])
        if self._still_frames >= threshold and not self.alert_fired:
            self.alert_fired = True
            return True
        return False

    def reset(self) -> None:
        self._still_frames = 0
        self.alert_fired = False
