"""
Per-track gait speed estimator.
Uses ankle keypoint displacement over a rolling window to estimate
pixels-per-frame walking speed. Flags declining speed trend.
"""
from collections import deque
from dataclasses import dataclass, field
import numpy as np
from pose_utils import _kp, L_ANKLE, R_ANKLE, MIN_CONF


@dataclass
class GaitTracker:
    track_id: int
    pixels_per_meter: float = 100.0   # calibrate per camera; rough default
    _ankle_history: deque = field(default_factory=lambda: deque(maxlen=90))  # 3s at 30fps
    _speed_history: deque = field(default_factory=lambda: deque(maxlen=300)) # 10s

    def update(self, keypoints: np.ndarray) -> float | None:
        """Returns smoothed gait speed in m/s, or None if insufficient data."""
        la = _kp(keypoints, L_ANKLE)
        ra = _kp(keypoints, R_ANKLE)

        if max(la[2], ra[2]) < MIN_CONF:
            return None

        # Use whichever ankle is more confident
        ankle = la if la[2] >= ra[2] else ra
        self._ankle_history.append((ankle[0], ankle[1]))

        if len(self._ankle_history) < 10:
            return None

        # Displacement over last 10 frames
        pts = list(self._ankle_history)
        displacements = [
            ((pts[i][0]-pts[i-1][0])**2 + (pts[i][1]-pts[i-1][1])**2)**0.5
            for i in range(1, len(pts))
        ]
        px_per_frame = np.mean(displacements[-10:])
        speed_ms = px_per_frame * 30 / self.pixels_per_meter   # assuming 30fps

        self._speed_history.append(speed_ms)
        return float(np.mean(list(self._speed_history)[-30:]))   # 1s smoothed

    def is_declining(self) -> bool:
        """True if gait speed has dropped >30% over the last 300 frames."""
        if len(self._speed_history) < 60:
            return False
        early = np.mean(list(self._speed_history)[:30])
        recent = np.mean(list(self._speed_history)[-30:])
        if early == 0:
            return False
        return (early - recent) / early > 0.30
