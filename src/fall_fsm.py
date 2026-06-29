"""
Per-track fall detection finite state machine.

States:
  UPRIGHT  → normal
  SUSPECT  → angle/proximity thresholds crossed, counting down
  FALLEN   → confirmed fall, alert fired
  STANDING → recovery transition (counting up before returning to UPRIGHT)
"""
from enum import Enum, auto
from dataclasses import dataclass, field

from config import (
    FALL_ANGLE_THRESHOLD,
    FLOOR_PROXIMITY_FRACTION,
    FALL_CONFIRM_FRAMES,
    FALL_RESET_FRAMES,
)


class FallState(Enum):
    UPRIGHT  = auto()
    SUSPECT  = auto()
    FALLEN   = auto()
    STANDING = auto()   # recovering


@dataclass
class FallFSM:
    track_id: int
    state: FallState = FallState.UPRIGHT
    _suspect_frames: int = field(default=0, repr=False)
    _standing_frames: int = field(default=0, repr=False)
    alert_fired: bool = False

    def update(
        self,
        angle: float | None,
        hip_y_frac: float | None,
    ) -> tuple[FallState, bool]:
        """
        Feed latest pose features.
        Returns (new_state, alert_just_fired).
        """
        alert_just_fired = False
        is_down = self._is_down_pose(angle, hip_y_frac)

        match self.state:
            case FallState.UPRIGHT | FallState.STANDING:
                if is_down:
                    self.state = FallState.SUSPECT
                    self._suspect_frames = 1
                    self._standing_frames = 0
                else:
                    self.state = FallState.UPRIGHT
                    self._standing_frames = 0

            case FallState.SUSPECT:
                if is_down:
                    self._suspect_frames += 1
                    if self._suspect_frames >= FALL_CONFIRM_FRAMES and not self.alert_fired:
                        self.state = FallState.FALLEN
                        self.alert_fired = True
                        alert_just_fired = True
                else:
                    # Person got back up before confirmation
                    self.state = FallState.UPRIGHT
                    self._suspect_frames = 0

            case FallState.FALLEN:
                if not is_down:
                    self.state = FallState.STANDING
                    self._standing_frames = 1
                    self.alert_fired = False   # reset so a second fall can alert

            case FallState.STANDING:
                if not is_down:
                    self._standing_frames += 1
                    if self._standing_frames >= FALL_RESET_FRAMES:
                        self.state = FallState.UPRIGHT
                        self._standing_frames = 0
                else:
                    self.state = FallState.FALLEN
                    self._standing_frames = 0

        return self.state, alert_just_fired

    @staticmethod
    def _is_down_pose(angle: float | None, hip_y_frac: float | None) -> bool:
        if angle is None or hip_y_frac is None:
            return False
        return (
            angle > FALL_ANGLE_THRESHOLD
            and hip_y_frac > FLOOR_PROXIMITY_FRACTION
        )
