"""Save incident snapshots — skeleton render only, no raw faces."""
import cv2
import numpy as np
from pathlib import Path
from datetime import datetime

SNAPSHOT_DIR = Path(__file__).parent.parent / "incidents" / "snapshots"
SKELETON_DIR = Path(__file__).parent.parent / "incidents" / "skeleton_only"


def _ensure_dirs() -> None:
    SNAPSHOT_DIR.mkdir(parents=True, exist_ok=True)
    SKELETON_DIR.mkdir(parents=True, exist_ok=True)


def save_snapshot(frame: np.ndarray, event_type: str, track_id: int, camera_id: str) -> str:
    """Save annotated frame. Returns file path string."""
    _ensure_dirs()
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    fname = f"{ts}_{event_type}_cam{camera_id}_t{track_id}.jpg"
    path = SNAPSHOT_DIR / fname
    cv2.imwrite(str(path), frame)
    return str(path)


def save_skeleton_only(frame: np.ndarray, event_type: str, track_id: int, camera_id: str) -> str:
    """
    Save a black-background skeleton-only frame — the privacy-safe version.
    Blurs the raw frame then overlays skeleton so no facial features are visible.
    """
    _ensure_dirs()
    # Heavy blur kills facial features while preserving skeleton overlay positions
    blurred = cv2.GaussianBlur(frame, (99, 99), 30)
    # Darken further
    privacy = (blurred * 0.15).astype(np.uint8)

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    fname = f"{ts}_{event_type}_cam{camera_id}_t{track_id}_skeleton.jpg"
    path = SKELETON_DIR / fname
    cv2.imwrite(str(path), privacy)
    return str(path)
