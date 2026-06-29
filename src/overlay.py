"""
Overlay rendering — color-coded boxes, skeleton, alert banner, status HUD.

Box colors:
  Green  = UPRIGHT / normal
  Yellow = SUSPECT fall / inactivity warning
  Red    = FALLEN confirmed
  Blue   = Wandering
  Orange = Restricted zone breach
  Purple = Gait decline warning
"""
import cv2
import numpy as np
from ultralytics.engine.results import Results
from fall_fsm import FallState

SKELETON_PAIRS = [
    (5, 6), (5, 7), (7, 9), (6, 8), (8, 10),
    (5, 11), (6, 12), (11, 12),
    (11, 13), (13, 15), (12, 14), (14, 16),
    (0, 5), (0, 6),
]

MIN_KP_CONF = 0.3

# Event type → box color (BGR)
EVENT_COLORS = {
    "normal":          (0, 200, 0),
    "suspect":         (0, 200, 255),
    "fall":            (0, 0, 255),
    "inactivity":      (0, 165, 255),
    "wandering":       (255, 100, 0),
    "restricted_zone": (0, 100, 255),
    "gait_decline":    (180, 0, 180),
}

ALERT_COLORS = {
    "fall":            (0, 0, 200),
    "inactivity":      (0, 130, 200),
    "wandering":       (200, 80, 0),
    "restricted_zone": (0, 80, 220),
}


def draw_skeleton(frame: np.ndarray, keypoints: np.ndarray, color: tuple) -> None:
    for a, b in SKELETON_PAIRS:
        if a >= len(keypoints) or b >= len(keypoints):
            continue
        kp_a, kp_b = keypoints[a], keypoints[b]
        if kp_a[2] < MIN_KP_CONF or kp_b[2] < MIN_KP_CONF:
            continue
        cv2.line(frame,
                 (int(kp_a[0]), int(kp_a[1])),
                 (int(kp_b[0]), int(kp_b[1])),
                 color, 2, cv2.LINE_AA)
    for kp in keypoints:
        if kp[2] < MIN_KP_CONF:
            continue
        cv2.circle(frame, (int(kp[0]), int(kp[1])), 4, color, -1, cv2.LINE_AA)


def _box_color(fall_state: FallState, flags: dict) -> tuple:
    if flags.get("fall"):
        return EVENT_COLORS["fall"]
    if flags.get("restricted_zone"):
        return EVENT_COLORS["restricted_zone"]
    if flags.get("wandering"):
        return EVENT_COLORS["wandering"]
    if flags.get("inactivity"):
        return EVENT_COLORS["inactivity"]
    if flags.get("gait_decline"):
        return EVENT_COLORS["gait_decline"]
    if fall_state == FallState.SUSPECT:
        return EVENT_COLORS["suspect"]
    return EVENT_COLORS["normal"]


def draw_frame(
    frame: np.ndarray,
    results: Results,
    track_data: dict[int, dict],   # track_id → {fall_state, zone, flags, gait_speed}
    active_alerts: list[dict],     # [{type, track_id, camera_id, zone}]
) -> np.ndarray:
    out = frame.copy()
    h, w = out.shape[:2]

    if results.boxes.id is not None:
        ids = results.boxes.id.int().cpu().tolist()
        boxes = results.boxes.xyxy.cpu().numpy()
        kps_list = results.keypoints.data.cpu().numpy() if results.keypoints else []

        for i, tid in enumerate(ids):
            data = track_data.get(tid, {})
            fall_state = data.get("fall_state", FallState.UPRIGHT)
            zone = data.get("zone", "default")
            flags = data.get("flags", {})
            gait_speed = data.get("gait_speed")

            color = _box_color(fall_state, flags)
            x1, y1, x2, y2 = map(int, boxes[i])
            cv2.rectangle(out, (x1, y1), (x2, y2), color, 2)

            # Label
            status = fall_state.name
            if flags.get("fall"):           status = "FALL DETECTED"
            elif flags.get("inactivity"):   status = "INACTIVITY"
            elif flags.get("wandering"):    status = "WANDERING"
            elif flags.get("restricted_zone"): status = "RESTRICTED ZONE"

            label = f"ID {tid} | {status} | {zone}"
            cv2.putText(out, label, (x1, max(y1 - 8, 12)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.50, color, 1, cv2.LINE_AA)

            if gait_speed is not None:
                spd_label = f"gait {gait_speed:.2f} m/s"
                cv2.putText(out, spd_label, (x1, max(y1 - 22, 26)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.40, color, 1, cv2.LINE_AA)

            if i < len(kps_list):
                draw_skeleton(out, kps_list[i], color)

    # ── Alert banner ────────────────────────────────────────────────────────
    if active_alerts:
        banner_h = 32 * len(active_alerts) + 12
        overlay = out.copy()
        cv2.rectangle(overlay, (0, 0), (w, banner_h), (10, 10, 10), -1)
        cv2.addWeighted(overlay, 0.75, out, 0.25, 0, out)

        for j, alert in enumerate(active_alerts[-4:]):
            atype = alert.get("type", "fall")
            color = ALERT_COLORS.get(atype, (0, 0, 200))
            icons = {"fall": "🔴 FALL", "inactivity": "🟡 INACTIVITY",
                     "wandering": "🔵 WANDERING", "restricted_zone": "🟠 ZONE BREACH"}
            icon = icons.get(atype, atype.upper())
            msg = f"{icon}  Track {alert['track_id']} | {alert['camera_id']} | Zone: {alert['zone']}"
            cv2.putText(out, msg, (10, 26 + j * 32),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.65, color, 2, cv2.LINE_AA)

    # ── HUD ─────────────────────────────────────────────────────────────────
    n_tracks = len(track_data)
    cv2.putText(out, f"CareWatch AI | Tracks: {n_tracks}",
                (8, h - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.50, (200, 200, 200), 1)

    return out


def render_skeleton_only(frame: np.ndarray, results: Results) -> np.ndarray:
    """Privacy mode: black background with skeleton overlay only."""
    out = np.zeros_like(frame)
    if results.keypoints is not None:
        kps_list = results.keypoints.data.cpu().numpy()
        for kps in kps_list:
            draw_skeleton(out, kps, (0, 220, 0))
    return out
