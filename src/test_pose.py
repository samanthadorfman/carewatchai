"""
Step 1 — Pose sanity check: run YOLO-pose on a video, write annotated output.
Usage:
    python src/test_pose.py --source "path/to/video.mp4"
Output:
    data/sample_videos/pose_test_output.mp4
"""
import argparse
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

import cv2
import numpy as np
from ultralytics import YOLO
from config import POSE_MODEL, CONF_THRESHOLD
from pose_utils import body_axis_angle, hip_y_fraction, centroid_from_keypoints
from overlay import draw_skeleton

OUTPUT_DIR = Path(__file__).parent.parent / "data" / "sample_videos"


def test(source: str | int) -> None:
    model = YOLO(POSE_MODEL)
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open: {source}")

    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 24.0
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    input_stem = Path(source).stem if isinstance(source, str) else "webcam"
    output_path = OUTPUT_DIR / f"{input_stem}_pose.mp4"
    writer = cv2.VideoWriter(
        str(output_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (w, h),
    )

    frame_n = 0
    print(f"Processing {total} frames...")

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frame_n += 1

        results = model(frame, conf=CONF_THRESHOLD, verbose=False)
        out = frame.copy()

        n_people = 0
        if results[0].keypoints is not None:
            kps_all = results[0].keypoints.data.cpu().numpy()
            boxes = results[0].boxes.xyxy.cpu().numpy()
            n_people = len(kps_all)

            for i, kps in enumerate(kps_all):
                angle = body_axis_angle(kps)
                hip_y = hip_y_fraction(kps, h)

                color = (0, 200, 0)
                if angle is not None and angle > 45 and hip_y is not None and hip_y > 0.75:
                    color = (0, 0, 255)   # fall geometry detected
                elif angle is not None and angle > 45:
                    color = (0, 200, 255) # suspicious angle only

                draw_skeleton(out, kps, color)

                if i < len(boxes):
                    x1, y1 = int(boxes[i][0]), int(boxes[i][1])
                    x2, y2 = int(boxes[i][2]), int(boxes[i][3])
                    cv2.rectangle(out, (x1, y1), (x2, y2), color, 2)
                    angle_str = f"angle={angle:.1f}" if angle is not None else "angle=N/A"
                    hip_str   = f"hip_y={hip_y:.2f}" if hip_y is not None else "hip_y=N/A"
                    cv2.putText(out, angle_str, (x1, y1 - 18),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
                    cv2.putText(out, hip_str,   (x1, y1 - 4),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

        # HUD
        cv2.putText(out, f"Frame {frame_n}/{total} | People: {n_people}",
                    (8, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255,255,255), 2)

        # Legend
        for j, (label, col) in enumerate([
            ("Normal", (0,200,0)),
            ("Suspicious angle", (0,200,255)),
            ("Fall geometry", (0,0,255)),
        ]):
            cv2.putText(out, label, (8, h - 20 - j*20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, col, 1)

        writer.write(out)
        if frame_n % 30 == 0:
            print(f"  Frame {frame_n}/{total}")

    cap.release()
    writer.release()
    print(f"\nDone. Output saved to:\n  {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", required=True)
    args = parser.parse_args()
    source = int(args.source) if args.source.isdigit() else args.source
    test(source)
