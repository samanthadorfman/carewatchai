# CareWatch AI

Real-time overhead AI monitoring for senior living facilities. Detects falls, prolonged inactivity, nighttime wandering, and restricted zone breaches using existing IP cameras.

**No wearables. No cloud. No video leaves the building.**

## What it does

- 🔴 **Fall detection** — body axis angle + hip position, confirmed over 1.5 seconds
- 🟡 **Prolonged inactivity** — zone-aware stillness timer (floor: 10s, chair: 5min, bed: 10min)
- 🔵 **Nighttime wandering** — motion detection between 10pm–6am
- 🟠 **Restricted zone breach** — polygon zones per camera, time-gated

## Development 


```bash
source venv/bin/activate
pip install -r requirements.txt
export KAGGLE_API_TOKEN=your_token
python scripts/run_eval.py
```

### Pushing changes

```bash
git add .
git commit -m "describe what you changed"
git push
```

Then open a pull request on GitHub to merge into the main repo.

> **Tip:** Suspend your Codespace (just close the tab) rather than deleting it. Videos persist for up to 30 days, saving you the re-download.

---

## Stack

- YOLO26-pose / YOLOv11-pose — 17-point skeleton keypoints
- ByteTrack — multi-person tracking
- OpenCV — video processing
- Streamlit — staff dashboard + marketing site
- SQLite — incident logging

## Run locally

```bash
pip install -r requirements.txt
python src/pipeline.py --source your_video.mp4 --camera cam_01
python -m streamlit run dashboard/app.py --server.port 8501
python -m streamlit run website/app.py --server.port 8502
```

## Project structure
# src/

- pipeline.py           — main inference loop
- fall_fsm.py           — fall detection state machine
- inactivity_timer.py   — stillness detection
- wandering_detector.py — nighttime wandering
- zone_manager.py       — polygon zone logic
- pose_utils.py         — keypoint math
- gait_tracker.py       — walking speed estimation
- overlay.py            — visualization
- incident_log.py       — SQLite logging
- evaluate.py           — Le2i dataset evaluator
  
# scripts/
- run_eval.py             — download Le2i + run evaluation
  
# dashboard/
- app.py                — staff alert dashboard

# website/
- app.py                — marketing site

# data/
- zones/zones.json      — camera zone configuration
- videos/               — gitignored, download via run_eval.py

## Live demo
[carewatchai.com](https://carewatchai.com)
