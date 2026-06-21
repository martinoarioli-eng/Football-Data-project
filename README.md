# Football Data Project

Player tracking (YOLO + BoT-SORT), ball detection (SAHI + YOLOv11), and team assignment.

## Setup

```bash
cd Football-Data-project
python3 -m venv .venv --system-site-packages   # Jetson: reuse system PyTorch
source .venv/bin/activate
pip install -r requirements.txt
```

Requires Python 3.10+ and CUDA PyTorch.

Test clips (from `data/match_test.mp4`):

- `data/test_10s.mp4` — 10 s clip (~300 frames)
- `data/test_2min.mp4` — 2 minutes

## Player tracking

```bash
source .venv/bin/activate
python player_tracking.py \
  --video data/test_10s.mp4 \
  --conf 0.25 \
  --max-frames 300 \
  --debug-frame 150 \
  --output output/tracking_raw.csv
```

Output: `output/tracking_raw.csv`, optional `output/debug_tracking_N.jpg`.

Uses the **soccana** preset (YOLOv11n, player class) with BoT-SORT + ReID.

## Ball detection

```bash
source .venv/bin/activate
python ball_detection.py \
  --video data/test_10s.mp4 \
  --preset ball \
  --conf 0.4 \
  --max-frames 60 \
  --debug-frame 25 \
  --debug-frame 50 \
  --slice-sizes 640,1280 \
  --output output/ball_detection.csv
```

Output: `output/ball_detection.csv`, optional debug frames `output/debug_ball_detection_N.jpg`.

## Team assignment

Given a tracking CSV and video, assign `team_id` per player (0 = red kit, 1 = dark kit, -1 = referee/goalkeeper) using torso BGR features:

```bash
python assign_teams.py \
  --video data/test_2min.mp4 \
  --csv output/tracking_raw.csv \
  --debug-frame 180
```

Output: `output/tracking_teams.csv`, optional `output/debug_frame_180_teams.jpg`.
