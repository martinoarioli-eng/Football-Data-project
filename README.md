# Football Player Tracking

YOLO11 detection + BoT-SORT tracking on follow-cam match video. Writes per-frame boxes to CSV.

## Setup

```bash
cd Football-Data-project
python3 -m venv .venv --system-site-packages   # Jetson: reuse system PyTorch
source .venv/bin/activate
pip install -r requirements.txt
```

Requires Python 3.10+, CUDA PyTorch, and `models/yolo-football.pt` ([martinjolif YOLO11m](https://huggingface.co/martinjolif/yolo-football-player-detection)).

```bash
wget -O models/yolo-football.pt \
  https://huggingface.co/martinjolif/yolo-football-player-detection/resolve/main/yolo-football-player-detection.pt
```

## Run

Test clips (from `data/match_test.mp4`):

- `data/test_10s.mp4` — 10 s clip from 9:50–10:00 of `match_test.mp4` (~300 frames)
- `data/test_2min.mp4` — 2 minutes

```bash
source .venv/bin/activate
python -m pipeline.main --video data/test_10s.mp4
```

Output: `output/tracking_raw.csv`

Columns: `frame`, `timestamp_sec`, `type` (`player` | `ball`), `player_id`, `x_center`, `y_center`, `width`, `height`, `confidence`.

## Debug frame

Save one annotated frame (boxes + track IDs):

```bash
python -m pipeline.main --video data/test_2min.mp4 --debug-frame 25
```

Writes `output/debug_frame_25.jpg` (0-based frame index).

## Teams (phase 2)

After tracking, assign `team_id` (0 = red kit, 1 = dark kit, -1 = referee/goalkeeper) using torso BGR features, and save a team-colored debug frame:

```bash
python -m pipeline.assign_teams --video data/test_2min.mp4 --debug-frame 180
```

Output: `output/tracking_teams.csv`, `output/debug_frame_180_teams.jpg`.

## Config

CLI flags on `pipeline.main` (defaults tuned for wide Veo-style footage):

- `--imgsz 1280` — higher resolution for distant players
- `--conf 0.18` — lower threshold for small boxes
- `--new-track-thresh 0.45` — accept weaker detections as new tracks
- `--max-frames N` — process only the first N frames (quick tests)

Example quick test on one frame:

```bash
python -m pipeline.main --video data/test_2min.mp4 --max-frames 182 --debug-frame 181
python -m pipeline.assign_teams --video data/test_2min.mp4 --debug-frame 181
```
