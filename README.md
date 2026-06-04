# Football Player Tracking

YOLO11 detection + BoT-SORT tracking on follow-cam match video. Writes per-frame boxes to CSV.

## Setup

```bash
cd Football-Data-project
python3 -m venv .venv --system-site-packages   # Jetson: reuse system PyTorch
source .venv/bin/activate
pip install -r requirements.txt
```

Requires Python 3.10+, CUDA PyTorch, and `models/yolo11m.pt` (auto-downloaded by Ultralytics if missing).

## Run

```bash
source .venv/bin/activate
python -m pipeline.main --video data/test_2min.mp4
```

Output: `output/tracking_raw.csv`

Columns: `frame`, `timestamp_sec`, `type` (`player` | `ball`), `player_id`, `x_center`, `y_center`, `width`, `height`, `confidence`.

## Debug frame

Save one annotated frame (boxes + track IDs):

```bash
python -m pipeline.main --video data/test_2min.mp4 --debug-frame 25
```

Writes `output/debug_frame_25.jpg` (0-based frame index).

## Config

Edit constants in `pipeline/main.py` → `main()`: FPS, model path, device, confidence, classes, BoT-SORT `tracker` dict.
