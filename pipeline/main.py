import argparse
import csv
import logging
from pathlib import Path

import cv2
from ultralytics import YOLO

CSV_HEADER = [
    "frame",
    "timestamp_sec",
    "type",
    "player_id",
    "x_center",
    "y_center",
    "width",
    "height",
    "confidence",
]

ROOT = Path(__file__).resolve().parent.parent


def _ensure_parent(filepath: str) -> None:
    Path(filepath).parent.mkdir(parents=True, exist_ok=True)


def _tracker_yaml_path(cfg: dict) -> str:
    """Ultralytics accetta solo un path YAML; scriviamo la config da dict."""
    path = ROOT / "output" / ".tracker.yaml"
    _ensure_parent(str(path))

    def fmt(value):
        if isinstance(value, bool):
            return "true" if value else "false"
        if isinstance(value, str):
            return value
        return str(value)

    path.write_text("\n".join(f"{k}: {fmt(v)}" for k, v in cfg.items()) + "\n")
    return str(path)


class DetectAndTrack:
    def __init__(
        self,
        *,
        fps: float,
        model: str,
        device: str,
        imgsz: int,
        conf_threshold: float,
        classes: list[int],
        tracker: dict,
        raw_csv: str,
        debug_frame: int | None = None,
        debug_out: str | None = None,
    ):
        self.fps = fps
        self.model = model
        self.device = device
        self.imgsz = imgsz
        self.conf_threshold = conf_threshold
        self.classes = classes
        self.tracker = tracker
        self.raw_csv = raw_csv
        self.debug_frame = debug_frame
        self.debug_out = debug_out

    def run(self, video_path: str) -> None:
        _ensure_parent(self.raw_csv)

        model_path = self.model
        if not Path(model_path).is_absolute():
            p = Path(model_path)
            if not p.exists():
                p = ROOT / model_path
            model_path = str(p)
        yolo = YOLO(model_path)
        logging.info("YOLO loaded on %s (imgsz=%s)", self.device, self.imgsz)

        cap = cv2.VideoCapture(video_path)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0
        cap.release()

        stream = yolo.track(
            source=video_path,
            tracker=_tracker_yaml_path(self.tracker),
            classes=self.classes,
            stream=True,
            device=self.device,
            imgsz=self.imgsz,
            conf=self.conf_threshold,
            persist=True,
            verbose=False,
        )

        unique_ids: set[int] = set()

        with open(self.raw_csv, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(CSV_HEADER)
            for frame_idx, result in enumerate(stream):
                if frame_idx % 300 == 0 and total_frames:
                    pct = 100.0 * frame_idx / total_frames
                    logging.info("Frame %d/%d (%.1f%%)", frame_idx, total_frames, pct)

                if self.debug_frame is not None and frame_idx == self.debug_frame:
                    out = self.debug_out or str(ROOT / f"output/debug_frame_{frame_idx}.jpg")
                    _ensure_parent(out)
                    cv2.imwrite(out, result.plot())
                    logging.info("Debug image saved: %s", out)

                boxes = result.boxes
                if boxes is None or len(boxes) == 0:
                    continue

                ts = frame_idx / self.fps

                for i in range(len(boxes)):
                    cls = int(boxes.cls[i])
                    conf = float(boxes.conf[i])
                    xywh = boxes.xywh[i].cpu().numpy()
                    x, y, w, h = (float(v) for v in xywh)
                    track_id = int(boxes.id[i]) if boxes.id is not None else -1
                    if track_id < 0:
                        continue

                    if cls == 0:
                        obj_type = "player"
                        unique_ids.add(track_id)
                    elif cls == 32:
                        obj_type = "ball"
                    else:
                        continue

                    writer.writerow(
                        [
                            frame_idx,
                            f"{ts:.4f}",
                            obj_type,
                            track_id,
                            f"{x:.2f}",
                            f"{y:.2f}",
                            f"{w:.2f}",
                            f"{h:.2f}",
                            f"{conf:.4f}",
                        ]
                    )

        logging.info("Unique track IDs: %d", len(unique_ids))
        logging.info("Wrote %s", self.raw_csv)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", required=True)
    parser.add_argument(
        "--debug-frame",
        type=int,
        default=None,
        help="Save annotated image for this frame index (0-based)",
    )
    parser.add_argument(
        "--debug-out",
        default=None,
        help="Path for debug image (default output/debug_frame_N.jpg)",
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)

    DetectAndTrack(
        fps=29.97,
        model="models/yolo11m.pt",
        device="cuda",
        imgsz=640,
        conf_threshold=0.25,
        classes=[0, 32],
        tracker={
            "tracker_type": "botsort",
            "fuse_score": True,
            "gmc_method": "sparseOptFlow",
            "track_high_thresh": 0.5,
            "track_low_thresh": 0.1,
            "new_track_thresh": 0.6,
            "track_buffer": 150,
            "match_thresh": 0.8,
            "proximity_thresh": 0.5,
            "appearance_thresh": 0.4,
            "with_reid": True,
            "model": "auto",
        },
        raw_csv="output/tracking_raw.csv",
        debug_frame=args.debug_frame,
        debug_out=args.debug_out,
    ).run(args.video)


if __name__ == "__main__":
    main()
