"""YOLO + BoT-SORT player tracking for soccer video (soccana preset)."""

import argparse
import csv
import logging
from pathlib import Path

import cv2
from ultralytics import YOLO

from ball_detection import _resolve_device_fps, download_preset

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

ROOT = Path(__file__).resolve().parent

DEFAULT_TRACKER = {
    "tracker_type": "botsort",
    "fuse_score": True,
    "gmc_method": "sparseOptFlow",
    "track_high_thresh": 0.5,
    "track_low_thresh": 0.1,
    "new_track_thresh": 0.45,
    "track_buffer": 150,
    "match_thresh": 0.8,
    "proximity_thresh": 0.5,
    "appearance_thresh": 0.4,
    "with_reid": True,
    "model": "auto",
}

# soccana: 0=player, 1=ball, 2=referee
SOCCANA_CLASSES = {"player": 0, "referee": 2}


def _ensure_parent(filepath: str) -> None:
    Path(filepath).parent.mkdir(parents=True, exist_ok=True)


def _tracker_yaml_path(cfg: dict) -> str:
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


def _resolve_model(model_path: str | None, models_dir: str) -> Path:
    if model_path:
        path = Path(model_path)
        if not path.is_absolute() and not path.exists():
            path = ROOT / model_path
        if not path.exists():
            raise FileNotFoundError(f"Model not found: {path}")
        return path
    return download_preset("soccana", Path(models_dir))


def _resolve_classes(include_referee: bool) -> list[int]:
    classes = [SOCCANA_CLASSES["player"]]
    if include_referee:
        classes.append(SOCCANA_CLASSES["referee"])
    return classes


def run(args) -> None:
    device, fps = _resolve_device_fps(args.config, args.device, args.fps)
    model_path = _resolve_model(args.model, args.models_dir)
    detect_classes = _resolve_classes(args.include_referee)

    tracker_cfg = dict(DEFAULT_TRACKER)
    tracker_cfg["new_track_thresh"] = args.new_track_thresh

    logging.info("Model: %s", model_path)
    logging.info("Device: %s, imgsz=%d, conf=%.2f", device, args.imgsz, args.conf)
    logging.info("Tracking classes: %s", detect_classes)

    yolo = YOLO(str(model_path))
    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {args.video}")
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0
    cap.release()

    stream = yolo.track(
        source=args.video,
        tracker=_tracker_yaml_path(tracker_cfg),
        classes=detect_classes,
        stream=True,
        device=device,
        imgsz=args.imgsz,
        conf=args.conf,
        persist=True,
        verbose=False,
    )

    out_csv = args.output
    if not Path(out_csv).is_absolute():
        out_csv = str(ROOT / out_csv)
    _ensure_parent(out_csv)

    unique_ids: set[int] = set()
    row_count = 0

    with open(out_csv, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(CSV_HEADER)

        for frame_idx, result in enumerate(stream):
            if args.max_frames is not None and frame_idx >= args.max_frames:
                break

            if frame_idx % 100 == 0 and total_frames:
                logging.info(
                    "Frame %d/%d (%.1f%%)",
                    frame_idx,
                    total_frames,
                    100.0 * frame_idx / total_frames,
                )

            if args.debug_frame is not None and frame_idx == args.debug_frame:
                out = args.debug_out or str(ROOT / f"output/debug_tracking_{frame_idx}.jpg")
                _ensure_parent(out)
                cv2.imwrite(out, result.plot())
                logging.info("Debug image saved: %s", out)

            boxes = result.boxes
            if boxes is None or len(boxes) == 0:
                continue

            ts = frame_idx / fps
            for i in range(len(boxes)):
                cls = int(boxes.cls[i])
                if cls == SOCCANA_CLASSES["player"]:
                    obj_type = "player"
                elif cls == SOCCANA_CLASSES["referee"]:
                    obj_type = "referee"
                else:
                    continue

                track_id = int(boxes.id[i]) if boxes.id is not None else -1
                if track_id < 0:
                    continue

                conf = float(boxes.conf[i])
                x, y, w, h = (float(v) for v in boxes.xywh[i].cpu().numpy())
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
                unique_ids.add(track_id)
                row_count += 1

    logging.info("Rows written: %d", row_count)
    logging.info("Unique track IDs: %d", len(unique_ids))
    logging.info("Wrote %s", out_csv)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="YOLO + BoT-SORT player tracking (soccana preset)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--video", required=True, help="Input MP4 path")
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument(
        "--model",
        default=None,
        help="Local .pt path (default: download soccana from Hugging Face)",
    )
    parser.add_argument("--models-dir", default="models")
    parser.add_argument("--imgsz", type=int, default=1280, help="YOLO inference size")
    parser.add_argument("--conf", type=float, default=0.25, help="Detection confidence")
    parser.add_argument(
        "--new-track-thresh",
        type=float,
        default=0.45,
        help="BoT-SORT: min conf to start a new track",
    )
    parser.add_argument(
        "--include-referee",
        action="store_true",
        help="Also track referee detections (class 2)",
    )
    parser.add_argument("--device", default=None)
    parser.add_argument("--fps", type=float, default=None)
    parser.add_argument("--output", default="output/tracking_raw.csv")
    parser.add_argument("--max-frames", type=int, default=None)
    parser.add_argument("--debug-frame", type=int, default=None)
    parser.add_argument("--debug-out", default=None)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    run(args)


if __name__ == "__main__":
    main()
