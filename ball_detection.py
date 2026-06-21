"""YOLOv11 + SAHI multi-scale detection for soccer small objects (ball)."""

import argparse
import csv
import logging
from pathlib import Path

import cv2
import numpy as np
import supervision as sv
import yaml
from huggingface_hub import hf_hub_download
from sahi import AutoDetectionModel
from sahi.predict import get_sliced_prediction


def _ensure_parent(filepath: str) -> None:
    Path(filepath).parent.mkdir(parents=True, exist_ok=True)


def _resolve_device_fps(config_path: str, device: str | None, fps: float | None):
    resolved_device = device or "cuda"
    resolved_fps = fps or 29.97
    if not Path(config_path).exists():
        return resolved_device, resolved_fps
    with open(config_path) as f:
        raw = yaml.safe_load(f) or {}
    if device is None:
        resolved_device = raw.get("detection", {}).get("device", resolved_device)
    if fps is None:
        resolved_fps = float(raw.get("video", {}).get("fps", resolved_fps))
    return resolved_device, resolved_fps

CSV_HEADER = [
    "frame",
    "timestamp_sec",
    "type",
    "player_id",
    "team",
    "x_center",
    "y_center",
    "width",
    "height",
    "confidence",
]

DEFAULT_SLICE_SIZES = (160, 320, 640, 1280)
DEFAULT_OVERLAP = 0.2

# HF models found via Hub search (Jun 2025–Jan 2026)
HF_PRESETS = {
    "soccana": {
        "repo_id": "Adit-jain/soccana",
        "filename": "Model/weights/best.pt",
        "local_name": "soccana_best.pt",
        "ball_class": 1,
        "class_names": {0: "player", 1: "ball", 2: "referee"},
        "note": "YOLOv11n, player/ball/referee, trained with SAHI-style tiling at 1280",
    },
    "ball": {
        "repo_id": "martinjolif/yolo-football-ball-detection",
        "filename": "yolo-football-ball-detection.pt",
        "local_name": "yolo-football-ball-detection.pt",
        "ball_class": 0,
        "class_names": {0: "ball"},
        "note": "YOLOv11n ball-only, mAP50=0.89 on test split",
    },
}


def download_preset(preset: str, models_dir: Path) -> Path:
    spec = HF_PRESETS[preset]
    dest = models_dir / spec["local_name"]
    if dest.exists():
        return dest
    models_dir.mkdir(parents=True, exist_ok=True)
    logging.info("Downloading %s from Hugging Face …", spec["repo_id"])
    path = hf_hub_download(
        repo_id=spec["repo_id"],
        filename=spec["filename"],
        local_dir=str(models_dir / preset),
    )
    return Path(path)


def resolve_model(
    preset: str | None,
    model_path: str | None,
    models_dir: Path,
) -> tuple[Path, dict[int, str], int]:
    if model_path:
        path = Path(model_path)
        if not path.exists():
            raise FileNotFoundError(f"Model not found: {path}")
        return path, {32: "ball", 0: "player"}, 32

    if preset not in HF_PRESETS:
        raise ValueError(f"Unknown preset {preset!r}. Choose: {list(HF_PRESETS)}")

    spec = HF_PRESETS[preset]
    path = download_preset(preset, models_dir)
    return path, spec["class_names"], spec["ball_class"]


def _predictions_to_detections(predictions, class_names: dict[int, str]):
    if not predictions:
        return sv.Detections.empty()

    xyxy, conf, cls = [], [], []
    for p in predictions:
        cid = int(p.category.id)
        if cid not in class_names:
            continue
        xyxy.append([p.bbox.minx, p.bbox.miny, p.bbox.maxx, p.bbox.maxy])
        conf.append(float(p.score.value))
        cls.append(cid)
    if not xyxy:
        return sv.Detections.empty()
    return sv.Detections(
        xyxy=np.array(xyxy, dtype=np.float32),
        confidence=np.array(conf, dtype=np.float32),
        class_id=np.array(cls, dtype=int),
    )


def detect_frame_multiscale(
    frame: np.ndarray,
    detection_model: AutoDetectionModel,
    slice_sizes: tuple[int, ...],
    overlap: float,
    class_names: dict[int, str],
    nms_iou: float,
) -> sv.Detections:
    merged = sv.Detections.empty()
    for size in slice_sizes:
        result = get_sliced_prediction(
            frame,
            detection_model,
            slice_height=size,
            slice_width=size,
            overlap_height_ratio=overlap,
            overlap_width_ratio=overlap,
            verbose=0,
        )
        dets = _predictions_to_detections(result.object_prediction_list, class_names)
        if len(dets) == 0:
            continue
        merged = dets if len(merged) == 0 else sv.Detections.merge([merged, dets])

    if len(merged) == 0:
        return merged
    return merged.with_nms(threshold=nms_iou)


def _annotate_frame(
    frame: np.ndarray, dets: sv.Detections, class_names: dict[int, str]
) -> np.ndarray:
    labels = [
        f"{class_names.get(int(c), c)} {conf:.2f}"
        for c, conf in zip(dets.class_id, dets.confidence)
    ]
    scene = sv.BoxAnnotator().annotate(scene=frame.copy(), detections=dets)
    return sv.LabelAnnotator().annotate(scene=scene, detections=dets, labels=labels)


def _xyxy_to_row(frame_idx: int, ts: float, obj_type: str, xyxy, conf: float):
    x1, y1, x2, y2 = (float(v) for v in xyxy)
    w, h = x2 - x1, y2 - y1
    pid = 1 if obj_type == "ball" else -1
    return [
        frame_idx,
        f"{ts:.4f}",
        obj_type,
        pid,
        "",
        f"{(x1 + x2) / 2:.2f}",
        f"{(y1 + y2) / 2:.2f}",
        f"{w:.2f}",
        f"{h:.2f}",
        f"{conf:.4f}",
    ]


def run(args) -> None:
    device, fps = _resolve_device_fps(args.config, args.device, args.fps)

    model_path, class_names, ball_class = resolve_model(
        args.preset, args.model, Path(args.models_dir)
    )
    detect_classes = set(args.classes)
    if detect_classes == {"ball"}:
        detect_classes = {ball_class}

    logging.info("Model: %s", model_path)
    logging.info("SAHI slices: %s, overlap=%.0f%%", args.slice_sizes, args.overlap * 100)

    detection_model = AutoDetectionModel.from_pretrained(
        model_type="ultralytics",
        model_path=str(model_path),
        confidence_threshold=args.conf,
        device=device,
    )

    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {args.video}")

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0
    _ensure_parent(args.output)
    writer = None
    if args.save_video:
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(args.save_video, fourcc, fps, (w, h))

    slice_sizes = tuple(int(s) for s in args.slice_sizes.split(","))
    ball_count = 0
    debug_frames = set(args.debug_frame or [])

    with open(args.output, "w", newline="") as f:
        csv_writer = csv.writer(f)
        csv_writer.writerow(CSV_HEADER)
        frame_idx = 0
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            if args.max_frames is not None and frame_idx >= args.max_frames:
                break
            if frame_idx % 100 == 0 and total_frames:
                logging.info(
                    "Frame %d/%d (%.1f%%)",
                    frame_idx,
                    total_frames,
                    100.0 * frame_idx / total_frames,
                )

            dets = detect_frame_multiscale(
                frame,
                detection_model,
                slice_sizes,
                args.overlap,
                class_names,
                args.nms_iou,
            )
            ts = frame_idx / fps

            if len(dets) > 0:
                for i in range(len(dets)):
                    cid = int(dets.class_id[i])
                    if cid not in detect_classes:
                        continue
                    obj_type = class_names.get(cid, "unknown")
                    row = _xyxy_to_row(
                        frame_idx, ts, obj_type, dets.xyxy[i], float(dets.confidence[i])
                    )
                    csv_writer.writerow(row)
                    if obj_type == "ball":
                        ball_count += 1

            if frame_idx in debug_frames:
                out = args.debug_out or f"output/debug_ball_detection_{frame_idx}.jpg"
                _ensure_parent(out)
                scene = (
                    _annotate_frame(frame, dets, class_names)
                    if len(dets) > 0
                    else frame.copy()
                )
                cv2.imwrite(out, scene)
                logging.info("Debug image saved: %s", out)

            if writer is not None and len(dets) > 0:
                writer.write(_annotate_frame(frame, dets, class_names))
            elif writer is not None:
                writer.write(frame)

            frame_idx += 1

    cap.release()
    if writer is not None:
        writer.release()

    logging.info("Ball detections written: %d", ball_count)
    logging.info("Wrote %s", args.output)


def main():
    parser = argparse.ArgumentParser(
        description="YOLOv11 + SAHI multi-scale soccer ball detection",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        epilog=(
            "HF presets (from Hub research):\n"
            "  soccana  Adit-jain/soccana — YOLOv11n player/ball/referee (8k+ downloads)\n"
            "  ball     martinjolif/yolo-football-ball-detection — ball-only YOLOv11n\n"
            "Also on Hub: julianzu9612/RFDETR-Soccernet (RF-DETR, SoccerNet-Tracking)\n"
            "Dataset: Voxel51/SoccerNet-V3, Adit-jain/Soccana_player_ball_detection_v1"
        ),
    )
    parser.add_argument("--video", help="Input MP4 path")
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument(
        "--preset",
        choices=list(HF_PRESETS),
        default="soccana",
        help="Hugging Face model preset",
    )
    parser.add_argument(
        "--model",
        default=None,
        help="Local .pt path (overrides --preset; COCO classes 0/32 assumed)",
    )
    parser.add_argument("--models-dir", default="models", help="HF download cache dir")
    parser.add_argument(
        "--classes",
        nargs="+",
        default=["ball"],
        choices=["ball", "player", "referee"],
        help="Object types to export",
    )
    parser.add_argument(
        "--slice-sizes",
        default=",".join(str(s) for s in DEFAULT_SLICE_SIZES),
        help="Comma-separated SAHI patch sizes (px)",
    )
    parser.add_argument("--overlap", type=float, default=DEFAULT_OVERLAP)
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--nms-iou", type=float, default=0.5)
    parser.add_argument("--device", default=None)
    parser.add_argument("--fps", type=float, default=None)
    parser.add_argument(
        "--output",
        default="output/ball_detection.csv",
        help="Output CSV path",
    )
    parser.add_argument("--save-video", default=None, help="Optional annotated MP4 output")
    parser.add_argument(
        "--max-frames",
        type=int,
        default=None,
        help="Stop after N frames (for quick tests)",
    )
    parser.add_argument(
        "--debug-frame",
        type=int,
        action="append",
        help="Save annotated JPG for frame index (repeatable, 0-based)",
    )
    parser.add_argument(
        "--debug-out",
        default=None,
        help="Path for debug image when a single --debug-frame is set",
    )
    parser.add_argument(
        "--list-presets",
        action="store_true",
        help="Print HF preset info and exit",
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    if args.list_presets:
        for name, spec in HF_PRESETS.items():
            print(f"{name}: {spec['repo_id']} — {spec['note']}")
        return

    if not args.video:
        parser.error("--video is required")

    run(args)


if __name__ == "__main__":
    main()
