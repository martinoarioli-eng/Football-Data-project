"""Unified player + ball tracking pipeline (single CSV output)."""

import argparse
import csv
import logging
from pathlib import Path

import cv2
import numpy as np
import supervision as sv
from sahi import AutoDetectionModel
from ultralytics import YOLO

from assign_teams import _fit_teams
from ball_detection import (
    DEFAULT_OVERLAP,
    detect_frame_multiscale,
    resolve_model,
)
from ball_tracker import BallCandidate, BallTracker
from player_tracking import (
    DEFAULT_TRACKER,
    SOCCANA_CLASSES,
    _resolve_model as resolve_player_model,
    _tracker_yaml_path,
)

ROOT = Path(__file__).resolve().parent

CSV_HEADER = [
    "frame",
    "timestamp_sec",
    "type",
    "player_id",
    "team_id",
    "x_center",
    "y_center",
    "width",
    "height",
    "confidence",
    "ball_predicted",
]

BALL_COLOR = (255, 0, 255)
PLAYER_COLOR = (255, 120, 0)


def _ensure_parent(filepath: str) -> None:
    Path(filepath).parent.mkdir(parents=True, exist_ok=True)


def _resolve_path(path: str) -> str:
    p = Path(path)
    if p.is_absolute():
        return str(p)
    if p.exists():
        return str(p.resolve())
    return str(ROOT / path)


def _detections_to_candidates(dets: sv.Detections) -> list[BallCandidate]:
    out: list[BallCandidate] = []
    for i in range(len(dets)):
        x1, y1, x2, y2 = (float(v) for v in dets.xyxy[i])
        out.append(
            BallCandidate(
                x=(x1 + x2) / 2,
                y=(y1 + y2) / 2,
                w=x2 - x1,
                h=y2 - y1,
                conf=float(dets.confidence[i]),
            )
        )
    return out


def _player_row(
    frame_idx: int,
    ts: float,
    track_id: int,
    team_id: str,
    x: float,
    y: float,
    w: float,
    h: float,
    conf: float,
) -> list:
    return [
        frame_idx,
        f"{ts:.4f}",
        "player",
        track_id,
        team_id,
        f"{x:.2f}",
        f"{y:.2f}",
        f"{w:.2f}",
        f"{h:.2f}",
        f"{conf:.4f}",
        "",
    ]


def _ball_row(frame_idx: int, ts: float, obs) -> list:
    return [
        frame_idx,
        f"{ts:.4f}",
        "ball",
        1,
        "",
        f"{obs.x:.2f}",
        f"{obs.y:.2f}",
        f"{obs.w:.2f}",
        f"{obs.h:.2f}",
        f"{obs.conf:.4f}",
        1 if obs.predicted else 0,
    ]


def _annotate_debug(
    frame: np.ndarray,
    player_boxes,
    ball_obs,
    team_by_player: dict[int, int],
) -> np.ndarray:
    scene = frame.copy()
    if player_boxes is not None and len(player_boxes) > 0:
        for i in range(len(player_boxes)):
            track_id = int(player_boxes.id[i]) if player_boxes.id is not None else -1
            if track_id < 0:
                continue
            x, y, w, h = (float(v) for v in player_boxes.xywh[i].cpu().numpy())
            x1, y1 = int(x - w / 2), int(y - h / 2)
            x2, y2 = int(x + w / 2), int(y + h / 2)
            team_id = team_by_player.get(track_id, "")
            color = PLAYER_COLOR
            label = f"id:{track_id}"
            if team_id != "":
                label = f"id:{track_id} t:{team_id}"
            cv2.rectangle(scene, (x1, y1), (x2, y2), color, 2)
            cv2.putText(
                scene,
                label,
                (x1, max(y1 - 4, 12)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                color,
                1,
                cv2.LINE_AA,
            )

    if ball_obs is not None:
        x1 = int(ball_obs.x - ball_obs.w / 2)
        y1 = int(ball_obs.y - ball_obs.h / 2)
        x2 = int(ball_obs.x + ball_obs.w / 2)
        y2 = int(ball_obs.y + ball_obs.h / 2)
        label = "ball pred" if ball_obs.predicted else f"ball {ball_obs.conf:.2f}"
        cv2.rectangle(scene, (x1, y1), (x2, y2), BALL_COLOR, 2)
        cv2.putText(
            scene,
            label,
            (x1, max(y1 - 4, 12)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            BALL_COLOR,
            1,
            cv2.LINE_AA,
        )
    return scene


def run(args) -> None:
    device = args.device or "cuda"
    fps = args.fps or 29.97

    player_model_path = resolve_player_model(args.player_model, args.models_dir)
    ball_model_path, ball_class_names, ball_class = resolve_model(
        args.ball_preset, args.ball_model, Path(args.models_dir)
    )

    tracker_cfg = dict(DEFAULT_TRACKER)
    tracker_cfg["new_track_thresh"] = args.new_track_thresh
    tracker_yaml = _tracker_yaml_path(tracker_cfg)

    logging.info("Player model: %s", player_model_path)
    logging.info("Ball model: %s (SAHI slices: %s)", ball_model_path, args.slice_sizes)

    yolo = YOLO(str(player_model_path))
    ball_model = AutoDetectionModel.from_pretrained(
        model_type="ultralytics",
        model_path=str(ball_model_path),
        confidence_threshold=args.ball_conf,
        device=device,
    )

    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {args.video}")
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0

    slice_sizes = tuple(int(s) for s in args.slice_sizes.split(","))
    ball_tracker = BallTracker(
        max_dist=args.ball_max_dist,
        max_size=args.ball_max_size,
        max_gap=args.ball_max_gap,
        min_conf_init=args.ball_min_conf_init,
        player_proximity=args.ball_player_proximity,
    )

    out_csv = _resolve_path(args.output)
    _ensure_parent(out_csv)

    all_rows: list[dict] = []
    player_rows_for_teams: list[dict] = []
    ball_frames = 0
    ball_predicted = 0
    debug_frames = set(args.debug_frame or [])

    with open(out_csv, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(CSV_HEADER)
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

            ts = frame_idx / fps
            result = yolo.track(
                frame,
                persist=True,
                tracker=tracker_yaml,
                classes=[SOCCANA_CLASSES["player"]],
                device=device,
                imgsz=args.imgsz,
                conf=args.player_conf,
                verbose=False,
            )[0]

            player_positions: list[tuple[float, float]] = []
            frame_player_rows: list[list] = []
            boxes = result.boxes

            if boxes is not None and len(boxes) > 0:
                for i in range(len(boxes)):
                    track_id = int(boxes.id[i]) if boxes.id is not None else -1
                    if track_id < 0:
                        continue
                    conf = float(boxes.conf[i])
                    x, y, w, h = (float(v) for v in boxes.xywh[i].cpu().numpy())
                    player_positions.append((x, y))
                    row = _player_row(frame_idx, ts, track_id, "", x, y, w, h, conf)
                    frame_player_rows.append(row)
                    player_rows_for_teams.append(
                        {
                            "frame": str(frame_idx),
                            "timestamp_sec": f"{ts:.4f}",
                            "type": "player",
                            "player_id": str(track_id),
                            "x_center": f"{x:.2f}",
                            "y_center": f"{y:.2f}",
                            "width": f"{w:.2f}",
                            "height": f"{h:.2f}",
                            "confidence": f"{conf:.4f}",
                        }
                    )

            ball_dets = detect_frame_multiscale(
                frame,
                ball_model,
                slice_sizes,
                args.overlap,
                ball_class_names,
                args.nms_iou,
            )
            if len(ball_dets) > 0:
                keep = ball_dets.class_id == ball_class
                ball_dets = ball_dets[keep]

            ball_obs = ball_tracker.update(
                _detections_to_candidates(ball_dets),
                players=player_positions,
            )

            for row in frame_player_rows:
                writer.writerow(row)
                all_rows.append(row)

            if ball_obs is not None:
                ball_row = _ball_row(frame_idx, ts, ball_obs)
                writer.writerow(ball_row)
                all_rows.append(ball_row)
                ball_frames += 1
                if ball_obs.predicted:
                    ball_predicted += 1

            if frame_idx in debug_frames:
                out = args.debug_out or str(
                    ROOT / f"output/debug_unified_{frame_idx}.jpg"
                )
                _ensure_parent(out)
                scene = _annotate_debug(frame, boxes, ball_obs, {})
                cv2.imwrite(out, scene)
                logging.info("Debug image saved: %s", out)

            frame_idx += 1

    cap.release()

    if not args.no_teams and player_rows_for_teams:
        team_by_player = _fit_teams(
            args.video,
            player_rows_for_teams,
            sample_stride=args.sample_stride,
            max_samples_per_player=args.max_samples_per_player,
        )
        with open(out_csv, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(CSV_HEADER)
            for row in all_rows:
                if row[2] == "player":
                    row[4] = team_by_player.get(int(row[3]), 0)
                writer.writerow(row)
        logging.info("Team assignment applied")

    logging.info("Ball frames: %d (predicted: %d)", ball_frames, ball_predicted)
    logging.info("Wrote %s", out_csv)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Unified player (BoT-SORT) + ball (SAHI + temporal) tracking",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--video", required=True)
    parser.add_argument("--output", default="output/unified.csv")
    parser.add_argument("--models-dir", default="models")
    parser.add_argument("--device", default=None)
    parser.add_argument("--fps", type=float, default=None)
    parser.add_argument("--max-frames", type=int, default=None)

    parser.add_argument("--player-model", default=None, help="Player YOLO .pt (default: soccana)")
    parser.add_argument("--player-conf", type=float, default=0.25)
    parser.add_argument("--imgsz", type=int, default=1280)
    parser.add_argument("--new-track-thresh", type=float, default=0.45)

    parser.add_argument("--ball-preset", choices=["ball", "soccana"], default="ball")
    parser.add_argument("--ball-model", default=None)
    parser.add_argument("--ball-conf", type=float, default=0.25)
    parser.add_argument("--slice-sizes", default="640,1280")
    parser.add_argument("--overlap", type=float, default=DEFAULT_OVERLAP)
    parser.add_argument("--nms-iou", type=float, default=0.3)
    parser.add_argument("--ball-max-dist", type=float, default=150.0)
    parser.add_argument("--ball-max-size", type=float, default=20.0)
    parser.add_argument("--ball-max-gap", type=int, default=5)
    parser.add_argument("--ball-min-conf-init", type=float, default=0.57)
    parser.add_argument("--ball-player-proximity", type=float, default=150.0)

    parser.add_argument("--no-teams", action="store_true", help="Skip team assignment pass")
    parser.add_argument("--sample-stride", type=int, default=15)
    parser.add_argument("--max-samples-per-player", type=int, default=20)

    parser.add_argument("--debug-frame", type=int, action="append")
    parser.add_argument("--debug-out", default=None)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    run(args)


if __name__ == "__main__":
    main()
