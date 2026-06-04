#!/usr/bin/env python3
"""Visualizza posizioni 2D (x_center, y_center) da tracking_teams.csv come video."""

import argparse
import csv
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np

TEAM_COLORS = {
    0: (60, 60, 220),   # BGR rosso
    1: (220, 140, 40),  # BGR blu/arancio
    -1: (160, 160, 160),
}
BALL_COLOR = (0, 220, 255)
FIELD_COLOR = (34, 120, 34)
LINE_COLOR = (220, 220, 220)
TEXT_COLOR = (255, 255, 255)


def load_frames(csv_path: Path) -> tuple[dict[int, list[dict]], float]:
    by_frame: dict[int, list[dict]] = defaultdict(list)
    timestamps: list[float] = []
    with csv_path.open(newline="") as f:
        for row in csv.DictReader(f):
            frame = int(row["frame"])
            by_frame[frame].append(row)
            timestamps.append(float(row["timestamp_sec"]))
    if len(timestamps) < 2:
        fps = 30.0
    else:
        deltas = [
            timestamps[i + 1] - timestamps[i]
            for i in range(len(timestamps) - 1)
            if timestamps[i + 1] > timestamps[i]
        ]
        fps = 1.0 / float(np.median(deltas)) if deltas else 30.0
    return dict(by_frame), fps


def bounds(by_frame: dict[int, list[dict]], margin: float) -> tuple[float, float, float, float]:
    xs, ys = [], []
    for rows in by_frame.values():
        for row in rows:
            xs.append(float(row["x_center"]))
            ys.append(float(row["y_center"]))
    xmin, xmax = min(xs), max(xs)
    ymin, ymax = min(ys), max(ys)
    pad_x = (xmax - xmin) * margin or margin * 100
    pad_y = (ymax - ymin) * margin or margin * 100
    return xmin - pad_x, xmax + pad_x, ymin - pad_y, ymax + pad_y


def to_canvas(
    x: float,
    y: float,
    xmin: float,
    xmax: float,
    ymin: float,
    ymax: float,
    width: int,
    height: int,
) -> tuple[int, int]:
    px = int((x - xmin) / (xmax - xmin) * (width - 1))
    py = int((y - ymin) / (ymax - ymin) * (height - 1))
    return px, py


def draw_pitch(
    img: np.ndarray,
    xmin: float,
    xmax: float,
    ymin: float,
    ymax: float,
) -> None:
    h, w = img.shape[:2]
    cv2.rectangle(img, (0, 0), (w - 1, h - 1), LINE_COLOR, 2)
    mid_x, _ = to_canvas((xmin + xmax) / 2, ymin, xmin, xmax, ymin, ymax, w, h)
    cv2.line(img, (mid_x, 0), (mid_x, h - 1), LINE_COLOR, 1)
    cx, cy = to_canvas((xmin + xmax) / 2, (ymin + ymax) / 2, xmin, xmax, ymin, ymax, w, h)
    radius = int(min(w, h) * 0.12)
    cv2.circle(img, (cx, cy), radius, LINE_COLOR, 1)


def draw_frame(
    img: np.ndarray,
    rows: list[dict],
    xmin: float,
    xmax: float,
    ymin: float,
    ymax: float,
    frame_idx: int,
    timestamp: float,
) -> None:
    h, w = img.shape[:2]
    img[:] = FIELD_COLOR
    draw_pitch(img, xmin, xmax, ymin, ymax)

    players = [r for r in rows if r["type"] == "player"]
    balls = [r for r in rows if r["type"] == "ball"]

    for row in players:
        team_raw = row.get("team_id", "").strip()
        team = int(team_raw) if team_raw not in ("", None) else 0
        color = TEAM_COLORS.get(team, TEAM_COLORS[-1])
        px, py = to_canvas(
            float(row["x_center"]),
            float(row["y_center"]),
            xmin,
            xmax,
            ymin,
            ymax,
            w,
            h,
        )
        cv2.circle(img, (px, py), 10, color, -1)
        cv2.circle(img, (px, py), 10, (0, 0, 0), 1)
        label = str(row["player_id"])
        cv2.putText(
            img,
            label,
            (px + 12, py + 4),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.4,
            TEXT_COLOR,
            1,
            cv2.LINE_AA,
        )

    for row in balls:
        px, py = to_canvas(
            float(row["x_center"]),
            float(row["y_center"]),
            xmin,
            xmax,
            ymin,
            ymax,
            w,
            h,
        )
        cv2.circle(img, (px, py), 7, BALL_COLOR, -1)
        cv2.circle(img, (px, py), 7, (0, 0, 0), 2)

    legend_y = 24
    for team, (name, color) in [
        (0, ("Team 0", TEAM_COLORS[0])),
        (1, ("Team 1", TEAM_COLORS[1])),
        (-1, ("Staff", TEAM_COLORS[-1])),
    ]:
        cv2.circle(img, (16, legend_y), 6, color, -1)
        cv2.putText(
            img,
            name,
            (28, legend_y + 5),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            TEXT_COLOR,
            1,
            cv2.LINE_AA,
        )
        legend_y += 22
    cv2.circle(img, (16, legend_y), 5, BALL_COLOR, -1)
    cv2.putText(
        img,
        "Ball",
        (28, legend_y + 5),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.45,
        TEXT_COLOR,
        1,
        cv2.LINE_AA,
    )

    info = f"frame {frame_idx}  t={timestamp:.2f}s  players={len(players)}  ball={'yes' if balls else 'no'}"
    cv2.putText(
        img,
        info,
        (8, h - 12),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        TEXT_COLOR,
        1,
        cv2.LINE_AA,
    )


def main() -> None:
    root = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description="Video 2D da tracking_teams.csv")
    parser.add_argument(
        "--csv",
        type=Path,
        default=root / "output" / "tracking_teams.csv",
        help="CSV con frame, x_center, y_center, type, team_id",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=root / "output" / "tracking_2d.mp4",
        help="Video MP4 in uscita",
    )
    parser.add_argument("--width", type=int, default=1280, help="Larghezza canvas")
    parser.add_argument("--height", type=int, default=720, help="Altezza canvas")
    parser.add_argument(
        "--margin",
        type=float,
        default=0.05,
        help="Margine relativo sui bound degli assi",
    )
    args = parser.parse_args()

    by_frame, fps = load_frames(args.csv)
    if not by_frame:
        raise SystemExit(f"Nessun dato in {args.csv}")

    frame_ids = sorted(by_frame)
    xmin, xmax, ymin, ymax = bounds(by_frame, args.margin)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(
        str(args.out),
        fourcc,
        fps,
        (args.width, args.height),
    )
    if not writer.isOpened():
        raise SystemExit(f"Impossibile aprire writer: {args.out}")

    canvas = np.zeros((args.height, args.width, 3), dtype=np.uint8)
    for frame_idx in frame_ids:
        rows = by_frame[frame_idx]
        ts = float(rows[0]["timestamp_sec"]) if rows else 0.0
        draw_frame(canvas, rows, xmin, xmax, ymin, ymax, frame_idx, ts)
        writer.write(canvas)

    writer.release()
    print(f"Scritto {args.out} ({len(frame_ids)} frame, {fps:.2f} fps)")


if __name__ == "__main__":
    main()
