import argparse
import csv
import logging
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent

CSV_OUT_HEADER = [
    "frame",
    "timestamp_sec",
    "type",
    "player_id",
    "x_center",
    "y_center",
    "width",
    "height",
    "confidence",
    "team_id",
]

TEAM_COLORS = {
    0: (255, 120, 0),
    1: (0, 100, 255),
    -1: (160, 160, 160),
}


def _ensure_parent(filepath: str) -> None:
    Path(filepath).parent.mkdir(parents=True, exist_ok=True)


def _xywh_to_xyxy(x: float, y: float, w: float, h: float) -> tuple[int, int, int, int]:
    x1 = int(round(x - w / 2))
    y1 = int(round(y - h / 2))
    x2 = int(round(x + w / 2))
    y2 = int(round(y + h / 2))
    return x1, y1, x2, y2


def _torso_bgr_mean(frame: np.ndarray, x: float, y: float, w: float, h: float) -> tuple[float, float, float] | None:
    h_img, w_img = frame.shape[:2]
    x1, y1, x2, y2 = _xywh_to_xyxy(x, y, w, h)
    x1, x2 = max(0, x1), min(w_img, x2)
    y1, y2 = max(0, y1), min(h_img, y2)
    if x2 <= x1 or y2 <= y1:
        return None
    crop = frame[y1:y2, x1:x2]
    torso_h = max(1, (y2 - y1) // 2)
    torso = crop[:torso_h, :]
    cw = torso.shape[1]
    if cw < 2:
        return None
    torso = torso[:, cw // 4 : 3 * cw // 4]
    b, g, r = torso.reshape(-1, 3).mean(axis=0)
    return float(r), float(g), float(b)


def _is_staff(r: float, g: float, b: float) -> bool:
    if g > r + 18 and g > 70:
        return True
    if r > 75 and g > 75 and b < 45:
        return True
    return False


def _fit_teams(
    video_path: str,
    rows: list[dict],
    *,
    sample_stride: int,
    max_samples_per_player: int,
) -> dict[int, int]:
    by_player: dict[int, list[dict]] = defaultdict(list)
    for row in rows:
        if row["type"] != "player":
            continue
        by_player[int(row["player_id"])].append(row)

    field_feats: list[list[float]] = []
    field_owner: list[int] = []
    staff_players: set[int] = set()

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")

    frames_cache: dict[int, np.ndarray] = {}
    try:
        for player_id, player_rows in by_player.items():
            picked = player_rows[::sample_stride][:max_samples_per_player]
            staff_votes = 0
            field_votes = 0
            for row in picked:
                frame_idx = int(row["frame"])
                if frame_idx not in frames_cache:
                    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
                    ok, frame = cap.read()
                    frames_cache[frame_idx] = frame if ok else None
                frame = frames_cache.get(frame_idx)
                if frame is None:
                    continue
                rgb = _torso_bgr_mean(
                    frame,
                    float(row["x_center"]),
                    float(row["y_center"]),
                    float(row["width"]),
                    float(row["height"]),
                )
                if rgb is None:
                    continue
                r, g, b = rgb
                if _is_staff(r, g, b):
                    staff_votes += 1
                    continue
                field_votes += 1
                field_feats.append([r - g, r - b])
                field_owner.append(player_id)

            if staff_votes > field_votes:
                staff_players.add(player_id)
    finally:
        cap.release()

    team_by_player: dict[int, int] = {pid: -1 for pid in staff_players}

    if len(field_feats) < 4:
        logging.warning("Not enough field-player samples; marking unknown as team 0")
        for pid in by_player:
            team_by_player.setdefault(pid, 0)
        return team_by_player

    data = np.array(field_feats, dtype=np.float32)
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 20, 0.5)
    _compact, labels, centers = cv2.kmeans(
        data, 2, None, criteria, 5, cv2.KMEANS_PP_CENTERS
    )
    labels = labels.flatten()
    red_cluster = int(0 if centers[0, 0] > centers[1, 0] else 1)

    votes: dict[int, list[int]] = defaultdict(list)
    for player_id, label in zip(field_owner, labels):
        team = 0 if label == red_cluster else 1
        votes[player_id].append(team)

    for player_id, teams in votes.items():
        if player_id in staff_players:
            continue
        team_by_player[player_id] = max(set(teams), key=teams.count)

    for player_id in by_player:
        team_by_player.setdefault(player_id, 0)

    n0 = sum(1 for t in team_by_player.values() if t == 0)
    n1 = sum(1 for t in team_by_player.values() if t == 1)
    n_staff = sum(1 for t in team_by_player.values() if t == -1)
    logging.info("Teams: red=%d dark=%d staff=%d", n0, n1, n_staff)
    return team_by_player


def _load_rows(csv_path: str) -> list[dict]:
    with open(csv_path, newline="") as f:
        return list(csv.DictReader(f))


def _write_csv(rows: list[dict], team_by_player: dict[int, int], out_path: str) -> None:
    _ensure_parent(out_path)
    with open(out_path, "w", newline="") as f:
        writer = csv.writer(f)
        extra_cols = []
        if rows and "ball_predicted" in rows[0]:
            extra_cols = ["ball_predicted"]
        writer.writerow(CSV_OUT_HEADER + extra_cols)
        for row in rows:
            team_id = ""
            if row["type"] == "player":
                team_id = team_by_player.get(int(row["player_id"]), 0)
            out_row = [
                row["frame"],
                row["timestamp_sec"],
                row["type"],
                row["player_id"],
                row["x_center"],
                row["y_center"],
                row["width"],
                row["height"],
                row["confidence"],
                team_id,
            ]
            for col in extra_cols:
                out_row.append(row.get(col, ""))
            writer.writerow(out_row)


def _save_team_debug_frame(
    video_path: str,
    rows: list[dict],
    team_by_player: dict[int, int],
    frame_idx: int,
    out_path: str,
) -> None:
    cap = cv2.VideoCapture(video_path)
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
    ok, frame = cap.read()
    cap.release()
    if not ok:
        raise RuntimeError(f"Cannot read frame {frame_idx} from {video_path}")

    for row in rows:
        if row["type"] != "player" or int(row["frame"]) != frame_idx:
            continue
        x, y, w, h = (
            float(row["x_center"]),
            float(row["y_center"]),
            float(row["width"]),
            float(row["height"]),
        )
        pid = int(row["player_id"])
        team_id = team_by_player.get(pid, 0)
        color = TEAM_COLORS.get(team_id, (160, 160, 160))
        x1, y1, x2, y2 = _xywh_to_xyxy(x, y, w, h)
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        if team_id == -1:
            label = f"id:{pid} staff"
        else:
            label = f"id:{pid} team:{team_id}"
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(frame, (x1, y1 - th - 6), (x1 + tw + 4, y1), color, -1)
        cv2.putText(
            frame,
            label,
            (x1 + 2, y1 - 4),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )

    _ensure_parent(out_path)
    cv2.imwrite(out_path, frame)


def main() -> None:
    parser = argparse.ArgumentParser(description="Assign team_id from jersey colors (red vs dark).")
    parser.add_argument("--video", required=True)
    parser.add_argument("--csv", default="output/tracking_raw.csv")
    parser.add_argument("--out-csv", default="output/tracking_teams.csv")
    parser.add_argument("--sample-stride", type=int, default=15)
    parser.add_argument("--max-samples-per-player", type=int, default=20)
    parser.add_argument("--debug-frame", type=int, default=None)
    parser.add_argument("--debug-out", default=None)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO)

    csv_path = args.csv
    if not Path(csv_path).is_absolute():
        p = Path(csv_path)
        csv_path = str(ROOT / p) if not p.exists() else str(p.resolve())

    rows = _load_rows(csv_path)
    team_by_player = _fit_teams(
        args.video,
        rows,
        sample_stride=args.sample_stride,
        max_samples_per_player=args.max_samples_per_player,
    )

    out_csv = args.out_csv
    if not Path(out_csv).is_absolute():
        out_csv = str(ROOT / out_csv)
    _write_csv(rows, team_by_player, out_csv)
    logging.info("Wrote %s", out_csv)

    if args.debug_frame is not None:
        debug_out = args.debug_out or str(
            ROOT / f"output/debug_frame_{args.debug_frame}_teams.jpg"
        )
        _save_team_debug_frame(
            args.video, rows, team_by_player, args.debug_frame, debug_out
        )
        logging.info("Team debug image saved: %s", debug_out)


if __name__ == "__main__":
    main()
