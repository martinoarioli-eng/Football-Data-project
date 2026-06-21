"""Canonical player IDs + stable fine tactical roles from unified.csv (9v9)."""

import argparse
import csv
import logging
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent

CSV_OUT_HEADER = [
    "frame",
    "timestamp_sec",
    "type",
    "player_id",
    "canonical_id",
    "team_id",
    "role",
    "x_center",
    "y_center",
    "width",
    "height",
    "confidence",
    "ball_predicted",
]

TEAM_COLORS = {
    0: (40, 120, 255),    # team 0 -> orange/red kit
    1: (255, 120, 0),     # team 1 -> dark/blue kit
    -1: (160, 160, 160),  # referee
}
GK_COLOR = (0, 220, 255)  # goalkeeper -> yellow
BALL_COLOR = (255, 0, 255)


@dataclass
class Track:
    track_id: int
    team: int
    frames: list[int] = field(default_factory=list)
    xs: list[float] = field(default_factory=list)
    ys: list[float] = field(default_factory=list)

    @property
    def start(self) -> int:
        return self.frames[0]

    @property
    def end(self) -> int:
        return self.frames[-1]

    @property
    def start_pos(self) -> tuple[float, float]:
        return self.xs[0], self.ys[0]

    @property
    def end_pos(self) -> tuple[float, float]:
        return self.xs[-1], self.ys[-1]

    @property
    def n(self) -> int:
        return len(self.frames)

    @property
    def mean_x(self) -> float:
        return float(np.mean(self.xs))

    @property
    def mean_y(self) -> float:
        return float(np.mean(self.ys))


def _load(csv_path: str) -> tuple[list[dict], list[Track], float]:
    rows = list(csv.DictReader(open(csv_path, newline="")))
    by_track: dict[int, Track] = {}
    max_x = 1.0
    for r in rows:
        if r["type"] != "player":
            continue
        tid = int(r["player_id"])
        team = int(r["team_id"]) if r["team_id"] not in ("", None) else 0
        x, y = float(r["x_center"]), float(r["y_center"])
        max_x = max(max_x, x)
        t = by_track.get(tid)
        if t is None:
            t = by_track[tid] = Track(track_id=tid, team=team)
        t.frames.append(int(r["frame"]))
        t.xs.append(x)
        t.ys.append(y)
    for t in by_track.values():
        order = np.argsort(t.frames)
        t.frames = [t.frames[i] for i in order]
        t.xs = [t.xs[i] for i in order]
        t.ys = [t.ys[i] for i in order]
    return rows, list(by_track.values()), max_x


def _majority_team(rows: list[dict]) -> dict[int, int]:
    votes: dict[int, list[int]] = defaultdict(list)
    for r in rows:
        if r["type"] != "player" or r["team_id"] in ("", None):
            continue
        votes[int(r["player_id"])].append(int(r["team_id"]))
    return {tid: max(set(v), key=v.count) for tid, v in votes.items()}


def _referees(tracks: list[Track]) -> set[int]:
    """All team -1 tracks are non-players (referee / sideline staff)."""
    return {t.track_id for t in tracks if t.team == -1}


def _attack_signs(rows: list[dict]) -> dict[int, int]:
    """Deduce attack direction by comparing per-frame mean x of the two teams.

    The team more often on the left half defends left, so it attacks +x (sign +1);
    the other attacks -x. Signs are opposite by construction.
    """
    per_frame: dict[int, dict[int, list[float]]] = defaultdict(lambda: defaultdict(list))
    for r in rows:
        if r["type"] != "player" or r["team_id"] in ("", None):
            continue
        team = int(r["team_id"])
        if team not in (0, 1):
            continue
        per_frame[int(r["frame"])][team].append(float(r["x_center"]))
    team0_left = 0
    team1_left = 0
    for teams in per_frame.values():
        if 0 in teams and 1 in teams:
            if np.mean(teams[0]) < np.mean(teams[1]):
                team0_left += 1
            else:
                team1_left += 1
    if team0_left >= team1_left:
        return {0: 1, 1: -1}
    return {0: -1, 1: 1}


def _merge_tracklets(
    tracks: list[Track], *, max_gap: int, max_dist: float, min_frames: int
) -> dict[int, int]:
    """Greedy chaining of non-overlapping same-team tracks. Returns track_id -> canonical_id."""
    by_team: dict[int, list[Track]] = defaultdict(list)
    for t in tracks:
        if t.n >= min_frames:
            by_team[t.team].append(t)

    canonical: dict[int, int] = {}
    next_cid = 1
    for team, team_tracks in by_team.items():
        team_tracks.sort(key=lambda t: t.start)
        chains: list[dict] = []
        for t in team_tracks:
            best, best_d = None, max_dist
            for ch in chains:
                if ch["end"] >= t.start:
                    continue
                if t.start - ch["end"] > max_gap:
                    continue
                ex, ey = ch["end_pos"]
                sx, sy = t.start_pos
                d = float(np.hypot(sx - ex, sy - ey))
                if d < best_d:
                    best, best_d = ch, d
            if best is None:
                chains.append(
                    {"cid": next_cid, "end": t.end, "end_pos": t.end_pos, "members": [t.track_id]}
                )
                next_cid += 1
            else:
                best["end"] = t.end
                best["end_pos"] = t.end_pos
                best["members"].append(t.track_id)
        for ch in chains:
            for tid in ch["members"]:
                canonical[tid] = ch["cid"]
    return canonical


def _fine_roles(n: int, line: str) -> list[str]:
    """Return left->right role labels for a line of n players."""
    if line == "DEF":
        base = {1: ["DC"], 2: ["DC sx", "DC dx"], 3: ["TS", "DC", "TD"],
                4: ["TS", "DC sx", "DC dx", "TD"]}
    elif line == "MID":
        base = {1: ["MED"], 2: ["MED sx", "MED dx"], 3: ["MZ sx", "MED", "MZ dx"],
                4: ["MZ sx", "MED sx", "MED dx", "MZ dx"]}
    else:  # ATT
        base = {1: ["PC"], 2: ["PC sx", "PC dx"], 3: ["AS", "PC", "AD"],
                4: ["AS", "PC sx", "PC dx", "AD"]}
    if n in base:
        return base[n]
    return [f"{line[0]}{i+1}" for i in range(n)]


def _assign_roles(
    outfield: list[Track],
    canonical: dict[int, int],
    attack_sign: dict[int, int],
    *,
    gk_gap: float,
    max_x: float,
) -> tuple[dict[int, str], set[int]]:
    """Per team: detect GK (deepest isolated), cluster rest into 3 lines, assign fine roles.

    Returns (role_by_canonical_id, set_of_gk_canonical_ids).
    """
    cid_team: dict[int, int] = {}
    cid_x: dict[int, list[float]] = defaultdict(list)
    cid_y: dict[int, list[float]] = defaultdict(list)
    for t in outfield:
        cid = canonical.get(t.track_id)
        if cid is None:
            continue
        cid_team[cid] = t.team
        cid_x[cid].extend(t.xs)
        cid_y[cid].extend(t.ys)

    roles: dict[int, str] = {}
    gk_ids: set[int] = set()
    by_team_cids: dict[int, list[int]] = defaultdict(list)
    for cid, team in cid_team.items():
        by_team_cids[team].append(cid)

    for team, cids in by_team_cids.items():
        sign = attack_sign.get(team, 1)
        depth = {cid: sign * float(np.mean(cid_x[cid])) for cid in cids}
        width = {cid: float(np.mean(cid_y[cid])) for cid in cids}
        ordered = sorted(cids, key=lambda c: depth[c])

        # GK = deepest player if clearly isolated behind the next one
        if len(ordered) >= 3 and (depth[ordered[1]] - depth[ordered[0]]) >= gk_gap * max_x:
            gk = ordered.pop(0)
            gk_ids.add(gk)
            roles[gk] = "POR"

        n = len(ordered)
        n_lines = min(3, n)
        if n_lines == 0:
            continue
        depths = np.array([depth[c] for c in ordered])
        edges = np.quantile(depths, np.linspace(0, 1, n_lines + 1))
        line_names = ["DEF", "MID", "ATT"][:n_lines] if n_lines == 3 else (
            ["DEF", "ATT"] if n_lines == 2 else ["DEF"]
        )
        for li in range(n_lines):
            lo, hi = edges[li], edges[li + 1]
            if li < n_lines - 1:
                members = [c for c in ordered if lo <= depth[c] < hi and c not in roles]
            else:
                members = [c for c in ordered if depth[c] >= lo and c not in roles]
            members.sort(key=lambda c: width[c])
            labels = _fine_roles(len(members), line_names[li])
            for c, lab in zip(members, labels):
                roles[c] = lab
    return roles, gk_ids


def _write_csv(
    rows: list[dict],
    canonical: dict[int, int],
    roles: dict[int, str],
    referees: set[int],
    out_path: str,
) -> None:
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(CSV_OUT_HEADER)
        for r in rows:
            cid, role, team = "", "", r["team_id"]
            if r["type"] == "player":
                tid = int(r["player_id"])
                if tid in referees:
                    role, team = "ARB", "-1"
                else:
                    cid = canonical.get(tid, "")
                    role = roles.get(cid, "") if cid != "" else ""
            w.writerow([
                r["frame"], r["timestamp_sec"], r["type"], r["player_id"], cid, team, role,
                r["x_center"], r["y_center"], r["width"], r["height"], r["confidence"],
                r.get("ball_predicted", ""),
            ])


def _draw(rows: list[dict], video: str, frame_idx: int, out_path: str) -> None:
    cap = cv2.VideoCapture(video)
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
    ok, frame = cap.read()
    cap.release()
    if not ok:
        raise RuntimeError(f"Cannot read frame {frame_idx}")

    for r in rows:
        if int(r["frame"]) != frame_idx:
            continue
        x, y = float(r["x_center"]), float(r["y_center"])
        w, h = float(r["width"]), float(r["height"])
        x1, y1 = int(x - w / 2), int(y - h / 2)
        x2, y2 = int(x + w / 2), int(y + h / 2)

        if r["type"] == "ball":
            label = "ball" if r.get("ball_predicted") != "1" else "ball?"
            cv2.rectangle(frame, (x1, y1), (x2, y2), BALL_COLOR, 2)
            cv2.putText(frame, label, (x1, max(y1 - 4, 12)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, BALL_COLOR, 1, cv2.LINE_AA)
            continue

        role = r["role"]
        team = int(r["team_id"]) if r["team_id"] not in ("", None) else 0
        if role == "POR":
            color = GK_COLOR
        elif role == "ARB":
            color = TEAM_COLORS[-1]
        else:
            color = TEAM_COLORS.get(team, (160, 160, 160))
        cid = r["canonical_id"]
        label = f"{cid}:{role}" if role else f"{cid}"
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
        cv2.rectangle(frame, (x1, y1 - th - 6), (x1 + tw + 4, y1), color, -1)
        cv2.putText(frame, label, (x1 + 2, y1 - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 1, cv2.LINE_AA)

    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(out_path, frame)


def run(args) -> None:
    rows, tracks, max_x = _load(args.csv)
    team_by_track = _majority_team(rows)
    for t in tracks:
        if t.track_id in team_by_track:
            t.team = team_by_track[t.track_id]

    referees = _referees(tracks)
    outfield = [t for t in tracks if t.team in (0, 1)]

    canonical = _merge_tracklets(
        outfield, max_gap=args.max_gap, max_dist=args.max_dist, min_frames=args.min_frames
    )

    attack_sign = _attack_signs(rows)
    roles, gk_ids = _assign_roles(
        outfield, canonical, attack_sign, gk_gap=args.gk_gap, max_x=max_x
    )

    n_can = len(set(canonical.values()))
    logging.info("Canonical outfield ids: %d (team0+team1)", n_can)
    logging.info("Goalkeepers: %d, referees: %d", len(gk_ids), len(referees))
    logging.info("Attack sign: team0=%+d team1=%+d", attack_sign.get(0, 0), attack_sign.get(1, 0))

    out_csv = args.output
    if not Path(out_csv).is_absolute():
        out_csv = str(ROOT / out_csv)
    _write_csv(rows, canonical, roles, referees, out_csv)
    logging.info("Wrote %s", out_csv)

    if args.debug_frame:
        out_rows = list(csv.DictReader(open(out_csv, newline="")))
        for frame_idx in args.debug_frame:
            out = args.debug_out or str(ROOT / f"output/debug_roles_{frame_idx}.jpg")
            _draw(out_rows, args.video, frame_idx, out)
            logging.info("Debug image saved: %s", out)


def main() -> None:
    p = argparse.ArgumentParser(
        description="Canonical IDs + stable fine tactical roles (9v9)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--csv", default="output/unified.csv")
    p.add_argument("--video", default="data/test_10s.mp4")
    p.add_argument("--output", default="output/unified_roles.csv")
    p.add_argument("--max-gap", type=int, default=90, help="Max frame gap to merge tracklets")
    p.add_argument("--max-dist", type=float, default=250.0, help="Max px gap to merge tracklets")
    p.add_argument("--min-frames", type=int, default=20, help="Drop tracks shorter than this")
    p.add_argument("--gk-gap", type=float, default=0.12,
                   help="Min depth gap (fraction of frame width) to flag deepest player as GK")
    p.add_argument(
        "--debug-frame",
        type=int,
        action="append",
        help="Save annotated JPG for frame index (repeatable, 0-based)",
    )
    p.add_argument("--debug-out", default=None)
    args = p.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    run(args)


if __name__ == "__main__":
    main()
