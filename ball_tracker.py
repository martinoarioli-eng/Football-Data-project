"""Temporal ball tracker: one ball per frame from SAHI detections."""

from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass(frozen=True)
class BallCandidate:
    x: float
    y: float
    w: float
    h: float
    conf: float


@dataclass(frozen=True)
class BallObservation:
    x: float
    y: float
    w: float
    h: float
    conf: float
    predicted: bool


class BallTracker:
    """Pick one ball detection per frame using motion + proximity."""

    def __init__(
        self,
        *,
        max_dist: float = 150.0,
        max_size: float = 20.0,
        max_gap: int = 5,
        min_conf_init: float = 0.57,
        player_proximity: float = 150.0,
    ):
        self.max_dist = max_dist
        self.max_size = max_size
        self.max_gap = max_gap
        self.min_conf_init = min_conf_init
        self.player_proximity = player_proximity
        self._x = 0.0
        self._y = 0.0
        self._w = 0.0
        self._h = 0.0
        self._conf = 0.0
        self._vx = 0.0
        self._vy = 0.0
        self._active = False
        self._missed = 0

    def reset(self) -> None:
        self._active = False
        self._missed = 0
        self._vx = 0.0
        self._vy = 0.0

    def _size_ok(self, c: BallCandidate) -> bool:
        return c.w <= self.max_size and c.h <= self.max_size

    def _near_players(
        self, c: BallCandidate, players: list[tuple[float, float]]
    ) -> bool:
        if not players:
            return True
        for px, py in players:
            if math.hypot(c.x - px, c.y - py) <= self.player_proximity:
                return True
        return False

    def _predict(self) -> tuple[float, float]:
        return self._x + self._vx, self._y + self._vy

    def _score(
        self, c: BallCandidate, pred_x: float, pred_y: float
    ) -> float:
        dist = math.hypot(c.x - pred_x, c.y - pred_y)
        if dist > self.max_dist:
            return -1.0
        return c.conf * math.exp(-dist / self.max_dist)

    def _commit(self, c: BallCandidate, predicted: bool) -> BallObservation:
        if self._active:
            self._vx = c.x - self._x
            self._vy = c.y - self._y
        self._x, self._y = c.x, c.y
        self._w, self._h = c.w, c.h
        self._conf = c.conf
        self._active = True
        self._missed = 0
        return BallObservation(
            x=c.x, y=c.y, w=c.w, h=c.h, conf=c.conf, predicted=predicted
        )

    def _predicted_observation(self) -> BallObservation:
        pred_x, pred_y = self._predict()
        self._x, self._y = pred_x, pred_y
        self._missed += 1
        return BallObservation(
            x=pred_x,
            y=pred_y,
            w=self._w,
            h=self._h,
            conf=self._conf,
            predicted=True,
        )

    def update(
        self,
        candidates: list[BallCandidate],
        players: list[tuple[float, float]] | None = None,
    ) -> BallObservation | None:
        players = players or []
        valid = [c for c in candidates if self._size_ok(c)]
        if not valid:
            if self._active and self._missed < self.max_gap:
                return self._predicted_observation()
            self.reset()
            return None

        if not self._active:
            near = [
                c
                for c in valid
                if c.conf >= self.min_conf_init and self._near_players(c, players)
            ]
            pool = near or [c for c in valid if c.conf >= self.min_conf_init]
            if not pool:
                return None
            best = max(pool, key=lambda c: c.conf)
            return self._commit(best, predicted=False)

        pred_x, pred_y = self._predict()
        scored = [(self._score(c, pred_x, pred_y), c) for c in valid]
        scored = [(s, c) for s, c in scored if s >= 0]
        if scored:
            _, best = max(scored, key=lambda item: item[0])
            return self._commit(best, predicted=False)

        far_strong = [
            c
            for c in valid
            if c.conf >= self.min_conf_init
            and math.hypot(c.x - self._x, c.y - self._y) > self.max_dist * 2
            and self._near_players(c, players)
        ]
        if far_strong:
            best = max(far_strong, key=lambda c: c.conf)
            return self._commit(best, predicted=False)

        if self._missed < self.max_gap:
            return self._predicted_observation()

        self.reset()
        return self.update(valid, players)
