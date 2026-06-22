"""
target_movement.py  —  TargetMovementSystem

Holds the goal the octopus is crawling toward (set by a mouse click) and
answers geometric questions about it: direction, angle, distance, arrival.

IMPORTANT: this system never moves the body.  It only describes *where* the
body should try to go.  All actual motion comes from tentacle forces, so this
class deliberately has no write access to the body transform.
"""

import math
import numpy as np


class TargetMovementSystem:

    ARRIVE_RADIUS = 26.0   # px: within this of the target, the goal is reached

    def __init__(self):
        self._target: np.ndarray | None = None

    # -- target management -------------------------------------------------

    def set_target(self, x: float, y: float):
        self._target = np.array([float(x), float(y)], dtype=np.float64)

    def clear(self):
        self._target = None

    @property
    def has_target(self) -> bool:
        return self._target is not None

    @property
    def target(self) -> np.ndarray | None:
        return None if self._target is None else self._target.copy()

    # -- per-frame -------------------------------------------------------

    def update(self, body_pos: np.ndarray):
        """Drop the target once the body has arrived."""
        if self._target is None:
            return
        if float(np.linalg.norm(self._target - body_pos)) <= self.ARRIVE_RADIUS:
            self._target = None

    # -- geometry queries -------------------------------------------------

    def direction_from(self, body_pos: np.ndarray) -> np.ndarray | None:
        """Unit vector body→target, or None if no/at target."""
        if self._target is None:
            return None
        d = self._target - body_pos
        dist = float(np.linalg.norm(d))
        if dist < 1e-6:
            return None
        return d / dist

    def angle_from(self, body_pos: np.ndarray) -> float | None:
        d = self.direction_from(body_pos)
        return None if d is None else math.atan2(d[1], d[0])

    def distance_from(self, body_pos: np.ndarray) -> float:
        if self._target is None:
            return 0.0
        return float(np.linalg.norm(self._target - body_pos))
