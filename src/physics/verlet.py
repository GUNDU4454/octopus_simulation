"""
verlet.py

2D Verlet integration for the top-down seabed plane.

Key design decisions:
  - NO gravity.  Gravity is irrelevant for top-down crawling.
  - Damping models water resistance + seabed friction.
  - All positions/velocities live in the XY plane only.
"""

import numpy as np


class Point:
    """
    A single point mass on the 2D seabed plane.

    Velocity is implicit: v ≈ (pos - prev) / dt
    Forces accumulate in self.acc and are cleared after each integrate().
    """

    def __init__(self, x: float, y: float,
                 mass: float = 1.0, fixed: bool = False):
        self.pos  = np.array([x, y], dtype=np.float64)
        self.prev = np.array([x, y], dtype=np.float64)
        self.acc  = np.zeros(2,      dtype=np.float64)
        self.mass  = mass
        self.fixed = fixed

    def integrate(self, dt: float, damping: float = 0.90):
        """Verlet step.  damping < 1 dissipates energy (friction / drag)."""
        if self.fixed:
            self.prev = self.pos.copy()
            self.acc[:] = 0
            return
        vel        = (self.pos - self.prev) * damping
        self.prev  = self.pos.copy()
        self.pos  += vel + self.acc * (dt * dt)
        self.acc[:] = 0

    def apply_force(self, f: np.ndarray):
        if not self.fixed:
            self.acc += f / self.mass

    def teleport(self, x: float, y: float):
        """Move without changing implicit velocity."""
        delta     = self.pos - self.prev
        self.pos  = np.array([x, y], dtype=np.float64)
        self.prev = self.pos - delta

    def pin(self, x: float, y: float):
        """Fix at position (no velocity preserved)."""
        self.pos[:]  = [x, y]
        self.prev[:] = [x, y]
        self.fixed   = True

    def unpin(self):
        self.fixed = False

    @property
    def vel(self) -> np.ndarray:
        return self.pos - self.prev

    @property
    def speed(self) -> float:
        return float(np.linalg.norm(self.vel))


def constrain_distance(a: Point, b: Point,
                       rest: float, stiffness: float = 1.0):
    """
    Push/pull two points to satisfy a distance constraint.
    Moves both equally unless one is fixed.
    """
    d    = b.pos - a.pos
    dist = np.linalg.norm(d)
    if dist < 1e-9:
        return
    correction = (dist - rest) / dist * stiffness
    half = d * correction * 0.5

    if   a.fixed and not b.fixed: b.pos -= half * 2
    elif b.fixed and not a.fixed: a.pos += half * 2
    elif not a.fixed and not b.fixed:
        a.pos += half
        b.pos -= half
