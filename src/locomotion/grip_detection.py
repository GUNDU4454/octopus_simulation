"""
grip_detection.py  —  GripDetection

Answers "can a tentacle tip grip / push at this world point, and how well?"

The current seabed is a uniform flat plane, so every in-bounds point affords a
full grip (an octopus crawling on open sand).  This class exists so that the
"grip anywhere" assumption lives in exactly one place.  To introduce rocks or
obstacles later, give grip_quality()/find_anchor() real geometry to test
against — and nothing in the controllers or manager needs to change.
"""

import numpy as np


class GripDetection:

    def __init__(self, width: int, height: int, margin: float = 8.0):
        self.width  = width
        self.height = height
        self.margin = margin

    def in_bounds(self, point: np.ndarray) -> bool:
        x, y = float(point[0]), float(point[1])
        return (self.margin <= x <= self.width  - self.margin and
                self.margin <= y <= self.height - self.margin)

    def grip_quality(self, point: np.ndarray) -> float:
        """Grip affordance in [0, 1].  Flat seabed: 1 in-bounds, 0 off-map."""
        return 1.0 if self.in_bounds(point) else 0.0

    def has_grip(self, point: np.ndarray) -> bool:
        return self.grip_quality(point) > 0.0
