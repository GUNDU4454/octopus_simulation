"""
body_locomotion.py  —  BodyLocomotionController + BodyDrive

The central locomotion brain.  Where TargetMovementSystem only stores *where*
the goal is, this turns that goal into a single body-level command each frame:

  * the direction the whole body should travel       (target direction)
  * how far it still has to go                        (distance)
  * the orientation it should be facing               (desired heading)
  * which way and how hard it should rotate           (turn sign / gain)
  * how much forward effort vs. stabilisation to use  (distance-scaled gains)

This is the "central controller that calculates target direction, distance,
desired body orientation and required pulling/pushing forces" — expressed about
the BODY as a whole rather than any single arm.

IMPORTANT: it never moves the body transform.  It only produces a `BodyDrive`
that the per-arm TentacleControllers read to choose roles and force levels, so
every bit of real motion still emerges from tentacle forces on the body.

The angle convention matches the physics integrator exactly: `body.angle`
advances with `angular_vel`, which advances with torque = r × F (the 2D scalar
cross product, +z out of the screen).  So a *positive* `heading_error`
(target_angle ahead of body.angle in the CCW sense) needs *positive* net torque,
and `turn_sign` is fed straight through to bias the arms.  No handedness fixups.
"""

from dataclasses import dataclass
import math
import numpy as np


def wrap(angle: float) -> float:
    """Wrap an angle to [-pi, pi]."""
    while angle >  math.pi:
        angle -= 2 * math.pi
    while angle < -math.pi:
        angle += 2 * math.pi
    return angle


@dataclass
class BodyDrive:
    """One frame's body-level locomotion command (read-only goal, never a move).

    Fields with no target collapse to a benign "idle" drive: zero direction,
    zero turn, and full stabilize so any leftover motion settles.
    """
    has_target:     bool
    direction:      np.ndarray   # unit vector body → target (zeros if none)
    distance:       float        # px to the target
    target_angle:   float        # atan2(direction) — desired heading
    heading_error:  float        # wrap(target_angle - body.angle), [-pi, pi]
    turn_sign:      float        # -1 / 0 / +1 desired net-torque direction
    turn_gain:      float        # [0,1] how hard to rotate (0 inside deadzone)
    drive_gain:     float        # [0,1] forward effort, strong far / gentle near
    stabilize_gain: float        # [0,1] damping emphasis, gentle far / strong near


class BodyLocomotionController:

    # Distance band over which forward effort eases off as the body arrives.
    # NEAR_DIST sits just outside the arrival radius so the body keeps real pull
    # until it has essentially arrived (too gentle too early stalls on friction).
    NEAR_DIST = 35.0    # px: at/under this, small corrective strokes
    FAR_DIST  = 200.0   # px: at/over this, full-strength crawling

    # Heading band: under the deadzone the body is "facing" the target and we
    # stop steering (prevents jitter); at HEADING_FULL we turn at full effort.
    HEADING_DEADZONE = 0.10   # rad (~6°)
    HEADING_FULL     = 1.20   # rad (~69°)

    def compute(self, body, target_system) -> BodyDrive:
        """Build this frame's BodyDrive from the body pose and the target."""
        direction = target_system.direction_from(body.pos)

        # No target (or sitting on it): idle drive — face current heading, full
        # stabilise so the body damps out instead of drifting.
        if direction is None:
            return BodyDrive(
                has_target=False,
                direction=np.zeros(2),
                distance=0.0,
                target_angle=body.angle,
                heading_error=0.0,
                turn_sign=0.0,
                turn_gain=0.0,
                drive_gain=0.0,
                stabilize_gain=1.0,
            )

        distance     = target_system.distance_from(body.pos)
        target_angle = math.atan2(direction[1], direction[0])
        heading_error = wrap(target_angle - body.angle)

        # Desired rotation: nothing inside the deadzone, ramping to full effort.
        abs_err = abs(heading_error)
        if abs_err < self.HEADING_DEADZONE:
            turn_sign = 0.0
            turn_gain = 0.0
        else:
            turn_sign = 1.0 if heading_error > 0.0 else -1.0
            turn_gain = min(1.0, (abs_err - self.HEADING_DEADZONE) /
                                  (self.HEADING_FULL - self.HEADING_DEADZONE))

        # Distance-based behaviour: far → big extensions / strong pulls,
        # near → small corrections and more stabilisation (less overshoot).
        span = self.FAR_DIST - self.NEAR_DIST
        drive_gain     = float(np.clip((distance - self.NEAR_DIST) / span, 0.35, 1.0))
        stabilize_gain = float(np.clip(1.0 - (distance - self.NEAR_DIST) / span, 0.25, 1.0))

        return BodyDrive(
            has_target=True,
            direction=direction,
            distance=distance,
            target_angle=target_angle,
            heading_error=heading_error,
            turn_sign=turn_sign,
            turn_gain=turn_gain,
            drive_gain=drive_gain,
            stabilize_gain=stabilize_gain,
        )
