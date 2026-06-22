"""
locomotion_manager.py  —  LocomotionManager

Top-level coordinator that drives the octopus toward a clicked target.

Each frame it:
  1. lets the TargetMovementSystem retire the goal once the body arrives
  2. computes the target direction/angle/distance from the body
  3. asks every arm's TentacleController to pick a role and write its commands

It writes only the arms' command fields — never the body transform — so the
body still moves purely from the resulting tentacle forces.  This is the
component a reinforcement-learning policy would replace: the policy would set
the same per-arm command fields, and the physics below would be untouched.
"""

import numpy as np

from src.locomotion.target_movement import TargetMovementSystem
from src.locomotion.grip_detection import GripDetection
from src.locomotion.tentacle_controller import TentacleController, Role
from src.locomotion.body_locomotion import BodyLocomotionController, BodyDrive


class LocomotionManager:

    # PD steering on body heading.  The arms turn the body by tensioning
    # differentially against their grips; rather than hope that emerges from
    # per-arm force scaling (it is far too weak to overcome the crawl's own
    # incidental torque), we close the loop: torque ∝ heading error, damped by
    # current spin so it eases in and settles without snapping or oscillating.
    STEER_KP   = 16.0   # torque per radian of heading error (× inertia)
    STEER_KD   = 7.0    # damping per rad/s of spin        (× inertia)
    STEER_MAX  = 20000.0 # clamp so a big turn can't fling the body

    def __init__(self, octopus, width: int, height: int):
        self.octopus = octopus
        self.grip    = GripDetection(width, height)
        self.targets = TargetMovementSystem()
        self.body    = BodyLocomotionController()
        self.controllers = [
            TentacleController(arm, self.grip) for arm in octopus.arms
        ]
        self.enabled = True

        # Last body-level command computed (read by the debug overlay).
        self.drive: BodyDrive = self.body.compute(octopus, self.targets)

    # -- target control --------------------------------------------------

    def set_target(self, x: float, y: float):
        self.targets.set_target(x, y)

    def clear_target(self):
        self.targets.clear()

    @property
    def target(self) -> np.ndarray | None:
        return self.targets.target

    @property
    def target_angle(self):
        return self.targets.angle_from(self.octopus.pos)

    # -- per-frame -------------------------------------------------------

    def update(self, dt: float):
        oct_ = self.octopus
        self.targets.update(oct_.pos)
        if not self.enabled:
            return

        # Central controller turns the raw target into one body-level command;
        # each arm then chooses its role and force level from that command.
        self.drive = self.body.compute(oct_, self.targets)

        # Closed-loop heading control: ask the body to generate a steering torque
        # that drives heading_error → 0 (eased by spin damping).  Octopus.update
        # gates it by how many arms are gripping, so it is still tentacle-borne.
        inertia = oct_.body_physics.INERTIA
        if self.drive.has_target:
            steer = inertia * (self.STEER_KP * self.drive.heading_error
                               - self.STEER_KD * oct_.angular_vel)
        else:
            steer = -inertia * self.STEER_KD * oct_.angular_vel   # settle to rest
        oct_.steer_torque = float(np.clip(steer, -self.STEER_MAX, self.STEER_MAX))

        for ctrl in self.controllers:
            ctrl.decide(oct_, self.drive, dt)

    # -- debug introspection --------------------------------------------

    def role_summary(self) -> list[dict]:
        """One row per arm for the debug panel."""
        rows = []
        for i, arm in enumerate(self.octopus.arms):
            ext = float(np.linalg.norm(arm.tip_pos() - arm.root_pos()))
            reach_full = arm.n * arm.SEG_LEN * 0.9
            rows.append({
                "idx":       i,
                "role":      arm.role,
                "gripping":  arm.is_anchored,
                "slip":      arm.grip_slip_fraction(),
                "tension":   arm._last_tension_mag,
                "push":      arm._last_push_mag,
                "extension": min(1.0, ext / reach_full),
                "reach_deg": np.degrees(arm.reach_offset),
            })
        return rows
