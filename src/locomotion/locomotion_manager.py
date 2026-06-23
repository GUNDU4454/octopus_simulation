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

    # NOTE: there is deliberately NO body-level steering torque here any more.
    # Rotation is 100 % emergent: the per-arm TentacleControllers bias their
    # pull strengths (see TentacleController._turn_factor) so the arms on the
    # side that helps the wanted turn haul harder, and the body rotates from the
    # resulting net r×F just like it translates from the net pull.  The old PD
    # loop that wrote octopus.steer_torque was a controller-driven rotation —
    # measured at ~1.1× the real tentacle torque — i.e. "the controller turned
    # the body," not "the tentacles turned it."  It has been removed.

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
        # each arm then chooses its role and force level from that command.  The
        # drive's turn_sign/turn_gain are read by each arm's _turn_factor to bias
        # its pull — that biasing is the ONLY thing that rotates the body now.
        self.drive = self.body.compute(oct_, self.targets)

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
