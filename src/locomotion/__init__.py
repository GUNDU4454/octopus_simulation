"""
locomotion  —  AI-ready, target-driven octopus locomotion control layer.

This package sits ON TOP of the proven Verlet physics (src/entities/tentacle.py
= the TentaclePhysics layer, src/entities/octopus.py = the body).  It adds no
new integrator; it only *decides* what each arm should do each frame and writes
the arm's existing low-level command fields (grip / reach / contract / push).

Separation of concerns (each is a small, swappable module):

  TargetMovementSystem      where the body is trying to go (click target)
  GripDetection             which world points afford a grip / push
  BodyLocomotionController  central brain: target → body-level BodyDrive
  TentacleController        one arm's role decision + command translation
  BodyPhysicsController     body force→motion integration (mass, drag, friction)
  LocomotionManager         orchestrates all arms toward the target

There is NO global gait clock.  Each arm picks a role from local geometry and
feedback, and the crawl emerges from many arms' independent grip lifecycles.
A reinforcement-learning policy can later replace LocomotionManager wholesale
without touching the physics, since it drives the very same command fields.
"""

from src.locomotion.target_movement import TargetMovementSystem
from src.locomotion.grip_detection import GripDetection
from src.locomotion.body_locomotion import BodyLocomotionController, BodyDrive
from src.locomotion.tentacle_controller import TentacleController, Role
from src.locomotion.body_physics import BodyPhysicsController
from src.locomotion.locomotion_manager import LocomotionManager

__all__ = [
    "TargetMovementSystem",
    "GripDetection",
    "BodyLocomotionController",
    "BodyDrive",
    "TentacleController",
    "Role",
    "BodyPhysicsController",
    "LocomotionManager",
]
