"""
octopus.py  —  Top-down octopus body with muscular hydrostat arms

Locomotion physics
------------------
  Body motion is 100 % emergent from tentacle force transfer.
  No scripted translation or procedural animation is used.

  Planted tentacles pull the body through active muscle contraction.
  Each arm's tension_force() returns a pull vector toward its anchor.
  The body integrates all arm forces with realistic inertia and drag.

  Drag is applied as a per-second exponential decay (fps-independent):
    vel *= exp(-DRAG_K * dt)
  This gives the body genuine momentum: a single pull impulse persists
  for ~1/DRAG_K seconds rather than dying within a frame.

  Seabed friction opposes sliding when not being actively pulled,
  preventing the body from drifting aimlessly between grip cycles.

  One contracted arm:     weak single-direction pull, barely moves body
  Multiple aligned arms:  force vectors sum — strong crawling emerges
  This is the physics basis for RL to discover coordinated gait patterns.

RL interface
------------
  apply_action(action)   24 values: 3 per arm (grip, reach, contract)
  get_observation()      79 values: 7 body + 9 × 8 arms
"""

import numpy as np
import math
from src.entities.tentacle import Tentacle, GripState


class Octopus:
    RADIUS         = 28

    # Body mass — heavier body requires more coordinated pulling to move
    MASS           = 9.0
    INERTIA        = 1200.0

    # Per-second drag coefficients (fps-independent, applied as exp(-k*dt))
    DRAG_K         = 3.5    # linear drag (water resistance)
    ANGULAR_DRAG_K = 2.8    # rotational drag

    # Seabed friction: constant force opposing body sliding
    # Reduced by active gripping (tentacles provide traction)
    SEABED_FRICTION = 14.0

    N_ARMS = 8

    def __init__(self, x: float, y: float):
        self.pos         = np.array([x, y], dtype=np.float64)
        self.vel         = np.zeros(2, dtype=np.float64)
        self.angle       = 0.0
        self.angular_vel = 0.0

        self.arms: list[Tentacle] = []
        for i in range(self.N_ARMS):
            attach = (i / self.N_ARMS) * 2 * math.pi
            arm = Tentacle(attach_angle=attach, idx=i)
            rx = x + math.cos(attach) * self.RADIUS * 0.88
            ry = y + math.sin(attach) * self.RADIUS * 0.88
            for j, pt in enumerate(arm.points):
                px = rx + math.cos(attach) * j * arm.SEG_LEN
                py = ry + math.sin(attach) * j * arm.SEG_LEN
                pt.pos[:] = [px, py]
                pt.prev[:] = [px, py]
            self.arms.append(arm)

        self.total_distance = 0.0
        self.time_alive     = 0.0
        self.planted_count  = 0
        self.body_net_force = np.zeros(2, dtype=np.float64)  # visualisation

    # ------------------------------------------------------------------
    # Main update — ALL motion comes from tentacle force transfer
    # ------------------------------------------------------------------

    def update(self, dt: float, width: int, height: int):
        self.time_alive += dt

        # Sync all arm roots to current body pose
        for arm in self.arms:
            arm.update_root(self.pos[0], self.pos[1], self.angle, self.RADIUS)

        # Accumulate locomotion forces from anchored, contracting arms
        total_force  = np.zeros(2)
        total_torque = 0.0
        planted = 0

        for arm in self.arms:
            f = arm.tension_force()
            fmag = float(np.linalg.norm(f))
            if fmag > 0.0:
                planted += 1
                total_force += f
                r = arm.root_pos() - self.pos
                total_torque += r[0] * f[1] - r[1] * f[0]

        self.planted_count  = planted
        self.body_net_force = total_force.copy()

        # Seabed friction — opposes sliding, reduced by active traction
        speed = float(np.linalg.norm(self.vel))
        if speed > 0.5:
            grip_fraction  = planted / self.N_ARMS
            friction_mag   = self.SEABED_FRICTION * (1.0 - grip_fraction * 0.55)
            total_force   -= (self.vel / speed) * friction_mag

        # Integrate body kinematics — fps-independent exponential drag
        self.vel         += (total_force / self.MASS) * dt
        self.vel         *= math.exp(-self.DRAG_K * dt)

        self.angular_vel += (total_torque / self.INERTIA) * dt
        self.angular_vel *= math.exp(-self.ANGULAR_DRAG_K * dt)

        self.angle += self.angular_vel * dt

        old_pos = self.pos.copy()
        self.pos += self.vel * dt
        self.total_distance += float(np.linalg.norm(self.pos - old_pos))

        # Soft boundary bounce
        margin = self.RADIUS + 15
        for axis, limit in enumerate([width, height]):
            if self.pos[axis] < margin:
                self.pos[axis] = margin
                self.vel[axis] = abs(self.vel[axis]) * 0.4
            elif self.pos[axis] > limit - margin:
                self.pos[axis] = limit - margin
                self.vel[axis] = -abs(self.vel[axis]) * 0.4

        # Step all arm physics (body_vel passed for drag-along coupling)
        for arm in self.arms:
            arm.update(dt, self.vel * 0.06)

    # ------------------------------------------------------------------
    # RL interface
    # ------------------------------------------------------------------

    def apply_action(self, action: np.ndarray):
        """
        action shape: (N_ARMS * 3,) = 24 values

        Per arm i:
          action[i*3 + 0]  grip_cmd          [-1,1] → [0,1]
          action[i*3 + 1]  reach_offset      [-1,1] → [-π/2, π/2]
          action[i*3 + 2]  contract_strength [-1,1] → [0,1]
        """
        for i, arm in enumerate(self.arms):
            arm.grip_cmd          = float((action[i * 3]     + 1.0) / 2.0)
            arm.reach_offset      = float(action[i * 3 + 1]) * (math.pi / 2.0)
            arm.contract_strength = float((action[i * 3 + 2] + 1.0) / 2.0)

    def get_observation(self) -> np.ndarray:
        """
        Observation vector (79 values):
          body state          7
          per-arm state × 8  72  (9 per arm)

        Per-arm: rel_tip_x, rel_tip_y, dist, is_anchored, is_reaching,
                 grip_cmd, avg_contraction, tension_mag, plant_timer

        Note: is_anchored = 1 for both PLANTED and SLIPPING states.
        """
        body = [
            self.pos[0] / 600.0,
            self.pos[1] / 400.0,
            self.vel[0] / 200.0,
            self.vel[1] / 200.0,
            math.cos(self.angle),
            math.sin(self.angle),
            self.angular_vel / 4.0,
        ]

        max_len     = self.arms[0].n * self.arms[0].SEG_LEN
        max_tension = 200.0

        arms_obs = []
        for arm in self.arms:
            tip     = arm.tip_pos()
            rel_tip = tip - self.pos
            dist    = float(np.linalg.norm(rel_tip))
            tension = float(np.linalg.norm(arm.tension_force()))

            arms_obs += [
                rel_tip[0] / max_len,
                rel_tip[1] / max_len,
                dist        / max_len,
                float(arm.is_anchored),           # PLANTED or SLIPPING
                float(arm.state == GripState.REACHING),
                arm.grip_cmd,
                arm.avg_contraction(),
                min(tension / max_tension, 1.0),
                min(arm.plant_timer / 2.0, 1.0),
            ]

        return np.array(body + arms_obs, dtype=np.float32)

    # ------------------------------------------------------------------
    # Queries
    # ------------------------------------------------------------------

    @property
    def speed(self) -> float:
        return float(np.linalg.norm(self.vel))

    def planted_arms(self) -> list[Tentacle]:
        return [a for a in self.arms if a.is_anchored]

    def free_arms(self) -> list[Tentacle]:
        return [a for a in self.arms if not a.is_anchored]
