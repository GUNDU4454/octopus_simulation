"""
tentacle.py  —  Muscular hydrostat arm with grapple anchor physics

Each arm cycles through four phases that produce emergent seabed crawling:

  REACHING    tip explores nearby space, steers toward reach_target
  PLANTED     tip spring-anchored to seabed; muscles contract to pull body
  SLIPPING    anchor under excess load — grip_point drifts toward body
  RETRACTING  tip free, arm recoils back toward body before next reach

Grapple / anchor physics
------------------------
  PLANTED: a stiff position-projection constraint locks the tip to grip_point
  each solver iteration.  The constraint allows limited slip: if the residual
  tip-to-anchor error after all chain constraints exceeds SLIP_THRESHOLD, the
  grip_point drifts toward the tip (SLIPPING state).  Accumulated drift beyond
  MAX_SLIP_DIST breaks the grip automatically.

  The grip_point lives in world space — it never moves except during slip.
  This means a planted arm acts as a genuine temporary anchor against the
  seabed, not a kinematic target.

Force transfer
--------------
  Muscle contraction shortens segment target rest lengths.
  With the tip anchored and the root fixed to the body, the constraint solver
  cannot shorten the chain without pulling intermediate points inward.
  tension_force() reads the resulting geometry (root-to-anchor spring model)
  and returns the body-side pull vector.

  Single arm:    weak, directional pull
  Multiple arms: force vectors sum — aligned pulls compound strongly
  RL discovers the coordination pattern that maximises net locomotion force.
"""

import numpy as np
import math
from src.physics.verlet import Point, constrain_distance


class GripState:
    REACHING   = "reaching"
    PLANTED    = "planted"
    SLIPPING   = "slipping"    # anchor under stress, grip_point drifting
    RETRACTING = "retracting"


class Tentacle:
    N_SEGS      = 8
    SEG_LEN     = 26.0
    SEG_MASS    = 0.25

    # Muscle contraction
    MAX_CONTRACTION  = 0.45    # arm can shorten to 55 % of rest length
    CONTRACT_RATE    = 6.0     # activation ramp (1/s)
    RELAX_RATE       = 2.5     # deactivation ramp (1/s)

    # Force coefficients
    MUSCLE_STIFFNESS  = 180.0  # active pull per unit stretch × contraction
    PASSIVE_STIFFNESS = 2.2    # passive spring beyond natural rest

    # Push (extension against anchor) — a planted arm shoving the body away.
    # Mirror of tension_force: the substrate reacts on the arm, driving the
    # body in the anchor→root direction.  Scaled by push_cmd from the controller.
    PUSH_STIFFNESS    = 230.0  # body-reaction force per unit push_cmd

    # Verlet physics
    DRAG             = 0.91    # per-point Verlet damping (slightly reduced for inertia)
    BASE_SEG_DRAG    = 10.0    # seabed drag coefficient at root
    TIP_SEG_DRAG     = 16.0    # higher drag at thin tip
    SOLVER_ITERS     = 8
    CONSTRAINT_STIFF = 0.94

    # Grapple anchor
    GRIP_SPRING_K    = 550.0   # spring force toward anchor during integration (N/px)
    ANCHOR_STIFF     = 0.86    # position-projection stiffness per solver iteration
    ANCHOR_FINAL     = 0.96    # final post-loop projection stiffness
    SLIP_THRESHOLD   = 22.0    # px residual after free chain solving before slip starts
    SLIP_RATE_FACTOR = 0.07    # fraction of excess displacement → grip_point drift per frame
    MAX_SLIP_DIST    = 65.0    # accumulated slip before grip breaks completely

    def __init__(self, attach_angle: float, idx: int):
        self.attach_angle = attach_angle
        self.idx          = idx
        self.n            = self.N_SEGS

        self.points: list[Point] = [
            Point(0.0, 0.0, mass=self.SEG_MASS) for _ in range(self.n + 1)
        ]
        self.points[0].fixed = True   # root teleported to body perimeter each frame

        self.radii = [max(1.5, 7.5 - i * 0.72) for i in range(self.n + 1)]
        self.contractions = np.zeros(self.n, dtype=np.float64)

        # RL control inputs — written by Octopus.apply_action each frame
        self.grip_cmd          = 0.0   # [0,1]
        self.reach_offset      = 0.0   # angular offset for tip target
        self.contract_strength = 0.0   # [0,1] target contraction level

        # Extra control input — written by the locomotion controller (not RL).
        # Defaults to 0 so the 24-value RL action space is unchanged: an arm
        # only pushes when something explicitly raises push_cmd.
        self.push_cmd          = 0.0   # [0,1] extend-and-shove against anchor

        # How far a free arm reaches out, as a fraction of full length.  The
        # controller lowers this near the target for short, fine strokes.
        # Default 1.0 keeps RL (which never sets it) reaching at full length.
        self.reach_scale       = 1.0   # [~0,1] multiplies reach distance

        # High-level role tag, set by TentacleController, read by the renderer.
        self.role              = "idle"

        self.world_angle: float = attach_angle

        # Grip state machine
        self.state       = GripState.REACHING
        self.grip_point  = np.zeros(2, dtype=np.float64)
        self.plant_timer = 0.0
        self.slip_amount = 0.0    # accumulated slip distance (px)

        # Timing
        self._reach_timer   = 0.0
        self._retract_timer = 0.0

        # Diagnostic fields (updated each frame, read by renderer)
        self._anchor_disp      = 0.0   # tip-to-grip_point distance after solver
        self._last_tension_mag = 0.0   # cached body-pull magnitude
        self._last_push_mag    = 0.0   # cached body-push magnitude

    # ------------------------------------------------------------------
    # Frame entry points called by Octopus
    # ------------------------------------------------------------------

    def update_root(self, body_x: float, body_y: float,
                    body_angle: float, body_radius: float):
        """Teleport root point to body perimeter."""
        self.world_angle = self.attach_angle + body_angle
        rx = body_x + math.cos(self.world_angle) * body_radius * 0.88
        ry = body_y + math.sin(self.world_angle) * body_radius * 0.88
        self.points[0].teleport(rx, ry)

    def update(self, dt: float, body_vel: np.ndarray):
        """Full physics step for this arm."""
        self._update_state(dt)
        self._update_muscles(dt)
        self._apply_forces(dt, body_vel)
        self._integrate(dt)
        self._satisfy_constraints()

    # ------------------------------------------------------------------
    # State machine
    # ------------------------------------------------------------------

    def _update_state(self, dt: float):
        if self.state == GripState.REACHING:
            self._reach_timer += dt
            if self.grip_cmd > 0.5 and self._reach_timer > 0.10:
                self._plant()

        elif self.state == GripState.PLANTED:
            self.plant_timer += dt
            self._reach_timer = 0.0
            if self.grip_cmd < 0.5:
                self._release()

        elif self.state == GripState.SLIPPING:
            self.plant_timer += dt
            # Recover to PLANTED if tension eases and tip drifts back to anchor
            if self._anchor_disp < 2.0:
                self.state = GripState.PLANTED
                self.slip_amount = max(0.0, self.slip_amount - 5.0)
            # Break grip if slipped too far or commanded to release
            if self.slip_amount >= self.MAX_SLIP_DIST or self.grip_cmd < 0.5:
                self._release()

        elif self.state == GripState.RETRACTING:
            self._retract_timer += dt
            if (self._chain_length() < self.n * self.SEG_LEN * 0.55
                    or self._retract_timer > 0.8):
                self.state          = GripState.REACHING
                self._retract_timer = 0.0
                self._reach_timer   = 0.0
                self.plant_timer    = 0.0
                self.slip_amount    = 0.0
                self._anchor_disp   = 0.0

    def _plant(self):
        self.state       = GripState.PLANTED
        self.grip_point  = self.points[-1].pos.copy()
        self.plant_timer = 0.0
        self.slip_amount = 0.0
        # No hard pin — anchor is enforced through spring + position projection

    def _release(self):
        self.state        = GripState.RETRACTING
        self.slip_amount  = 0.0
        self._anchor_disp = 0.0

    # ------------------------------------------------------------------
    # Muscle dynamics
    # ------------------------------------------------------------------

    def _update_muscles(self, dt: float):
        if self.state in (GripState.PLANTED, GripState.SLIPPING):
            for i in range(self.n):
                # Taper: thick base segments contract more
                taper  = 1.0 - (i / self.n) * 0.35
                target = self.contract_strength * taper
                rate   = (self.CONTRACT_RATE if target > self.contractions[i]
                          else self.RELAX_RATE)
                self.contractions[i] += (target - self.contractions[i]) * dt * rate
                self.contractions[i]  = float(np.clip(self.contractions[i], 0.0, 1.0))
        else:
            decay = max(0.0, 1.0 - dt * self.RELAX_RATE)
            self.contractions *= decay

    # ------------------------------------------------------------------
    # Forces
    # ------------------------------------------------------------------

    def _apply_forces(self, dt: float, body_vel: np.ndarray):
        anchored = self.state in (GripState.PLANTED, GripState.SLIPPING)

        for i, p in enumerate(self.points):
            if p.fixed:
                continue
            pct = i / self.n

            # Per-segment underwater drag
            drag_coeff = self.BASE_SEG_DRAG + (self.TIP_SEG_DRAG - self.BASE_SEG_DRAG) * pct
            p.apply_force(-p.vel * drag_coeff / max(dt, 1e-6))

            # Body drag-along: only when free (planted arm behaves as anchored chain)
            if not anchored:
                p.apply_force(body_vel * (1.0 - pct) * 5.5)

        if not anchored:
            # Tip steering force toward reach target
            tip  = self.points[-1]
            root = self.points[0]
            target_angle = self.world_angle + self.reach_offset
            reach_scale  = float(np.clip(self.reach_scale, 0.2, 1.0))
            reach_dist   = self.SEG_LEN * self.n * 0.9 * reach_scale
            target = root.pos + np.array([
                math.cos(target_angle) * reach_dist,
                math.sin(target_angle) * reach_dist,
            ])
            to_target = target - tip.pos
            dist = float(np.linalg.norm(to_target))
            if dist > 2.0:
                strength = 95.0 if self.state == GripState.RETRACTING else 62.0
                tip.apply_force(to_target / dist * strength)
        else:
            # Anchor spring: pull tip toward grip_point during integration
            tip = self.points[-1]
            to_anchor = self.grip_point - tip.pos
            dist = float(np.linalg.norm(to_anchor))
            if dist > 0.1:
                tip.apply_force(to_anchor * self.GRIP_SPRING_K)

    # ------------------------------------------------------------------
    # Integration + constraints
    # ------------------------------------------------------------------

    def _integrate(self, dt: float):
        for p in self.points:
            p.integrate(dt, damping=self.DRAG)
        # No hard pin — anchor maintained entirely through _satisfy_constraints

    def _satisfy_constraints(self):
        anchored = self.state in (GripState.PLANTED, GripState.SLIPPING)

        # Chain constraints run first — tip is FREE to drift, allowing
        # muscle contraction tension to propagate without anchor interference.
        # This is the physics of real contraction: the chain wants to shorten,
        # pulling the tip toward the root.  The anchor then resists this.
        for _ in range(self.SOLVER_ITERS):
            for i in range(self.n):
                contracted_len = self.SEG_LEN * (
                    1.0 - self.contractions[i] * self.MAX_CONTRACTION
                )
                constrain_distance(
                    self.points[i], self.points[i + 1],
                    contracted_len, stiffness=self.CONSTRAINT_STIFF
                )

        if not anchored:
            return

        # Measure residual anchor displacement AFTER free chain solving.
        # This is the true grip-load: how far the contracting chain tried
        # to drag the tip away from its anchor.
        tip = self.points[-1]
        to_anchor = self.grip_point - tip.pos
        self._anchor_disp = float(np.linalg.norm(to_anchor))

        if self._anchor_disp > self.SLIP_THRESHOLD:
            # Chain tension exceeds grip strength — anchor drifts toward tip
            excess     = self._anchor_disp - self.SLIP_THRESHOLD
            slip_delta = excess * self.SLIP_RATE_FACTOR
            direction  = tip.pos - self.grip_point   # pointing toward tip
            dnorm      = float(np.linalg.norm(direction))
            if dnorm > 1e-6:
                self.grip_point += direction / dnorm * slip_delta
                self.slip_amount += slip_delta
                self.state = GripState.SLIPPING
                if self.slip_amount >= self.MAX_SLIP_DIST:
                    self._release()
                    return

        # Strong anchor projection — corrects tip to (possibly drifted) grip_point.
        # Applied twice for stability without adding a full extra solver loop.
        for _ in range(2):
            to_a = self.grip_point - self.points[-1].pos
            self.points[-1].pos += to_a * self.ANCHOR_FINAL
        # Damp oscillation at anchor to prevent bouncing
        self.points[-1].prev = (
            self.points[-1].pos - (self.points[-1].pos - self.points[-1].prev) * 0.25
        )

    # ------------------------------------------------------------------
    # Force on body  (called by Octopus each frame)
    # ------------------------------------------------------------------

    def tension_force(self) -> np.ndarray:
        """
        Pull force this arm exerts on the body when anchored.

        Modelled as a root-to-anchor spring, incorporating:
          passive  — arm extended beyond natural rest length
          active   — muscle contraction shortens target length,
                     amplifying effective stretch and pull force
          slip     — grip losing purchase reduces effective force linearly

        Emergent coordination:
          One arm alone:    single weak vector, barely moves the body
          Multiple anchored: force vectors sum — strong net locomotion
          RL discovers which coordination patterns maximise forward progress
        """
        if self.state not in (GripState.PLANTED, GripState.SLIPPING):
            self._last_tension_mag = 0.0
            return np.zeros(2)

        root   = self.points[0].pos
        anchor = self.grip_point
        vec    = anchor - root
        dist   = float(np.linalg.norm(vec))
        if dist < 1e-6:
            self._last_tension_mag = 0.0
            return np.zeros(2)

        direction = vec / dist
        natural   = self.n * self.SEG_LEN * 0.9

        passive = max(0.0, dist - natural) * self.PASSIVE_STIFFNESS

        avg_c             = float(np.mean(self.contractions))
        contracted_target = natural * (1.0 - avg_c * self.MAX_CONTRACTION)
        active = max(0.0, dist - contracted_target) * self.MUSCLE_STIFFNESS * avg_c

        # Slipping grip delivers reduced traction
        slip_factor = 1.0
        if self.state == GripState.SLIPPING:
            slip_factor = max(0.05, 1.0 - self.slip_amount / self.MAX_SLIP_DIST)

        total = (passive + active) * slip_factor
        self._last_tension_mag = total
        return direction * total

    def push_force(self) -> np.ndarray:
        """
        Reaction force a planted arm exerts on the body when it shoves itself
        away from its anchor (push_cmd > 0).

        Physical picture: the tip is gripped on the seabed and the arm extends,
        pressing against the substrate.  By Newton's third law the substrate
        pushes back along anchor→root, driving the body away from the grip
        point.  This is the mirror of tension_force(): pulling arms drag the
        body toward their anchor, pushing arms drive it away.

        Returns the body-side push vector (zero unless anchored and commanded).
        A pushing arm runs with low contraction, so its tension_force() is
        near zero — the two never fight for the same arm.
        """
        if self.state not in (GripState.PLANTED, GripState.SLIPPING):
            self._last_push_mag = 0.0
            return np.zeros(2)
        if self.push_cmd < 0.05:
            self._last_push_mag = 0.0
            return np.zeros(2)

        root   = self.points[0].pos
        anchor = self.grip_point
        vec    = root - anchor            # anchor → root: pushes body away
        dist   = float(np.linalg.norm(vec))
        if dist < 1e-6:
            self._last_push_mag = 0.0
            return np.zeros(2)

        direction = vec / dist

        # Slipping grip transmits less push, same as pull traction.
        slip_factor = 1.0
        if self.state == GripState.SLIPPING:
            slip_factor = max(0.05, 1.0 - self.slip_amount / self.MAX_SLIP_DIST)

        mag = self.push_cmd * self.PUSH_STIFFNESS * slip_factor
        self._last_push_mag = mag
        return direction * mag

    # ------------------------------------------------------------------
    # Per-segment tension for visualization
    # ------------------------------------------------------------------

    def segment_tension(self, seg_idx: int) -> float:
        if seg_idx >= self.n:
            return 0.0
        p_a = self.points[seg_idx]
        p_b = self.points[seg_idx + 1]
        dist = float(np.linalg.norm(p_b.pos - p_a.pos))
        contracted_len = self.SEG_LEN * (
            1.0 - self.contractions[seg_idx] * self.MAX_CONTRACTION
        )
        stretch = max(0.0, dist - contracted_len)
        return stretch * (1.0 + self.contractions[seg_idx] * 3.0)

    def grip_slip_fraction(self) -> float:
        """0.0 = solid grip, 1.0 = grip about to break."""
        return min(1.0, self.slip_amount / self.MAX_SLIP_DIST)

    @property
    def is_anchored(self) -> bool:
        """True when the tip has an active anchor (PLANTED or SLIPPING)."""
        return self.state in (GripState.PLANTED, GripState.SLIPPING)

    # ------------------------------------------------------------------
    # Queries
    # ------------------------------------------------------------------

    def get_positions(self) -> list[tuple[float, float]]:
        return [(p.pos[0], p.pos[1]) for p in self.points]

    def tip_pos(self) -> np.ndarray:
        return self.points[-1].pos.copy()

    def root_pos(self) -> np.ndarray:
        return self.points[0].pos.copy()

    def avg_contraction(self) -> float:
        return float(np.mean(self.contractions))

    def _chain_length(self) -> float:
        total = 0.0
        for i in range(self.n):
            total += float(np.linalg.norm(
                self.points[i + 1].pos - self.points[i].pos
            ))
        return total
