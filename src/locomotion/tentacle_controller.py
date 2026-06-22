"""
tentacle_controller.py  —  Role + TentacleController

Decides ONE arm's role each frame from the body-level `BodyDrive` (see
body_locomotion.py) plus local geometry and physics feedback, then translates
that role into the arm's low-level command fields
(grip_cmd / reach_offset / contract_strength / push_cmd / reach_scale).

Roles are NOT fixed to an arm.  Every frame each arm re-evaluates.

Roles
-----
  PULLING      anchor lies toward the target: contract → drag body toward it
  PUSHING      anchor lies behind: extend against it → shove body forward
  ANCHORING    side anchor held as a stable, low-tension base
  STABILIZING  side anchor contracting lightly to damp unwanted spin
  SEARCHING    reaching out with no grip yet, looking for a plant point
  IDLE         no target: relax

Two things fix the old "drifts but never arrives / never turns" behaviour:

1. Decide by the ACTUAL anchor direction, not the mount angle.
   A planted grip is fixed in world space, so an arm mounted toward the target
   can end up anchored *behind* the body after it moves.  Previously such an arm
   still counted as "pulling" and hauled the body backwards — the net force
   pointed away from the target and the crawl stalled.  Now an anchored arm
   pulls only while its grip is genuinely toward the target (so pulling always
   reduces the distance), pushes when its grip is behind, and otherwise just
   holds.  Free arms still pick where to reach from their mount angle.

2. Inchworm re-grip.  A pulling arm releases a spent grip (once the body has
   hauled itself up close to the anchor) so it recoils and reaches out fresh
   toward the target.  Repeated strokes carry the body the whole way.

Rotation is emergent: `_turn_factor` scales each arm's effort by whether the
torque it would add (r × F) matches the BodyDrive's desired turn direction, so
uneven arm forces gradually rotate the body to face the target — never a
scripted snap of the transform.
"""

import math
import numpy as np

from src.locomotion.body_locomotion import wrap


class Role:
    PULLING     = "pulling"
    PUSHING     = "pushing"
    ANCHORING   = "anchoring"
    STABILIZING = "stabilizing"
    SEARCHING   = "searching"
    IDLE        = "idle"


class TentacleController:

    # Cosine thresholds vs the target direction.  Free arms are classified by
    # their MOUNT direction (where to reach); anchored arms by their ACTUAL
    # anchor direction (what the grip can usefully do).  The bands are generous
    # so several arms reach toward the target and pull at once — too narrow and
    # only ~1 arm pulls at a time, leaving the body crawling in fits.
    # Pull cone is deliberately NARROW.  An anchored arm pulls the body toward
    # its own grip point; if arms whose grips are spread across a wide forward
    # cone all pull at once, their force vectors point in different directions
    # and largely cancel (measured: net translation collapses, the body stalls
    # and spins).  Keeping the pull cone tight means the arms that do pull are
    # all hauling in nearly the same (target-ward) direction, so their forces
    # COMPOUND instead of fighting — this is the core "coordinate, don't
    # compete" guarantee.  Arms whose grips fall outside the cone hold as
    # stabilisers or, if well behind, push.
    REACH_TOWARD_COS = -0.20   # free: mount within ~100° of target → reach for it
    REACH_AWAY_COS   = -0.60   # free: mount past ~127° away → reach out to push
    PULL_ANCHOR_COS  =  0.20   # anchored: grip ahead of body → pull toward it
    PUSH_ANCHOR_COS  = -0.50   # anchored: grip well behind → push body forward

    # Extension fractions (of full reach) at which a free arm commits a grip.
    # Eased toward PLANT_EXT_NEAR near the target for shorter, finer strokes.
    PLANT_EXT_PULL  = 0.70
    PLANT_EXT_PUSH  = 0.55
    PLANT_EXT_SIDE  = 0.50
    PLANT_EXT_NEAR  = 0.42

    PUSH_STROKE_T   = 0.55   # seconds a single push stroke lasts before release

    # Seconds a side/holding anchor may be kept before it must let go and reach
    # out fresh toward the target.  Without this an arm that grabs a side grip
    # holds it indefinitely, permanently removing itself from the pull/push
    # workforce; with only one or two arms left actively hauling, the body locks
    # into a no-progress limit cycle a hundred-odd px short of the goal
    # (measured).  Cycling stale anchors keeps every arm contributing — and is
    # what makes the arms visibly "reposition continuously" like a real octopus.
    ANCHOR_HOLD_T   = 0.7

    # Inchworm re-grip: release a pulling arm once the body has hauled itself to
    # within this fraction of full reach of the anchor (the grip is "spent").
    PULL_CONSUME_FRAC = 0.30

    # Release any anchor the body has drifted away from beyond this fraction of
    # full reach.  A real arm cannot pull from farther than its length; without
    # this, a stranded grip becomes a runaway spring that drags the body back.
    # NOTE: must stay > 1.0.  A freshly committed grip is captured at the tip,
    # which for a straight arm sits at exactly full length (span ≈ full).  A
    # value ≤ 1.0 therefore releases every new grip on the very next frame —
    # the arm never gets to pull.  Overstretch should only fire once the body
    # has genuinely drifted *past* arm's reach, i.e. span noticeably > full.
    OVERSTRETCH_FRAC = 1.15

    # Turn bias: an arm that helps the desired turn scales effort up by up to
    # +TURN_AUTHORITY, one that fights it scales down toward (1 - TURN_AUTHORITY).
    # Kept small: body heading is steered by the manager's closed-loop torque;
    # this only adds visible left/right unevenness, it is not the turn's engine.
    TURN_AUTHORITY = 0.25

    def __init__(self, arm, grip_detect):
        self.arm  = arm
        self.grip = grip_detect
        self._push_timer = 0.0
        self._hold_timer = 0.0   # how long this arm has held a side anchor
        self._body = None     # set each frame for the torque-bias helper

    # ------------------------------------------------------------------

    def decide(self, body, drive, dt) -> str:
        """Pick a role and write the arm's command fields.  Returns the role.

        `drive` is the BodyDrive from BodyLocomotionController for this frame.
        """
        arm = self.arm
        self._body = body

        if not drive.has_target:
            arm.role = self._idle()
            return arm.role

        # Far → reach far out for big strokes; near → short reach, fine steps.
        arm.reach_scale = 0.5 + 0.5 * drive.drive_gain

        if arm.is_anchored:
            role = self._anchored(body, drive, dt)
        else:
            role = self._free(body, drive)

        arm.role = role
        return role

    # ------------------------------------------------------------------
    # Free arm — choose where to reach from the mount direction
    # ------------------------------------------------------------------

    def _free(self, body, drive) -> str:
        arm = self.arm
        arm.contract_strength = 0.0
        arm.push_cmd          = 0.0
        self._push_timer      = 0.0
        self._hold_timer      = 0.0

        world_angle = arm.attach_angle + body.angle
        delta = wrap(drive.target_angle - world_angle)
        align = math.cos(delta)

        if align >= self.REACH_TOWARD_COS:
            # Front-ish arm: steer the tip toward the target and plant ahead.
            arm.reach_offset = float(np.clip(delta, -math.pi / 2, math.pi / 2))
            ext = self._plant_ext(self.PLANT_EXT_PULL, drive)
        elif align <= self.REACH_AWAY_COS:
            # Rear arm: reach straight out (away from target) to plant a pusher.
            arm.reach_offset = 0.0
            ext = self._plant_ext(self.PLANT_EXT_PUSH, drive)
        else:
            # Side arm: reach out to plant a traction/stabiliser anchor.
            arm.reach_offset = 0.0
            ext = self._plant_ext(self.PLANT_EXT_SIDE, drive)

        arm.grip_cmd = 1.0 if self._extended_enough(ext) else 0.0
        return Role.SEARCHING

    # ------------------------------------------------------------------
    # Anchored arm — behave from where the grip ACTUALLY is vs the target
    # ------------------------------------------------------------------

    def _anchored(self, body, drive, dt) -> str:
        arm = self.arm

        # Reach from root to the grip.  Too short → spent (inchworm); too long →
        # the body has drifted off it and it would drag backward.  Either way let
        # go so the arm recoils and reaches out fresh toward the target.
        span = float(np.linalg.norm(arm.grip_point - arm.root_pos()))
        full = arm.n * arm.SEG_LEN
        if span < full * self.PULL_CONSUME_FRAC or span > full * self.OVERSTRETCH_FRAC:
            return self._release()

        anchor_vec = arm.grip_point - body.pos
        adist = float(np.linalg.norm(anchor_vec))
        if adist < 1e-6:
            return self._release()
        align = float(np.dot(anchor_vec / adist, drive.direction))  # +ahead / -behind

        arm.grip_cmd     = 1.0
        arm.reach_offset = 0.0

        if align >= self.PULL_ANCHOR_COS:
            # Anchor is toward the target: pulling toward it reduces distance.
            arm.push_cmd = 0.0
            self._push_timer = 0.0
            self._hold_timer = 0.0
            strength = (0.45 + 0.55 * align) * (0.4 + 0.6 * drive.drive_gain)
            strength *= self._turn_factor(arm, drive, pulling=True)
            arm.contract_strength = float(np.clip(strength, 0.0, 1.0))
            return Role.PULLING

        if align <= self.PUSH_ANCHOR_COS:
            # Anchor is behind: extend against it to shove the body forward.
            arm.contract_strength = 0.0
            self._hold_timer = 0.0
            self._push_timer += dt
            if self._push_timer > self.PUSH_STROKE_T:
                return self._release()    # end of stroke — re-plant fresh
            push = (0.5 + 0.5 * drive.drive_gain) * self._turn_factor(
                arm, drive, pulling=False
            )
            arm.push_cmd = float(np.clip(push, 0.0, 1.0))
            return Role.PUSHING

        # Side anchor: hold for traction with only light, spin-damping pull so it
        # never drags the body off course (the old constant pull cancelled the
        # pullers).  It can also lend a little to a commanded turn.  Held only
        # briefly (ANCHOR_HOLD_T) before letting go and reaching out fresh, so a
        # side grip can't permanently sideline the arm and stall the crawl.
        arm.push_cmd     = 0.0
        self._push_timer = 0.0
        self._hold_timer += dt
        if self._hold_timer > self.ANCHOR_HOLD_T:
            return self._release()
        spin   = abs(body.angular_vel)
        damp   = float(np.clip(spin * 0.30, 0.0, 0.30))
        assist = max(0.0, self._turn_factor(arm, drive, pulling=True) - 1.0)
        arm.contract_strength = float(np.clip(damp + assist, 0.0, 0.40))
        if spin > 0.3 or drive.turn_gain > 0.2:
            return Role.STABILIZING
        return Role.ANCHORING

    def _idle(self) -> str:
        arm = self.arm
        arm.grip_cmd          = 0.0
        arm.contract_strength = 0.0
        arm.push_cmd          = 0.0
        arm.reach_offset      = 0.0
        arm.reach_scale       = 1.0
        self._push_timer      = 0.0
        self._hold_timer      = 0.0
        return Role.IDLE

    def _release(self) -> str:
        """Let go: the grip state machine will retract, then the arm re-reaches."""
        arm = self.arm
        arm.grip_cmd          = 0.0
        arm.contract_strength = 0.0
        arm.push_cmd          = 0.0
        self._push_timer      = 0.0
        self._hold_timer      = 0.0
        return Role.SEARCHING

    # ------------------------------------------------------------------
    # Turn bias — emergent rotation from uneven arm forces
    # ------------------------------------------------------------------

    def _turn_factor(self, arm, drive, pulling: bool) -> float:
        """Scale an arm's effort by how well its torque serves the wanted turn.

        Returns 1.0 when no turn is demanded.  Otherwise predicts the sign of
        the torque (r × F) this arm's force would add and scales up (helps the
        turn) or down (fights it).  Since the body integrates torque = r × F
        from these very forces, biasing the strengths yields a real net torque
        that rotates the body toward the target.
        """
        if drive.turn_gain <= 0.0 or drive.turn_sign == 0.0 or self._body is None:
            return 1.0

        r = arm.root_pos() - self._body.pos          # lever arm from body centre

        if pulling:
            fdir = (arm.grip_point - arm.root_pos()) if arm.is_anchored else drive.direction
        else:
            fdir = (arm.root_pos() - arm.grip_point) if arm.is_anchored else -drive.direction

        tau = r[0] * fdir[1] - r[1] * fdir[0]        # 2D cross product (z)
        if abs(tau) < 1e-6:
            return 1.0

        helps  = (tau > 0.0) == (drive.turn_sign > 0.0)
        signed = 1.0 if helps else -1.0
        factor = 1.0 + self.TURN_AUTHORITY * drive.turn_gain * signed
        return float(np.clip(factor, 0.3, 1.7))

    # ------------------------------------------------------------------

    def _plant_ext(self, base_frac: float, drive) -> float:
        """Grip-commit extension threshold, eased toward PLANT_EXT_NEAR as the
        body nears the target so it takes shorter, finer strokes."""
        return self.PLANT_EXT_NEAR + (base_frac - self.PLANT_EXT_NEAR) * drive.drive_gain

    def _extended_enough(self, frac: float) -> bool:
        """True if the arm has reached out past `frac` of full length and the
        tip is over grippable ground."""
        arm = self.arm
        ext        = float(np.linalg.norm(arm.tip_pos() - arm.root_pos()))
        reach_full = arm.n * arm.SEG_LEN * 0.9
        return ext > reach_full * frac and self.grip.has_grip(arm.tip_pos())
