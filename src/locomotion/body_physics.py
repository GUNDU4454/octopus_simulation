"""
body_physics.py  —  BodyPhysicsController

Integrates the body's rigid-motion state from the net force and torque the
tentacles produce.  Pure Newtonian update: no scripted translation ever
touches the body transform — it only moves because forces moved it.

  vel  += (F / mass) * dt   ; vel  *= exp(-DRAG_K * dt)        (water drag)
  w    += (T / I)    * dt   ; w    *= exp(-ANGULAR_DRAG_K * dt)
  angle += w * dt
  pos   += vel * dt

Seabed friction opposes sliding and is reduced by how many arms are gripping
(traction), so the body doesn't drift aimlessly between grip cycles.

This is a straight extraction of the body integration that used to live inside
Octopus.update — same constants, same maths — now isolated so it can be tuned
or swapped independently of arm logic.
"""

import math
import numpy as np


class BodyPhysicsController:

    MASS            = 9.0
    INERTIA         = 1200.0
    DRAG_K          = 1.8     # linear water drag (per second) — lower = more
                              # glide between strokes, so the inchworm crawl
                              # covers ground instead of stop-starting
    ANGULAR_DRAG_K  = 12.0    # rotational drag (per second) — high, so the
                              # crawl's chaotic incidental torque is damped out
                              # and the controller's steering torque dominates,
                              # letting the body settle facing the target
    SEABED_FRICTION = 14.0    # constant force opposing slide, eased by gripping

    def integrate(self, body, total_force, total_torque,
                  grip_fraction: float, dt: float,
                  width: int, height: int) -> float:
        """Advance `body` (needs pos/vel/angle/angular_vel/RADIUS) one step.
        Returns distance moved this step."""
        force = np.asarray(total_force, dtype=np.float64).copy()

        # Seabed friction — opposes sliding, reduced by active traction.
        speed = float(np.linalg.norm(body.vel))
        if speed > 0.5:
            friction_mag = self.SEABED_FRICTION * (1.0 - grip_fraction * 0.55)
            force -= (body.vel / speed) * friction_mag

        # Linear — fps-independent exponential drag gives genuine momentum.
        body.vel += (force / self.MASS) * dt
        body.vel *= math.exp(-self.DRAG_K * dt)

        # Angular.
        body.angular_vel += (total_torque / self.INERTIA) * dt
        body.angular_vel *= math.exp(-self.ANGULAR_DRAG_K * dt)
        body.angle       += body.angular_vel * dt

        # Position.
        old = body.pos.copy()
        body.pos += body.vel * dt
        moved = float(np.linalg.norm(body.pos - old))

        # Soft boundary bounce.
        margin = body.RADIUS + 15
        for axis, limit in enumerate((width, height)):
            if body.pos[axis] < margin:
                body.pos[axis] = margin
                body.vel[axis] = abs(body.vel[axis]) * 0.4
            elif body.pos[axis] > limit - margin:
                body.pos[axis] = limit - margin
                body.vel[axis] = -abs(body.vel[axis]) * 0.4

        return moved
