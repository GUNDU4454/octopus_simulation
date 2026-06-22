# Octopus Top-Down Locomotion

True top-down seabed crawling. The octopus moves **only** from forces produced
by its tentacles — there are no scripted walk cycles or position animations.
Click a spot and it crawls there: front arms reach out, grip and pull; rear
arms plant and push; side arms anchor to keep it from spinning. A reinforcement-
learning agent can later replace the hand-written controller without touching
the physics.

## Run

```bash
pip install -r requirements.txt

# Demo: click anywhere to set a crawl target
python simulate.py

# Watch a trained agent instead (manager is bypassed)
python simulate.py --checkpoint models/checkpoints/locomotion_agent

# Train the RL agent
python train.py
python train.py --timesteps 1000000
```

### Controls

| Key / action | Effect |
|--------------|--------|
| Left click   | Set crawl target |
| `D`          | Toggle debug overlay (roles, forces, panel) |
| `R`          | Reset |
| `Space`      | Pause / unpause |
| `Esc` / `Q`  | Quit |

## Physics model

**No gravity.** The seabed is the XY plane viewed from directly above. Each arm
is an 8-segment Verlet chain (a muscular hydrostat). Body motion is 100%
emergent from tentacle force transfer — nothing writes the body transform
directly.

- **Pull** — an arm grips the seabed, then muscle contraction shortens the
  chain. With the tip anchored, the body is dragged toward the grip point
  (`Tentacle.tension_force`).
- **Push** — an arm grips and *extends* against the substrate; the reaction
  drives the body away from the grip point (`Tentacle.push_force`).
- Grip lifecycle `REACHING → PLANTED → SLIPPING → RETRACTING` runs per arm from
  load feedback, so the gait desynchronises and emerges instead of being timed.

## Locomotion control (`src/locomotion/`)

Roles are reassigned every frame from each arm's angle to the target, its grip
state, extension, and the body's spin — no arm has a fixed job:

| Role | Colour | Behaviour |
|------|--------|-----------|
| Pulling | red-orange | aligned with target: reach, grip, contract → drag body |
| Pushing | cyan | opposite target: plant, extend → shove body forward |
| Anchoring | green | hold a stable grip as a base |
| Stabilizing | purple | side grip damping unwanted rotation |
| Searching | yellow | reaching out, no grip yet |
| Idle | dim blue | no target |

A central `BodyLocomotionController` turns the raw target into one body-level
command each frame — travel direction, distance, desired heading, and which way
to rotate — and the arms read that command to choose roles and force levels.
Three things make the body actually arrive *and* orient itself:

- **Anchor-aware roles** — an anchored arm acts on where its grip *actually* is,
  not where it is mounted: it pulls only while the grip is toward the target (so
  every pull shortens the distance), pushes when the grip is behind, and holds
  lightly otherwise. A grip the body has drifted off (over-stretched) is
  released instead of becoming a spring that drags the body backward.
- **Inchworm re-grip** — a pulling arm releases a *spent* grip (once the body has
  hauled itself up to the anchor) so it recoils and reaches out again. Repeated
  strokes carry the body all the way to the target instead of stalling. Far away
  it takes big strong strokes; close in it makes small corrections and
  stabilises to avoid overshoot.
- **Closed-loop heading** — a PD controller on the heading error produces a
  steering torque (gated by how many arms are gripping, so it is still
  tentacle-borne against the seabed) that turns the body to face the target
  without snapping. Arms also bias their effort left/right toward the turn, so
  the uneven forces are visible, but the PD loop is what makes it reliable.

Modules (each small and swappable, so RL can drop in later):
`TargetMovementSystem`, `GripDetection`, `BodyLocomotionController`,
`TentacleController`, `BodyPhysicsController`, `LocomotionManager`.

## Debug overlay (`D` to toggle)

Numbered arms tinted by role; per-arm pull (yellow) and push (cyan) force
arrows; green net-force and velocity vectors; the body-forward vector (blue) and
a pink rotation-torque arc showing which way the body is turning; grip-contact
rings with slip indicators; a center-of-mass crosshair; the crawl target with a
heading arrow; a heading-error / turn-effort readout; and a right-side panel
listing every tentacle's role, grip, force and extension.

## RL

- **Observation**: 79 values (7 body + 9 × 8 arms)
- **Action**: 24 values (grip, reach offset, contract per arm)
- **Reward**: distance moved + multi-arm efficiency − spin penalty

Push is controller-only (`push_cmd`, default 0), so the RL action space is
unchanged and backward compatible.

## File structure

```
simulate.py              Interactive demo + trained-agent viewer
train.py                 PPO training
src/
  physics/verlet.py      2D Verlet point + distance constraint
  entities/
    tentacle.py          Arm chain: grip state machine, pull + push forces
    octopus.py           Body + 8 arms + force/torque integration
  locomotion/            Target-driven role control layer (see above)
  rl/env.py              Gymnasium locomotion environment
  rendering/renderer.py  Top-down debug visualiser
```

## Privacy & security

This simulation is fully **local and offline**. It collects, stores, and
transmits **no** personal data, gameplay data, telemetry, or analytics, and
makes no network calls (the only dependencies are pygame, numpy, gymnasium, and
stable-baselines3). There are no accounts, logins, identifiers, or stored IP/
location data, and no API keys or secrets in the codebase. The debug overlay
shows simulation state only. If networked/AI features are added later, keep
data local by default and load any secrets from environment variables rather
than committing them.
