# Octopus Top-Down Locomotion

True top-down seabed crawling simulation with RL-based movement learning.

## Run

```bash
pip install -r requirements.txt

# Watch demo gait (no training needed)
python simulate.py

# Click to set a crawl target in demo mode
# Press R to reset, Space to pause, Esc to quit

# Train the RL agent
python train.py

# Train longer
python train.py --timesteps 1000000

# Watch trained agent
python simulate.py --checkpoint models/checkpoints/locomotion_agent
```

## Physics model

**No gravity.** The seabed is the XY plane viewed from directly above.

Each arm is a Verlet chain of 8 segments. The key locomotion mechanism:

```
arm PLANTS tip on seabed (tip.fixed = True)
  ↓
body moves → root moves away from grip point
  ↓
arm stretches → creates tension force
  ↓
tension pulls body toward grip point
  ↓
arm reaches locomotion threshold → tip RELEASES → arm RETRACTS → REACHES again
```

Multiple arms cycle through REACHING → PLANTED → RETRACTING asynchronously,
producing smooth distributed traction like a real octopus.

## What the debug view shows

| Visual | Meaning |
|--------|---------|
| Blue arms | Reaching or retracting (free) |
| Red arms | Planted (gripping seabed) |
| Yellow arms | Retracting |
| Orange circle | Grip contact point |
| Yellow arrow | Tension force vector |
| Green arrow | Body velocity |
| White dot | Heading direction |
| Bottom strip | Per-arm state (0–7) |

## RL

- **Obs**: 55 values (body state + per-arm state)
- **Action**: 16 values (grip command + reach offset per arm)
- **Reward**: distance moved + stability bonus - spinning penalty

## File structure

```
simulate.py          Interactive demo + trained agent viewer
train.py             PPO training
src/
  physics/verlet.py  2D Verlet point + distance constraint
  entities/
    tentacle.py      Arm chain with grip state machine
    octopus.py       Body + 8 arms + force integration
  rl/env.py          Gymnasium locomotion environment
  rendering/
    renderer.py      Top-down debug visualiser
```
