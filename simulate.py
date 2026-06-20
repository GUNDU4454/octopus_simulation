"""
simulate.py  —  Interactive top-down octopus locomotion demo

Without a trained agent the octopus uses a hand-coded gait:
  arms alternate between gripping and reaching in two offset groups,
  producing a basic crawling motion.

With a trained agent:
  python simulate.py --checkpoint models/checkpoints/locomotion_agent

Controls
--------
  Click       Set crawl target (demo mode)
  R           Reset
  Space       Pause / unpause
  Esc / Q     Quit
"""

import sys
import math
import argparse

import pygame
import numpy as np

from src.entities.octopus import Octopus
from src.entities.tentacle import GripState
from src.rendering.renderer import Renderer

WIDTH  = 1200
HEIGHT = 800
FPS    = 60


# ------------------------------------------------------------------
# Demo gait: simple alternating two-group crawl
# ------------------------------------------------------------------

def demo_gait(octopus: Octopus, t: float, target: np.ndarray | None):
    """
    Hand-coded gait for demo mode.

    Arms 0,2,4,6 (group A) grip while 1,3,5,7 (group B) reach,
    then they swap.  The reach direction is biased toward the target.
    """
    period    = 1.2      # seconds per full cycle
    phase     = (t % period) / period   # 0→1

    body_to_target = None
    if target is not None:
        diff = target - octopus.pos
        dist = np.linalg.norm(diff)
        if dist > 20:
            body_to_target = math.atan2(diff[1], diff[0])

    action = np.zeros(octopus.N_ARMS * 3, dtype=np.float32)

    for i, arm in enumerate(octopus.arms):
        group_a = (i % 2 == 0)
        grip    = phase < 0.55 if group_a else phase >= 0.45

        # Reach toward target if given, else forward
        if body_to_target is not None:
            world_target_angle = body_to_target
        else:
            world_target_angle = octopus.angle

        # Offset so leading arms reach further forward
        arm_world  = arm.attach_angle + octopus.angle
        delta      = world_target_angle - arm_world
        # Normalise to [-π, π]
        while delta >  math.pi: delta -= 2*math.pi
        while delta < -math.pi: delta += 2*math.pi
        reach_off  = float(np.clip(delta / (math.pi/2), -1, 1))

        grip_val     = 1.0 if grip else -1.0
        contract_val = 1.0 if grip else -1.0   # contract when planted, relax when reaching
        action[i*3]     = grip_val
        action[i*3 + 1] = reach_off if not grip else 0.0
        action[i*3 + 2] = contract_val

    octopus.apply_action(action)


# ------------------------------------------------------------------
# Main loop
# ------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, default=None)
    args = parser.parse_args()

    # Load agent if checkpoint provided
    agent = None
    if args.checkpoint:
        from stable_baselines3 import PPO
        from src.rl.env import LocomotionEnv
        env   = LocomotionEnv()
        agent = PPO.load(args.checkpoint, env=env)
        obs   = None
        print(f"Loaded agent from {args.checkpoint}")
    else:
        print("Demo mode — hand-coded gait.  Click to set target.")

    pygame.init()
    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    pygame.display.set_caption(
        "Octopus Locomotion" + (" — Trained Agent" if agent else " — Demo")
    )
    clock    = pygame.time.Clock()
    renderer = Renderer(screen, WIDTH, HEIGHT)

    octopus = Octopus(WIDTH / 2, HEIGHT / 2)
    target  = None
    t       = 0.0
    step    = 0
    paused  = False

    if agent:
        from src.rl.env import LocomotionEnv
        _env = LocomotionEnv()
        _env._octopus = octopus
        obs = octopus.get_observation()

    running = True
    while running:
        dt = clock.tick(FPS) / 1000.0
        dt = min(dt, 0.034)

        # ── Events ──────────────────────────────────────────────────
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key in (pygame.K_ESCAPE, pygame.K_q):
                    running = False
                elif event.key == pygame.K_r:
                    octopus = Octopus(WIDTH/2, HEIGHT/2)
                    target  = None
                    t       = 0.0
                    step    = 0
                    if agent:
                        obs = octopus.get_observation()
                elif event.key == pygame.K_SPACE:
                    paused = not paused
            elif event.type == pygame.MOUSEBUTTONDOWN:
                if event.button == 1:
                    mx, my = pygame.mouse.get_pos()
                    target = np.array([float(mx), float(my)])

        if paused:
            renderer.draw(octopus, step)
            pygame.display.flip()
            continue

        # ── Action ──────────────────────────────────────────────────
        if agent:
            action, _ = agent.predict(obs, deterministic=True)
            octopus.apply_action(action)
            obs = octopus.get_observation()
        else:
            demo_gait(octopus, t, target)

        # ── Physics ─────────────────────────────────────────────────
        octopus.update(dt, WIDTH, HEIGHT)

        t    += dt
        step += 1

        # ── Render ──────────────────────────────────────────────────
        renderer.draw(octopus, step)
        pygame.display.flip()

    pygame.quit()
    sys.exit()


if __name__ == "__main__":
    main()
