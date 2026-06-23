"""
simulate.py  —  Interactive top-down octopus locomotion demo

Click anywhere and the octopus crawls there.  It is NOT animated toward the
point — clicking only sets a goal.  All motion emerges from the tentacles:
front arms reach out, grip and pull; rear arms plant and push; side arms
anchor to keep it from spinning.  The LocomotionManager assigns those roles
every frame from the target direction and physics feedback (no scripted gait).

With a trained agent the manager is bypassed and the RL policy drives the arms:
  python simulate.py --checkpoint models/checkpoints/locomotion_agent

Controls
--------
  Click       Set crawl target
  D           Toggle debug overlay (roles, forces, panel)
  R           Reset
  Space       Pause / unpause
  Esc / Q     Quit
"""

import sys
import argparse

import pygame
import numpy as np

from src.entities.octopus import Octopus
from src.rendering.renderer import Renderer
from src.locomotion.locomotion_manager import LocomotionManager

WIDTH  = 1200
HEIGHT = 800
FPS    = 60


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, default=None)
    args = parser.parse_args()

    # Load agent if a checkpoint is provided (manager is bypassed in that case).
    agent = None
    if args.checkpoint:
        from stable_baselines3 import PPO
        from src.rl.env import LocomotionEnv
        env   = LocomotionEnv()
        agent = PPO.load(args.checkpoint, env=env)
        print(f"Loaded agent from {args.checkpoint}")
    else:
        print("Demo mode — click to set a crawl target.")

    pygame.init()
    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    pygame.display.set_caption(
        "Octopus Locomotion" + (" — Trained Agent" if agent else " — Demo")
    )
    clock    = pygame.time.Clock()
    renderer = Renderer(screen, WIDTH, HEIGHT)

    
    octopus = Octopus(WIDTH / 2, HEIGHT / 2)
    manager = LocomotionManager(octopus, WIDTH, HEIGHT)

    step    = 0
    paused  = False
    debug   = True

    obs = octopus.get_observation() if agent else None

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
                    octopus = Octopus(WIDTH / 2, HEIGHT / 2)
                    manager = LocomotionManager(octopus, WIDTH, HEIGHT)
                    step    = 0
                    if agent:
                        obs = octopus.get_observation()
                elif event.key == pygame.K_SPACE:
                    paused = not paused
                elif event.key == pygame.K_d:
                    debug = not debug
            elif event.type == pygame.MOUSEBUTTONDOWN:
                if event.button == 1:
                    mx, my = pygame.mouse.get_pos()
                    manager.set_target(float(mx), float(my))

        if paused:
            renderer.draw(octopus, step, manager=manager, debug=debug)
            pygame.display.flip()
            continue

        # ── Control ─────────────────────────────────────────────────
        if agent:
            action, _ = agent.predict(obs, deterministic=True)
            octopus.apply_action(action)
        else:
            # Manager writes each arm's grip/reach/contract/push commands.
            manager.update(dt)

        # ── Physics (body moves only from tentacle forces) ──────────
        octopus.update(dt, WIDTH, HEIGHT)
        if agent:
            obs = octopus.get_observation()

        step += 1

        # ── Render ──────────────────────────────────────────────────
        renderer.draw(octopus, step, manager=manager, debug=debug)
        pygame.display.flip()

    pygame.quit()
    sys.exit()


if __name__ == "__main__":
    main()
