"""
env.py  —  Top-down octopus locomotion RL environment

The agent learns to coordinate 8 arms to crawl across the seabed.

Observation  (79 values)   see Octopus.get_observation()
  7 body values + 9 per arm × 8 arms

Action       (24 values)   3 per arm: grip_cmd, reach_offset, contract_strength
  All values in [-1, 1]; octopus.apply_action maps them to biological ranges.

Reward shaping
--------------
  distance_reward   distance moved this step × 80     (primary signal)
  efficiency_bonus  bonus when ≥2 arms planted        (stable gait)
  smoothness        penalty for large angular velocity (wobbly = bad)
  time_bonus        small reward each step (stay alive)

Episode ends after MAX_STEPS or if octopus gets stuck (no movement).

The environment is intentionally minimal — no predators, no GAN —
just pure locomotion learning.  Once the agent learns to crawl,
complexity is added on top.
"""

import numpy as np
import math
import gymnasium as gym
from gymnasium import spaces

from src.entities.octopus import Octopus


class LocomotionEnv(gym.Env):

    metadata = {"render_modes": ["human"]}

    WIDTH     = 1200
    HEIGHT    = 800
    MAX_STEPS = 2000
    DT        = 1 / 60

    N_ARMS    = 8
    OBS_DIM   = 79            # 7 body + 9 per arm × 8 arms
    ACT_DIM   = N_ARMS * 3   # grip_cmd + reach_offset + contract_strength per arm

    def __init__(self, render_mode: str | None = None):
        super().__init__()
        self.render_mode = render_mode

        self.observation_space = spaces.Box(
            low=-2.0, high=2.0, shape=(self.OBS_DIM,), dtype=np.float32
        )
        self.action_space = spaces.Box(
            low=-1.0, high=1.0, shape=(self.ACT_DIM,), dtype=np.float32
        )

        self._octopus:   Octopus | None = None
        self._step_count = 0
        self._prev_pos   = np.zeros(2)
        self._stuck_timer = 0.0

        # Pygame (lazy init)
        self._screen   = None
        self._renderer = None
        self._clock    = None

    # ------------------------------------------------------------------
    # Gym interface
    # ------------------------------------------------------------------

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        rng = np.random.default_rng(seed)

        # Spawn octopus near centre with small random offset
        x = self.WIDTH  / 2 + float(rng.uniform(-80, 80))
        y = self.HEIGHT / 2 + float(rng.uniform(-80, 80))

        self._octopus    = Octopus(x, y)
        self._octopus.angle = float(rng.uniform(0, 2 * math.pi))
        self._step_count  = 0
        self._prev_pos    = self._octopus.pos.copy()
        self._stuck_timer = 0.0

        return self._octopus.get_observation(), {}

    def step(self, action: np.ndarray):
        assert self._octopus is not None, "Call reset() first."
        self._step_count += 1

        # Apply RL action
        self._octopus.apply_action(action)

        # Step physics
        self._octopus.update(self.DT, self.WIDTH, self.HEIGHT)

        # --- Reward ---
        current_pos  = self._octopus.pos
        delta        = current_pos - self._prev_pos
        dist_moved   = float(np.linalg.norm(delta))
        self._prev_pos = current_pos.copy()

        # Primary: distance covered
        reward = dist_moved * 80.0

        # Efficiency: reward using multiple arms
        planted = self._octopus.planted_count
        if planted >= 2:
            reward += planted * 0.4

        # Smoothness: penalise excessive spinning
        reward -= abs(self._octopus.angular_vel) * 0.8

        # Survival bonus
        reward += 0.05

        # Stuck detection
        if dist_moved < 0.05:
            self._stuck_timer += self.DT
            reward -= 0.3
        else:
            self._stuck_timer = 0.0

        # Termination
        terminated = False
        truncated  = (self._step_count >= self.MAX_STEPS
                      or self._stuck_timer > 6.0)

        if self.render_mode == "human":
            self.render()

        return (
            self._octopus.get_observation(),
            float(reward),
            terminated,
            truncated,
            {"dist_moved": dist_moved, "planted": planted},
        )

    # ------------------------------------------------------------------

    def render(self):
        import pygame
        if self._screen is None:
            pygame.init()
            self._screen = pygame.display.set_mode(
                (self.WIDTH, self.HEIGHT)
            )
            pygame.display.set_caption("Octopus Locomotion Training")
            from src.rendering.renderer import Renderer
            self._renderer = Renderer(self._screen, self.WIDTH, self.HEIGHT)
            self._clock    = pygame.time.Clock()

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.close()

        self._renderer.draw(self._octopus, self._step_count)
        pygame.display.flip()
        self._clock.tick(60)

    def close(self):
        if self._screen is not None:
            import pygame
            pygame.quit()
            self._screen = None
