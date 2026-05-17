import gymnasium as gym
import numpy as np
import torch
from gymnasium import spaces
from .generator import Generator
from .reward import combined_reward


class OctopusEnv(gym.Env):
    """
    The world the RL agent lives in.

    Each episode:
      - A background image is loaded (what the octopus needs to blend into)
      - The agent adjusts a latent vector step by step
      - The generator turns that latent vector into a camouflage pattern
      - The reward is how closely the pattern matches the background
    """

    def __init__(self, latent_dim=64, image_size=64, max_steps=50):
        super(OctopusEnv, self).__init__()

        self.latent_dim = latent_dim
        self.image_size = image_size
        self.max_steps = max_steps
        self.current_step = 0

        # Load the generator (frozen — the RL agent does not train this)
        self.generator = Generator(latent_dim=latent_dim, output_size=image_size)
        self.generator.eval()

        # The background is a random noise image for now.
        # Later we'll swap this for real underwater photos.
        self.background = None

        # Current latent vector — the agent tweaks this each step
        self.latent_vector = None

        # ACTION SPACE: the agent outputs a small adjustment to the latent vector.
        # Each of the 64 values can shift between -1 and +1 per step.
        self.action_space = spaces.Box(
            low=-1.0, high=1.0, shape=(latent_dim,), dtype=np.float32
        )

        # OBSERVATION SPACE: background + current pattern flattened into a 1D vector.
        # MlpPolicy works on flat vectors, not image tensors.
        # 6 channels * 64 * 64 pixels = 24,576 values
        self.observation_space = spaces.Box(
            low=0.0, high=1.0,
            shape=(6 * image_size * image_size,),
            dtype=np.float32
        )

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.current_step = 0

        # Generate a new random background for this episode.
        # This is random noise for now — will be replaced with real images later.
        self.background = np.random.rand(3, self.image_size, self.image_size).astype(np.float32)

        # Start the latent vector from random noise
        self.latent_vector = np.random.randn(self.latent_dim).astype(np.float32)

        obs = self._get_observation()
        return obs, {}

    def step(self, action):
        self.current_step += 1

        # Apply the agent's action: shift the latent vector by the action values
        self.latent_vector = np.clip(self.latent_vector + action * 0.1, -3.0, 3.0)

        # Generate the current camouflage pattern from the updated latent vector
        pattern = self._generate_pattern()

        # Score how well the pattern matches the background
        reward = self._compute_reward(pattern)

        # Episode ends when we hit max steps
        terminated = self.current_step >= self.max_steps
        truncated = False

        obs = self._get_observation()
        return obs, reward, terminated, truncated, {}

    def _generate_pattern(self):
        with torch.no_grad():
            z = torch.tensor(self.latent_vector).unsqueeze(0)
            pattern = self.generator(z)
            # Shift from [-1, 1] (Tanh output) to [0, 1] for display and comparison
            pattern = (pattern.squeeze(0).numpy() + 1) / 2
        return pattern.astype(np.float32)

    def _compute_reward(self, pattern):
        return combined_reward(pattern, self.background)

    def _get_observation(self):
        pattern = self._generate_pattern()
        obs = np.concatenate([self.background, pattern], axis=0)
        return obs.flatten().astype(np.float32)


if __name__ == "__main__":
    env = OctopusEnv()

    obs, _ = env.reset()
    print(f"Observation shape: {obs.shape}")
    print(f"Action space: {env.action_space}")

    # Simulate a few random steps
    total_reward = 0
    for step in range(5):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, _ = env.step(action)
        total_reward += reward
        print(f"Step {step + 1} | Reward: {reward:.4f}")

    print(f"\nTotal reward over 5 steps: {total_reward:.4f}")
    print("Environment working correctly")
