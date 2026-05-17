import os
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.env_checker import check_env
from .environment import OctopusEnv


class TrainingLogger(BaseCallback):
    """
    Prints a summary every N episodes so you can watch the agent improve.
    Without this, training is completely silent.
    """

    def __init__(self, log_every: int = 500):
        super().__init__()
        self.log_every = log_every
        self.episode_rewards = []
        self.current_episode_reward = 0.0

    def _on_step(self) -> bool:
        reward = self.locals["rewards"][0]
        self.current_episode_reward += reward

        done = self.locals["dones"][0]
        if done:
            self.episode_rewards.append(self.current_episode_reward)
            self.current_episode_reward = 0.0

            if len(self.episode_rewards) % self.log_every == 0:
                recent = self.episode_rewards[-self.log_every:]
                avg = sum(recent) / len(recent)
                best = max(recent)
                print(f"  Episode {len(self.episode_rewards):>6} | "
                      f"Avg reward: {avg:.2f} | Best: {best:.2f}")

        return True


def build_agent(env: OctopusEnv, device: str = "auto") -> PPO:
    """
    Creates a PPO agent configured for the octopus environment.

    PPO (Proximal Policy Optimization) is a policy gradient algorithm.
    It learns by trying actions, observing rewards, and nudging its
    policy toward actions that led to higher rewards — but not too
    aggressively, which keeps training stable.
    """
    agent = PPO(
        policy="MlpPolicy",   # Multi-layer perceptron — suits our flat observation
        env=env,
        learning_rate=3e-4,   # How fast the agent updates its weights
        n_steps=2048,         # Steps collected before each policy update
        batch_size=64,        # Samples per gradient update
        n_epochs=10,          # How many passes over collected data per update
        gamma=0.99,           # Discount factor: how much future rewards matter
        verbose=0,            # We handle logging ourselves via the callback
        device=device,
    )
    return agent


def train(timesteps: int = 100_000, save_path: str = "models/checkpoints/octopus_agent"):
    print("Setting up environment...")
    env = OctopusEnv()

    print("Validating environment structure...")
    check_env(env, warn=True)

    print(f"Building PPO agent...")
    agent = build_agent(env)

    print(f"\nTraining for {timesteps:,} timesteps. "
          f"Progress logged every 500 episodes.\n")

    callback = TrainingLogger(log_every=500)
    agent.learn(total_timesteps=timesteps, callback=callback)

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    agent.save(save_path)
    print(f"\nAgent saved to {save_path}.zip")
    return agent


def load_agent(path: str = "models/checkpoints/octopus_agent") -> PPO:
    env = OctopusEnv()
    agent = PPO.load(path, env=env)
    print(f"Agent loaded from {path}.zip")
    return agent


if __name__ == "__main__":
    print("Running quick environment check...")
    env = OctopusEnv()
    check_env(env, warn=True)
    print("Environment passed all checks.")

    print("\nBuilding agent and running 3 test steps...")
    agent = build_agent(env)
    obs, _ = env.reset()

    for i in range(3):
        action, _ = agent.predict(obs, deterministic=False)
        obs, reward, terminated, truncated, _ = env.step(action)
        print(f"  Step {i + 1} | Action mean: {action.mean():.4f} | Reward: {reward:.4f}")

    print("\nAgent working correctly")
