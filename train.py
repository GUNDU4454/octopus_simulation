"""
train.py  —  Train the locomotion agent

Usage
-----
  python train.py                          # train 200k steps
  python train.py --timesteps 1000000     # train 1M steps
  python train.py --mode evaluate         # evaluate saved agent
  python train.py --mode watch            # watch agent live
"""

import os
import argparse
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

from src.rl.env import LocomotionEnv


# ------------------------------------------------------------------
# Logger
# ------------------------------------------------------------------

class LocomotionLogger(BaseCallback):

    def __init__(self, log_every: int = 200):
        super().__init__()
        self.log_every  = log_every
        self._ep_reward = 0.0
        self._ep_dist   = 0.0
        self._episodes  = []

    def _on_step(self) -> bool:
        info = self.locals["infos"][0]
        self._ep_reward += float(self.locals["rewards"][0])
        self._ep_dist   += info.get("dist_moved", 0.0)

        if self.locals["dones"][0]:
            self._episodes.append((self._ep_reward, self._ep_dist))
            self._ep_reward = 0.0
            self._ep_dist   = 0.0
            n = len(self._episodes)
            if n % self.log_every == 0:
                recent  = self._episodes[-self.log_every:]
                avg_r   = sum(r for r,_ in recent) / len(recent)
                avg_d   = sum(d for _,d in recent) / len(recent)
                best_d  = max(d for _,d in recent)
                print(f"  ep {n:>6} | "
                      f"avg_reward {avg_r:8.1f} | "
                      f"avg_dist {avg_d:6.1f} | "
                      f"best_dist {best_d:6.1f}")
        return True


# ------------------------------------------------------------------
# Training
# ------------------------------------------------------------------

def train(timesteps: int = 200_000,
          save_path: str = "models/checkpoints/locomotion_agent"):

    print("=" * 55)
    print("  Octopus Top-Down Locomotion Training")
    print("  Goal: learn to crawl across the seabed")
    print("=" * 55)

    print("\nChecking environment...")
    env = LocomotionEnv()
    check_env(env, warn=True)
    env.close()

    # Wrap in vectorised env for normalisation
    def make_env():
        return LocomotionEnv()

    vec_env = DummyVecEnv([make_env])
    vec_env = VecNormalize(vec_env, norm_obs=True, norm_reward=True,
                           clip_obs=5.0)

    print("Building PPO agent...")
    agent = PPO(
        policy        = "MlpPolicy",
        env           = vec_env,
        learning_rate = 3e-4,
        n_steps       = 2048,
        batch_size    = 256,
        n_epochs      = 10,
        gamma         = 0.995,
        gae_lambda    = 0.95,
        clip_range    = 0.2,
        ent_coef      = 0.005,   # small entropy bonus encourages exploration
        verbose       = 0,
        device        = "auto",
        policy_kwargs = dict(net_arch=[256, 256, 128]),
    )

    print(f"\nTraining for {timesteps:,} steps. "
          f"Logging every 200 episodes.\n")

    callback = LocomotionLogger(log_every=200)
    agent.learn(total_timesteps=timesteps, callback=callback)

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    agent.save(save_path)
    vec_env.save(save_path + "_vecnorm.pkl")
    print(f"\nAgent saved to {save_path}.zip")
    return agent, vec_env


# ------------------------------------------------------------------
# Evaluation
# ------------------------------------------------------------------

def evaluate(checkpoint: str = "models/checkpoints/locomotion_agent",
             episodes: int = 5):
    print(f"\nEvaluating {checkpoint} over {episodes} episodes...\n")

    env   = LocomotionEnv()
    agent = PPO.load(checkpoint, env=env)

    for ep in range(episodes):
        obs, _ = env.reset()
        total_r = 0.0
        total_d = 0.0
        done    = False
        while not done:
            action, _ = agent.predict(obs, deterministic=True)
            obs, r, term, trunc, info = env.step(action)
            total_r += r
            total_d += info.get("dist_moved", 0.0)
            done = term or trunc

        print(f"  ep {ep+1} | reward {total_r:8.1f} | "
              f"total distance {total_d:.1f} px")

    env.close()


# ------------------------------------------------------------------
# Entry point
# ------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Octopus Locomotion Trainer")
    parser.add_argument("--mode",       choices=["train","evaluate","watch"],
                        default="train")
    parser.add_argument("--timesteps",  type=int, default=200_000)
    parser.add_argument("--checkpoint", type=str,
                        default="models/checkpoints/locomotion_agent")
    args = parser.parse_args()

    if args.mode == "train":
        train(timesteps=args.timesteps, save_path=args.checkpoint)
        evaluate(checkpoint=args.checkpoint)

    elif args.mode == "evaluate":
        evaluate(checkpoint=args.checkpoint)

    elif args.mode == "watch":
        import subprocess, sys
        subprocess.run([sys.executable, "simulate.py",
                        "--checkpoint", args.checkpoint])


if __name__ == "__main__":
    main()
