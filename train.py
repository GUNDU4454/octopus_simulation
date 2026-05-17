import argparse
import os
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")  # Render without a display window
import matplotlib.pyplot as plt
import matplotlib.animation as animation

from src.environment import OctopusEnv
from src.agent import train, load_agent


def evaluate(agent, episodes: int = 5):
    """
    Runs the trained agent for a few episodes and prints the results.
    No learning happens here — this is purely observation.
    """
    env = OctopusEnv()
    print(f"\nEvaluating agent over {episodes} episodes...\n")

    for ep in range(episodes):
        obs, _ = env.reset()
        total_reward = 0.0
        done = False

        while not done:
            action, _ = agent.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, _ = env.step(action)
            total_reward += reward
            done = terminated or truncated

        print(f"  Episode {ep + 1} | Total reward: {total_reward:.2f}")


def save_demo_gif(agent, path: str = "outputs/gifs/camouflage_demo.gif"):
    """
    Runs one episode and saves an animated GIF showing the octopus
    camouflage pattern evolving step by step toward the background.
    """
    env = OctopusEnv()
    obs, _ = env.reset()

    background = env.background  # shape: (3, 64, 64)
    frames = []
    done = False

    while not done:
        action, _ = agent.predict(obs, deterministic=True)
        obs, _, terminated, truncated, _ = env.step(action)
        done = terminated or truncated

        # obs is now flat (24576,): reshape back and grab the pattern half
        obs_img = obs.reshape(6, env.image_size, env.image_size)
        pattern = obs_img[3:6]  # last 3 channels = current pattern
        frames.append(np.transpose(pattern, (1, 2, 0)))

    os.makedirs(os.path.dirname(path), exist_ok=True)

    fig, axes = plt.subplots(1, 2, figsize=(8, 4))
    bg_display = np.transpose(background, (1, 2, 0))
    axes[0].imshow(bg_display)
    axes[0].set_title("Background")
    axes[0].axis("off")

    im = axes[1].imshow(frames[0])
    axes[1].set_title("Octopus Pattern")
    axes[1].axis("off")

    def update(frame_idx):
        im.set_array(frames[frame_idx])
        axes[1].set_title(f"Octopus Pattern (step {frame_idx + 1})")
        return [im]

    anim = animation.FuncAnimation(
        fig, update, frames=len(frames), interval=100, blit=True
    )
    anim.save(path, writer="pillow", fps=10)
    plt.close()
    print(f"Demo GIF saved to {path}")


def main():
    parser = argparse.ArgumentParser(description="Octopus Camouflage RL Trainer")
    parser.add_argument("--mode", choices=["train", "evaluate", "demo"],
                        default="train", help="What to do")
    parser.add_argument("--timesteps", type=int, default=100_000,
                        help="Training timesteps (default: 100,000)")
    parser.add_argument("--checkpoint", type=str,
                        default="models/checkpoints/octopus_agent",
                        help="Path to save/load agent checkpoint")
    args = parser.parse_args()

    if args.mode == "train":
        print("=" * 50)
        print(" Octopus Camouflage — Training")
        print("=" * 50)
        agent = train(timesteps=args.timesteps, save_path=args.checkpoint)
        evaluate(agent)
        save_demo_gif(agent)

    elif args.mode == "evaluate":
        agent = load_agent(args.checkpoint)
        evaluate(agent)

    elif args.mode == "demo":
        agent = load_agent(args.checkpoint)
        save_demo_gif(agent)


if __name__ == "__main__":
    main()
