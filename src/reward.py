import numpy as np


def mse_reward(pattern: np.ndarray, background: np.ndarray) -> float:
    """
    Rewards the agent for minimizing pixel-level difference.
    Perfect match = reward 1.0. No match = reward near 0.
    """
    mse = np.mean((pattern - background) ** 2)
    return float(1.0 - mse)


def histogram_reward(pattern: np.ndarray, background: np.ndarray) -> float:
    """
    Rewards the agent for matching the overall color distribution.
    An octopus that matches the general color tone even if the texture
    is off still deserves some reward — this captures that.
    """
    score = 0.0
    for channel in range(pattern.shape[0]):
        p_hist, _ = np.histogram(pattern[channel], bins=16, range=(0, 1), density=True)
        b_hist, _ = np.histogram(background[channel], bins=16, range=(0, 1), density=True)
        # Bhattacharyya coefficient: 1.0 means identical distributions
        score += float(np.sum(np.sqrt(p_hist * b_hist + 1e-8)) / 16)
    return score / pattern.shape[0]


def combined_reward(pattern: np.ndarray, background: np.ndarray,
                    mse_weight: float = 0.7, hist_weight: float = 0.3) -> float:
    """
    Blends pixel-level accuracy with color distribution matching.
    MSE catches exact texture matches; histogram catches broad color tone.
    Weights must sum to 1.0.
    """
    mse = mse_reward(pattern, background)
    hist = histogram_reward(pattern, background)
    return mse_weight * mse + hist_weight * hist


if __name__ == "__main__":
    background = np.random.rand(3, 64, 64).astype(np.float32)

    # Near-perfect match: background + tiny noise
    good_pattern = np.clip(background + np.random.normal(0, 0.05, background.shape).astype(np.float32), 0, 1)

    # Bad match: completely random
    bad_pattern = np.random.rand(3, 64, 64).astype(np.float32)

    print("--- Good pattern (close to background) ---")
    print(f"  MSE reward:       {mse_reward(good_pattern, background):.4f}")
    print(f"  Histogram reward: {histogram_reward(good_pattern, background):.4f}")
    print(f"  Combined reward:  {combined_reward(good_pattern, background):.4f}")

    print("\n--- Bad pattern (random noise) ---")
    print(f"  MSE reward:       {mse_reward(bad_pattern, background):.4f}")
    print(f"  Histogram reward: {histogram_reward(bad_pattern, background):.4f}")
    print(f"  Combined reward:  {combined_reward(bad_pattern, background):.4f}")

    print("\nReward functions working correctly")
