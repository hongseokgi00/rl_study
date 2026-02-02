import argparse
import os
import numpy as np
import matplotlib.pyplot as plt


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--algo", type=str, required=True,
                   help="a2c | ppo | sac | ddpg")
    return p.parse_args()


def main():
    cfg = parse_args()
    algo = cfg.algo.lower()

    reward_path = f"results/{algo}/rewards.npy"

    if not os.path.exists(reward_path):
        raise FileNotFoundError(f"Reward file not found: {reward_path}")

    rewards = np.load(reward_path)

    plt.figure(figsize=(8, 4))
    plt.plot(rewards, label=f"{algo.upper()} Episode Reward")
    plt.xlabel("Episode")
    plt.ylabel("Reward")
    plt.title(f"Training Reward ({algo.upper()})")
    plt.legend()
    plt.grid()
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
