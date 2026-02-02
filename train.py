import argparse
import torch
import gymnasium as gym

from algorithms.model_free.on_policy.a2c import A2CAgent
from algorithms.model_free.on_policy.ppo import PPOAgent
from algorithms.model_free.off_policy.sac import SACAgent
from algorithms.model_free.off_policy.ddpg import DDPGAgent
from envs.make_env import make_env, EnvConfig

def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--env-id", type=str, default="CustomInvertedPendulum-v0")

    parser.add_argument("--algo", type=str, default="a2c")

    parser.add_argument("--episodes", type=int, default=1000)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--gamma", type=float, default=0.99)

    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="cuda")

    parser.add_argument("--render", action="store_true")

    return parser.parse_args()


def make_agent(cfg, env, device):
    algo = cfg.algo.lower()

    if algo == "a2c":
        return A2CAgent(
            env=env,
            lr=cfg.lr,
            gamma=cfg.gamma,
            device=device,
        )

    elif algo == "ppo":
        return PPOAgent(env=env)
    
    elif algo == "sac":
        return SACAgent(env = env)

    elif algo == "ddpg":
        return DDPGAgent(env = env)
    else:
        raise ValueError(f"Unknown algorithm: {cfg.algo}")
    

def main():
    cfg = parse_args()

    device = torch.device(
        cfg.device if cfg.device == "cuda" and torch.cuda.is_available() else "cpu"
    )
    print(f"[INFO] Using device: {device}")

    env_cfg = EnvConfig(
        env_id=cfg.env_id,
        seed=cfg.seed,
        render_mode="human" if cfg.render else None,
    )

    env = make_env(env_cfg)

    agent = make_agent(cfg, env, device)

    print(f"[INFO] Start training {cfg.algo.upper()} on {cfg.env_id}")
    agent.train(max_episodes=cfg.episodes)

    env.close()
    print("[INFO] Training finished")
if __name__ == "__main__":
    main()
