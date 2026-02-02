import argparse
import os
import time
import numpy as np
import torch

from algorithms.model_free.on_policy.a2c import A2CAgent
from algorithms.model_free.on_policy.ppo import PPOAgent
from algorithms.model_free.off_policy.sac import SACAgent
from algorithms.model_free.off_policy.ddpg import DDPGAgent
from envs.make_env import make_env, EnvConfig


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--env-id", type=str, default="CustomInvertedPendulum-v0")
    p.add_argument("--algo", type=str, required=True)
    p.add_argument("--episodes", type=int, default=10)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--sleep", type=float, default=0.03)
    return p.parse_args()


def load_weights(agent, algo):
    base = f"results/{algo}"

    if algo == "a2c":
        agent.actor.load_state_dict(torch.load(f"{base}/actor.pt", map_location=agent.device))
        agent.critic.load_state_dict(torch.load(f"{base}/critic.pt", map_location=agent.device))

    elif algo == "ppo":
        agent.actor.load_state_dict(torch.load(f"{base}/actor.pt", map_location="cpu"))
        agent.critic.load_state_dict(torch.load(f"{base}/critic.pt", map_location="cpu"))

    elif algo == "sac":
        agent.actor.load_state_dict(torch.load(f"{base}/actor.pt", map_location="cpu"))
        agent.q1.load_state_dict(torch.load(f"{base}/q1.pt", map_location="cpu"))
        agent.q2.load_state_dict(torch.load(f"{base}/q2.pt", map_location="cpu"))

    elif algo == "ddpg":
        agent.actor.load_state_dict(torch.load(f"{base}/actor.pt", map_location="cpu"))
        agent.critic.load_state_dict(torch.load(f"{base}/critic.pt", map_location="cpu"))

    else:
        raise ValueError(algo)

    print(f"[INFO] Loaded weights from {base}")


@torch.no_grad()
def get_action(agent, algo, state):
    s = torch.FloatTensor(state).unsqueeze(0)

    if algo in ["a2c", "ppo"]:
        mu, _ = agent.actor(s)
        return mu.squeeze(0).cpu().numpy()

    if algo == "sac":
        mu, _ = agent.actor(s)
        return mu.squeeze(0).cpu().numpy()

    if algo == "ddpg":
        return agent.actor(s).squeeze(0).cpu().numpy()

    raise ValueError(algo)


def make_agent(cfg, env, device):
    algo = cfg.algo.lower()

    if algo == "a2c":
        return A2CAgent(env=env, device=device)

    if algo == "ppo":
        return PPOAgent(env=env)

    if algo == "sac":
        return SACAgent(env=env)

    if algo == "ddpg":
        return DDPGAgent(env=env)

    raise ValueError(algo)


def main():
    cfg = parse_args()
    algo = cfg.algo.lower()

    device = torch.device(
        cfg.device if cfg.device == "cuda" and torch.cuda.is_available() else "cpu"
    )

    env_cfg = EnvConfig(
        env_id=cfg.env_id,
        seed=cfg.seed,
        render_mode="human",
    )
    env = make_env(env_cfg)

    agent = make_agent(cfg, env, device)
    load_weights(agent, algo)

    print(f"[INFO] Play {algo.upper()} on {cfg.env_id}")

    for ep in range(cfg.episodes):
        state, _ = env.reset(seed=cfg.seed + ep)
        done = False
        ep_reward = 0.0
        steps = 0

        while not done:
            action = get_action(agent, algo, state)
            state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            ep_reward += reward
            steps += 1
            time.sleep(cfg.sleep)

        print(f"[EP {ep+1}] reward={ep_reward:.2f} steps={steps}")

    env.close()
    print("[INFO] Play finished")


if __name__ == "__main__":
    main()
