from dataclasses import dataclass
from typing import Optional
import gymnasium as gym

from envs.task.inverted_pendulum import InvertedPendulumEnv


@dataclass
class EnvConfig:
    env_id: str
    seed: int = 0
    render_mode: Optional[str] = None

    # wrappers
    constant_reward: bool = False
    clip_action: bool = True


def make_env(cfg: EnvConfig):

    # 🔥 커스텀 env는 gym.make를 타지 않는다
    if cfg.env_id == "CustomInvertedPendulum-v0":
        print("[INFO] Using CUSTOM InvertedPendulumEnv")
        env = InvertedPendulumEnv(
            render_mode=cfg.render_mode,
        )
    else:
        print(f"[INFO] Using GYM env: {cfg.env_id}")
        env = gym.make(
            cfg.env_id,
            render_mode=cfg.render_mode,
        )

    # seed
    env.reset(seed=cfg.seed)
    env.action_space.seed(cfg.seed)

    # # wrappers
    # if cfg.clip_action:
    #     env = gym.wrappers.ClipAction(env)

    # 🔒 안전 확인 (강력 추천)
    print("[DEBUG] env class:", env.__class__)
    print("[DEBUG] env module:", env.__class__.__module__)

    return env
