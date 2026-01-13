

from dataclasses import dataclass
from typing import Optional

import gymnasium as gym



@dataclass
class EnvConfig:
    env_id: str
    seed: int = 0
    render_mode: Optional[str] = None

    # wrappers
    constant_reward: bool = False
    clip_action: bool = True



def make_env(cfg: EnvConfig):

    env = gym.make(
        cfg.env_id,
        render_mode=cfg.render_mode,
    )

    env.reset(seed=cfg.seed)
    env.action_space.seed(cfg.seed)

    if cfg.clip_action:
        env = gym.wrappers.ClipAction(env)  # policy는 기본 분포를 가정하고 학습 -> 가우시안 분포의 범위 (음의 무한, 양의 무한) 구간이므로 적절한 구간을 설정해야함

    return env
