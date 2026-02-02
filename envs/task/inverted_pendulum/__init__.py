import gymnasium as gym
from .env import InvertedPendulumEnv

gym.register(
    id="CustomInvertedPendulum-v0",
    entry_point="task.inverted_pendulum:InvertedPendulumEnv",
)