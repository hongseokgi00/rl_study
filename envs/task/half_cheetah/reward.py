import numpy as np


class HalfCheetahReward:
    def __init__(self, forward_reward_weight: float, ctrl_cost_weight: float):
        self.forward_reward_weight = forward_reward_weight
        self.ctrl_cost_weight = ctrl_cost_weight

    def control_cost(self, action: np.ndarray) -> float:
        return self.ctrl_cost_weight * np.sum(np.square(action))

    def compute(self, x_velocity: float, action: np.ndarray):
        forward_reward = self.forward_reward_weight * x_velocity
        ctrl_cost = self.control_cost(action)

        reward = forward_reward - ctrl_cost

        info = {
            "reward_forward": forward_reward,
            "reward_ctrl": -ctrl_cost,
        }
        return reward, info
