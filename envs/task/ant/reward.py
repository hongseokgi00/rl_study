import numpy as np


class AntReward:
    def __init__(
        self,
        forward_weight=1.0,
        ctrl_cost_weight=0.5,
        contact_cost_weight=5e-4,
        healthy_reward=1.0,
    ):
        self.forward_weight = forward_weight
        self.ctrl_cost_weight = ctrl_cost_weight
        self.contact_cost_weight = contact_cost_weight
        self.healthy_reward = healthy_reward

    def __call__(self, env, action, x_velocity):
        forward_reward = self.forward_weight * x_velocity
        healthy_reward = self.healthy_reward if env.is_healthy else 0.0

        ctrl_cost = self.ctrl_cost_weight * np.sum(action ** 2)
        contact_cost = self.contact_cost_weight * np.sum(
            np.square(env.contact_forces)
        )

        reward = forward_reward + healthy_reward - ctrl_cost - contact_cost

        info = {
            "reward_forward": forward_reward,
            "reward_survive": healthy_reward,
            "reward_ctrl": -ctrl_cost,
            "reward_contact": -contact_cost,
        }

        return reward, info
