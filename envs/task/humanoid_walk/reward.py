import numpy as np


class HumanoidReward:
    def __init__(
        self,
        forward_reward_weight,
        ctrl_cost_weight,
        contact_cost_weight,
        contact_cost_range,
        healthy_reward,
    ):
        self.forward_reward_weight = forward_reward_weight
        self.ctrl_cost_weight = ctrl_cost_weight
        self.contact_cost_weight = contact_cost_weight
        self.contact_cost_range = contact_cost_range
        self.healthy_reward_value = healthy_reward

    def control_cost(self, data):
        return self.ctrl_cost_weight * np.sum(np.square(data.ctrl))

    def contact_cost(self, data):
        cost = self.contact_cost_weight * np.sum(np.square(data.cfrc_ext))
        return np.clip(cost, *self.contact_cost_range)

    def healthy_reward(self, is_healthy: bool):
        return self.healthy_reward_value if is_healthy else 0.0

    def compute(self, x_velocity, is_healthy, data):
        forward_reward = self.forward_reward_weight * x_velocity
        healthy_reward = self.healthy_reward(is_healthy)

        ctrl_cost = self.control_cost(data)
        contact_cost = self.contact_cost(data)

        reward = forward_reward + healthy_reward - ctrl_cost - contact_cost

        info = {
            "reward_survive": healthy_reward,
            "reward_forward": forward_reward,
            "reward_ctrl": -ctrl_cost,
            "reward_contact": -contact_cost,
        }
        return reward, info
