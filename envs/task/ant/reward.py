import numpy as np


class AntReward:
    def __init__(
        self,
        forward_weight,
        ctrl_cost_weight,
        contact_cost_weight,
        healthy_reward,
        contact_force_range,
    ):
        self.forward_weight = forward_weight
        self.ctrl_cost_weight = ctrl_cost_weight
        self.contact_cost_weight = contact_cost_weight
        self.healthy_reward = healthy_reward
        self.contact_force_range = contact_force_range

    def control_cost(self, action):
        return self.ctrl_cost_weight * np.sum(np.square(action))

    def contact_cost(self, cfrc_ext):
        min_v, max_v = self.contact_force_range
        clipped = np.clip(cfrc_ext, min_v, max_v)
        return self.contact_cost_weight * np.sum(np.square(clipped))

    def compute(self, x_velocity, action, cfrc_ext, healthy):
        forward = self.forward_weight * x_velocity
        survive = self.healthy_reward * int(healthy)

        ctrl_cost = self.control_cost(action)
        contact_cost = self.contact_cost(cfrc_ext)

        reward = forward + survive - ctrl_cost - contact_cost

        info = {
            "reward_forward": forward,
            "reward_survive": survive,
            "reward_ctrl": -ctrl_cost,
            "reward_contact": -contact_cost,
        }
        return reward, info
