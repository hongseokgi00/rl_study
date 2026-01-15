class InvertedDoublePendulumReward:
    def __init__(self, healthy_reward: float):
        self.healthy_reward = healthy_reward

    def compute(self, x, y, v1, v2, terminated: bool):
        dist_penalty = 0.01 * x**2 + (y - 2.0) ** 2
        vel_penalty = 1e-3 * v1**2 + 5e-3 * v2**2
        alive_bonus = self.healthy_reward * int(not terminated)

        reward = alive_bonus - dist_penalty - vel_penalty

        info = {
            "reward_survive": alive_bonus,
            "distance_penalty": -dist_penalty,
            "velocity_penalty": -vel_penalty,
        }
        return reward, info
