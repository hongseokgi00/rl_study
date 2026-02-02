import numpy as np

class InvertedPendulumReward:
    def compute(self, pole_angle: float, terminated: bool):
        if terminated:
            return 0.0, {"terminated": True}

        balance = np.cos(pole_angle)
        reward = 0.1 + max(balance, 0.0)

        return reward, {
            "reward_balance": balance,
            "pole_angle": pole_angle,
        }
