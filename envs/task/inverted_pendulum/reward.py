class InvertedPendulumReward:
    def compute(self, terminated: bool):
        reward = int(not terminated)
        info = {"reward_survive": reward}
        return reward, info
