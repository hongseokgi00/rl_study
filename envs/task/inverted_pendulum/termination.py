import numpy as np


class InvertedPendulumTermination:
    def __init__(self, angle_threshold: float):
        self.angle_threshold = angle_threshold

    def __call__(self, observation: np.ndarray):
        angle = observation[1]
        return (not np.isfinite(observation).all()) or (
            abs(angle) > self.angle_threshold
        )
