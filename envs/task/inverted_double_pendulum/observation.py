import numpy as np


class InvertedDoublePendulumObservation:
    def get(self, data):
        return np.concatenate(
            [
                data.qpos[:1],                 # cart x
                np.sin(data.qpos[1:]),         # sin(theta1, theta2)
                np.cos(data.qpos[1:]),         # cos(theta1, theta2)
                np.clip(data.qvel, -10, 10),   # velocities
                np.clip(data.qfrc_constraint, -10, 10)[:1],
            ]
        ).ravel()

    def obs_size(self):
        return 9

