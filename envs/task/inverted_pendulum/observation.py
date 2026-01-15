import numpy as np


class InvertedPendulumObservation:
    def get(self, data):
        return np.concatenate([data.qpos, data.qvel]).ravel()

    def obs_size(self, data):
        return data.qpos.size + data.qvel.size

    def structure(self, data):
        return {
            "qpos": data.qpos.size,
            "qvel": data.qvel.size,
        }
