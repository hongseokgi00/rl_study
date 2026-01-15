import numpy as np


class HalfCheetahObservation:
    def __init__(self, exclude_current_positions: bool):
        self.exclude_current_positions = exclude_current_positions

    def get(self, data):
        position = data.qpos.flatten()
        velocity = data.qvel.flatten()

        if self.exclude_current_positions:
            position = position[1:]

        return np.concatenate((position, velocity)).ravel()

    def obs_size(self, model, data):
        return (
            data.qpos.size
            + data.qvel.size
            - int(self.exclude_current_positions)
        )

    def structure(self, data):
        return {
            "skipped_qpos": int(self.exclude_current_positions),
            "qpos": data.qpos.size - int(self.exclude_current_positions),
            "qvel": data.qvel.size,
        }
