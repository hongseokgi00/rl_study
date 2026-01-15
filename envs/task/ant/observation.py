import numpy as np


class AntObservation:
    def __init__(self, exclude_xy=True, include_cfrc_ext=True):
        self.exclude_xy = exclude_xy
        self.include_cfrc_ext = include_cfrc_ext

    def get(self, data, contact_forces):
        position = data.qpos.flatten()
        velocity = data.qvel.flatten()

        if self.exclude_xy:
            position = position[2:]

        if self.include_cfrc_ext:
            return np.concatenate((position, velocity, contact_forces[1:].flatten()))
        else:
            return np.concatenate((position, velocity))

    def obs_size(self, data):
        size = data.qpos.size + data.qvel.size
        if self.exclude_xy:
            size -= 2
        if self.include_cfrc_ext:
            size += data.cfrc_ext[1:].size
        return size
