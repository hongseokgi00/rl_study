import numpy as np


class HumanoidObservation:
    def __init__(
        self,
        exclude_xy,
        include_cinert,
        include_cvel,
        include_qfrc_actuator,
        include_cfrc_ext,
    ):
        self.exclude_xy = exclude_xy
        self.include_cinert = include_cinert
        self.include_cvel = include_cvel
        self.include_qfrc_actuator = include_qfrc_actuator
        self.include_cfrc_ext = include_cfrc_ext

    def get(self, data):
        qpos = data.qpos.flatten()
        qvel = data.qvel.flatten()

        if self.exclude_xy:
            qpos = qpos[2:]

        parts = [qpos, qvel]

        if self.include_cinert:
            parts.append(data.cinert[1:].flatten())
        if self.include_cvel:
            parts.append(data.cvel[1:].flatten())
        if self.include_qfrc_actuator:
            parts.append(data.qfrc_actuator[6:].flatten())
        if self.include_cfrc_ext:
            parts.append(data.cfrc_ext[1:].flatten())

        return np.concatenate(parts)

    def obs_size(self, data):
        size = data.qpos.size + data.qvel.size
        size -= 2 * self.exclude_xy
        size += data.cinert[1:].size * self.include_cinert
        size += data.cvel[1:].size * self.include_cvel
        size += (data.qvel.size - 6) * self.include_qfrc_actuator
        size += data.cfrc_ext[1:].size * self.include_cfrc_ext
        return size

    def structure(self, data):
        return {
            "skipped_qpos": 2 * self.exclude_xy,
            "qpos": data.qpos.size - 2 * self.exclude_xy,
            "qvel": data.qvel.size,
            "cinert": data.cinert[1:].size * self.include_cinert,
            "cvel": data.cvel[1:].size * self.include_cvel,
            "qfrc_actuator": (data.qvel.size - 6)
            * self.include_qfrc_actuator,
            "cfrc_ext": data.cfrc_ext[1:].size * self.include_cfrc_ext,
            "ten_length": 0,
            "ten_velocity": 0,
        }
