import numpy as np


class AntObservation:
    def __init__(
        self,
        exclude_current_positions: bool = True,
        include_contact_forces: bool = True,
    ):
        self.exclude_current_positions = exclude_current_positions
        self.include_contact_forces = include_contact_forces

    def __call__(self, env):
        qpos = env.data.qpos.flatten()
        qvel = env.data.qvel.flatten()

        if self.exclude_current_positions:
            qpos = qpos[2:]

        obs = [qpos, qvel]

        if self.include_contact_forces:
            cfrc = env.data.cfrc_ext[1:].flatten()
            obs.append(cfrc)

        return np.concatenate(obs)
