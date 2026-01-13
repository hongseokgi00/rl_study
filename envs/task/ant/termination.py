import numpy as np


class AntTermination:
    def __init__(self, healthy_z_range=(0.2, 1.0)):
        self.min_z, self.max_z = healthy_z_range

    def __call__(self, env):
        state = env.state_vector()
        z = state[2]
        return not (np.isfinite(state).all() and self.min_z <= z <= self.max_z)
