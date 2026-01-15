import numpy as np


class AntTermination:
    def __init__(self, healthy_z_range, terminate_when_unhealthy=True):
        self.min_z, self.max_z = healthy_z_range
        self.terminate_when_unhealthy = terminate_when_unhealthy

    def __call__(self, env):
        state = env.state_vector()
        z = state[2]

        healthy = (
            np.isfinite(state).all()
            and self.min_z <= z <= self.max_z
        )

        terminated = (not healthy) and self.terminate_when_unhealthy
        return terminated, healthy
