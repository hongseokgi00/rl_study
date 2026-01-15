import numpy as np


class HumanoidTermination:
    def __init__(self, healthy_z_range, terminate_when_unhealthy=True):
        self.min_z, self.max_z = healthy_z_range
        self.terminate_when_unhealthy = terminate_when_unhealthy

    def is_healthy(self, data):
        z = data.qpos[2]
        return self.min_z < z < self.max_z

    def __call__(self, env):
        if not self.terminate_when_unhealthy:
            return False
        return not self.is_healthy(env.data)
