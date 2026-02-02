import numpy as np

class InvertedPendulumObservation:
    def get(self, data):
        x, theta = data.qpos
        x_dot, theta_dot = data.qvel

        obs = np.array(
            [
                x,
                np.sin(theta),
                np.cos(theta),
                x_dot,
                theta_dot,
            ],
            dtype=np.float32,
        )

        obs = np.clip(obs, -10.0, 10.0)

        # 🚨 NaN / Inf 완전 차단
        if not np.isfinite(obs).all():
            print("[OBS WARN] Non-finite observation detected. Zeroing obs.")
            obs = np.zeros_like(obs, dtype=np.float32)

        return obs

    def obs_size(self, data):
        return 5

    def structure(self, data):
        return {
            "cart_pos": 1,
            "sin_theta": 1,
            "cos_theta": 1,
            "cart_vel": 1,
            "pole_ang_vel": 1,
        }
