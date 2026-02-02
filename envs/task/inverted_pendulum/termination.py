import numpy as np


class InvertedPendulumTermination:
    def __init__(self, angle_threshold: float):
        self.angle_threshold = angle_threshold

    def __call__(self, observation: np.ndarray):
        # observation structure:
        # [cart_pos, sin_theta, cos_theta, cart_vel, pole_ang_vel]

        sin_theta = observation[1]
        cos_theta = observation[2]

        theta = np.arctan2(sin_theta, cos_theta)

        return (
            not np.isfinite(observation).all() # NaN, Inf 발생 시 즉시 종료
            or abs(theta) > self.angle_threshold # 허용 각도 이상 기울어지면 실패
        )
    


