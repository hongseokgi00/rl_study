__credits__ = ["Kallinteris-Andreas"]

import numpy as np
from gymnasium import utils
from gymnasium.envs.mujoco import MujocoEnv
from gymnasium.spaces import Box

from .config import (
    DEFAULT_CAMERA_CONFIG,
    DEFAULT_FRAME_SKIP,
    DEFAULT_RESET_NOISE_SCALE,
    ANGLE_THRESHOLD,
    MAX_STEP,
)
from .observation import InvertedPendulumObservation
from .termination import InvertedPendulumTermination
from .reward import InvertedPendulumReward


class InvertedPendulumEnv(MujocoEnv, utils.EzPickle):
    metadata = {
        "render_modes": ["human", "rgb_array", "depth_array", "rgbd_tuple"],
    }

    def __init__(
        self,
        xml_file: str = "inverted_pendulum.xml",
        frame_skip: int = DEFAULT_FRAME_SKIP,
        default_camera_config: dict = DEFAULT_CAMERA_CONFIG,
        reset_noise_scale: float = DEFAULT_RESET_NOISE_SCALE,
        max_step: int = MAX_STEP,
        **kwargs,
    ):
        utils.EzPickle.__init__(
            self, xml_file, frame_skip, reset_noise_scale, **kwargs
        )

        self.reset_noise_scale = reset_noise_scale
        self.max_step = max_step
        self.cur_step = 0

        self.obs_fn = InvertedPendulumObservation()
        self.term_fn = InvertedPendulumTermination(ANGLE_THRESHOLD)
        self.reward_fn = InvertedPendulumReward()

        MujocoEnv.__init__(
            self,
            xml_file,
            frame_skip,
            observation_space=None,
            default_camera_config=default_camera_config,
            **kwargs,
        )

        print("[INFO] Using CUSTOM InvertedPendulumEnv")

        obs_size = self.obs_fn.obs_size(self.data)
        self.observation_space = Box(
            low=-np.inf, high=np.inf, shape=(obs_size,), dtype=np.float32
        )

        self.observation_structure = self.obs_fn.structure(self.data)
        self.metadata["render_fps"] = int(np.round(1.0 / self.dt))

    def step(self, action):
        self.cur_step += 1

        self.do_simulation(action, self.frame_skip)
        observation = self.obs_fn.get(self.data)

        if not np.isfinite(observation).all():
            return observation, 0.0, True, False, {"nan_observation": True}

        terminated = self.term_fn(observation)

        cart_pos = observation[0]
        cart_vel = observation[3]
        if abs(cart_pos) > 2.4 or abs(cart_vel) > 10.0:
            terminated = True

        sin_theta = observation[1]
        cos_theta = observation[2]
        pole_angle = np.arctan2(sin_theta, cos_theta)

        if not np.isfinite(pole_angle):
            return observation, 0.0, True, False, {"nan_angle": True}

        reward, reward_info = self.reward_fn.compute(
            pole_angle=pole_angle,
            terminated=terminated,
        )

        truncated = (not terminated) and (self.cur_step >= self.max_step)

        if self.render_mode == "human":
            self.render()

        return observation, reward, terminated, truncated, reward_info

    def reset_model(self):
        self.cur_step = 0

        noise_low = -self.reset_noise_scale
        noise_high = self.reset_noise_scale

        qpos = self.init_qpos + self.np_random.uniform(
            low=noise_low, high=noise_high, size=self.model.nq
        )
        qvel = self.init_qvel + self.np_random.uniform(
            low=noise_low, high=noise_high, size=self.model.nv
        )

        qpos = np.clip(qpos, -0.5, 0.5)
        qvel = np.clip(qvel, -1.0, 1.0)

        self.set_state(qpos, qvel)
        observation = self.obs_fn.get(self.data)

        if not np.isfinite(observation).all():
            observation = np.zeros_like(observation, dtype=np.float32)

        return observation
