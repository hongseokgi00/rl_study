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
)
from observation import InvertedPendulumObservation
from termination import InvertedPendulumTermination
from reward import InvertedPendulumReward


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
        **kwargs,
    ):
        utils.EzPickle.__init__(
            self, xml_file, frame_skip, reset_noise_scale, **kwargs
        )

        self.reset_noise_scale = reset_noise_scale

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

        obs_size = self.obs_fn.obs_size(self.data)
        self.observation_space = Box(
            low=-np.inf, high=np.inf, shape=(obs_size,), dtype=np.float64
        )

        self.observation_structure = self.obs_fn.structure(self.data)

        self.metadata["render_fps"] = int(np.round(1.0 / self.dt))

    def step(self, action):
        self.do_simulation(action, self.frame_skip)

        observation = self.obs_fn.get(self.data)
        terminated = self.term_fn(observation)

        reward, reward_info = self.reward_fn.compute(terminated)

        if self.render_mode == "human":
            self.render()

        return observation, reward, terminated, False, reward_info

    def reset_model(self):
        noise_low = -self.reset_noise_scale
        noise_high = self.reset_noise_scale

        qpos = self.init_qpos + self.np_random.uniform(
            low=noise_low, high=noise_high, size=self.model.nq
        )
        qvel = self.init_qvel + self.np_random.uniform(
            low=noise_low, high=noise_high, size=self.model.nv
        )

        self.set_state(qpos, qvel)
        return self.obs_fn.get(self.data)
