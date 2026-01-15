__credits__ = ["Kallinteris-Andreas"]

import numpy as np
from gymnasium import utils
from gymnasium.envs.mujoco import MujocoEnv
from gymnasium.spaces import Box

from .config import (
    DEFAULT_CAMERA_CONFIG,
    DEFAULT_FRAME_SKIP,
    DEFAULT_HEALTHY_REWARD,
    DEFAULT_RESET_NOISE_SCALE,
    TIP_HEIGHT_THRESHOLD,
)
from observation import InvertedDoublePendulumObservation
from termination import InvertedDoublePendulumTermination
from reward import InvertedDoublePendulumReward


class InvertedDoublePendulumEnv(MujocoEnv, utils.EzPickle):
    metadata = {
        "render_modes": ["human", "rgb_array", "depth_array", "rgbd_tuple"],
    }

    def __init__(
        self,
        xml_file: str = "inverted_double_pendulum.xml",
        frame_skip: int = DEFAULT_FRAME_SKIP,
        default_camera_config: dict = DEFAULT_CAMERA_CONFIG,
        healthy_reward: float = DEFAULT_HEALTHY_REWARD,
        reset_noise_scale: float = DEFAULT_RESET_NOISE_SCALE,
        **kwargs,
    ):
        utils.EzPickle.__init__(
            self, xml_file, frame_skip, reset_noise_scale, **kwargs
        )

        self.reset_noise_scale = reset_noise_scale

        self.obs_fn = InvertedDoublePendulumObservation()
        self.term_fn = InvertedDoublePendulumTermination(TIP_HEIGHT_THRESHOLD)
        self.reward_fn = InvertedDoublePendulumReward(healthy_reward)

        MujocoEnv.__init__(
            self,
            xml_file,
            frame_skip,
            observation_space=None,
            default_camera_config=default_camera_config,
            **kwargs,
        )

        self.observation_space = Box(
            low=-np.inf,
            high=np.inf,
            shape=(self.obs_fn.obs_size(),),
            dtype=np.float64,
        )

        self.metadata["render_fps"] = int(np.round(1.0 / self.dt))

    def step(self, action):
        self.do_simulation(action, self.frame_skip)

        x, _, y = self.data.site_xpos[0]
        observation = self.obs_fn.get(self.data)

        terminated = self.term_fn(y)

        v1, v2 = self.data.qvel[1:3]
        reward, reward_info = self.reward_fn.compute(
            x, y, v1, v2, terminated
        )

        if self.render_mode == "human":
            self.render()

        return observation, reward, terminated, False, reward_info

    def reset_model(self):
        noise_low = -self.reset_noise_scale
        noise_high = self.reset_noise_scale

        self.set_state(
            self.init_qpos
            + self.np_random.uniform(
                low=noise_low, high=noise_high, size=self.model.nq
            ),
            self.init_qvel
            + self.np_random.standard_normal(self.model.nv)
            * self.reset_noise_scale,
        )
        return self.obs_fn.get(self.data)
