__credits__ = ["Kallinteris-Andreas", "Rushiv Arora"]

import numpy as np
from gymnasium import utils
from gymnasium.envs.mujoco import MujocoEnv
from gymnasium.spaces import Box

from .config import (
    DEFAULT_CAMERA_CONFIG,
    DEFAULT_FORWARD_REWARD_WEIGHT,
    DEFAULT_CTRL_COST_WEIGHT,
    DEFAULT_RESET_NOISE_SCALE,
    DEFAULT_FRAME_SKIP,
)
from reward import HalfCheetahReward
from observation import HalfCheetahObservation


class HalfCheetahEnv(MujocoEnv, utils.EzPickle):
    metadata = {
        "render_modes": ["human", "rgb_array", "depth_array", "rgbd_tuple"],
    }

    def __init__(
        self,
        xml_file: str = "half_cheetah.xml",
        frame_skip: int = DEFAULT_FRAME_SKIP,
        default_camera_config: dict = DEFAULT_CAMERA_CONFIG,
        forward_reward_weight: float = DEFAULT_FORWARD_REWARD_WEIGHT,
        ctrl_cost_weight: float = DEFAULT_CTRL_COST_WEIGHT,
        reset_noise_scale: float = DEFAULT_RESET_NOISE_SCALE,
        exclude_current_positions_from_observation: bool = True,
        **kwargs,
    ):
        utils.EzPickle.__init__(
            self,
            xml_file,
            frame_skip,
            default_camera_config,
            forward_reward_weight,
            ctrl_cost_weight,
            reset_noise_scale,
            exclude_current_positions_from_observation,
            **kwargs,
        )

        self.reset_noise_scale = reset_noise_scale

        self.reward_fn = HalfCheetahReward(
            forward_reward_weight, ctrl_cost_weight
        )
        self.obs_fn = HalfCheetahObservation(
            exclude_current_positions_from_observation
        )

        MujocoEnv.__init__(
            self,
            xml_file,
            frame_skip,
            observation_space=None,
            default_camera_config=default_camera_config,
            **kwargs,
        )

        obs_size = self.obs_fn.obs_size(self.model, self.data)
        self.observation_space = Box(
            low=-np.inf, high=np.inf, shape=(obs_size,), dtype=np.float64
        )

        self.observation_structure = self.obs_fn.structure(self.data)

        self.metadata["render_fps"] = int(np.round(1.0 / self.dt))

    def step(self, action):
        x_before = self.data.qpos[0]
        self.do_simulation(action, self.frame_skip)
        x_after = self.data.qpos[0]

        x_velocity = (x_after - x_before) / self.dt

        observation = self.obs_fn.get(self.data)
        reward, reward_info = self.reward_fn.compute(x_velocity, action)

        info = {
            "x_position": x_after,
            "x_velocity": x_velocity,
            **reward_info,
        }

        if self.render_mode == "human":
            self.render()

        return observation, reward, False, False, info

    def reset_model(self):
        noise_low = -self.reset_noise_scale
        noise_high = self.reset_noise_scale

        qpos = self.init_qpos + self.np_random.uniform(
            low=noise_low, high=noise_high, size=self.model.nq
        )
        qvel = self.init_qvel + self.reset_noise_scale * self.np_random.standard_normal(
            self.model.nv
        )

        self.set_state(qpos, qvel)
        return self.obs_fn.get(self.data)

    def _get_reset_info(self):
        return {
            "x_position": self.data.qpos[0],
        }
