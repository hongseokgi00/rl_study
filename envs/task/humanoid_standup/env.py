__credits__ = ["Kallinteris-Andreas"]

import numpy as np
from gymnasium import utils
from gymnasium.envs.mujoco import MujocoEnv
from gymnasium.spaces import Box

from .config import *
from .reward import HumanoidStandupReward
from .observation import HumanoidStandupObservation



class HumanoidStandupEnv(MujocoEnv, utils.EzPickle):
    metadata = {
        "render_modes": ["human", "rgb_array", "depth_array", "rgbd_tuple"],
    }

    def __init__(
        self,
        xml_file: str = "humanoidstandup.xml",
        frame_skip: int = DEFAULT_FRAME_SKIP,
        default_camera_config: dict = DEFAULT_CAMERA_CONFIG,
        uph_cost_weight: float = DEFAULT_UPH_COST_WEIGHT,
        ctrl_cost_weight: float = DEFAULT_CTRL_COST_WEIGHT,
        impact_cost_weight: float = DEFAULT_IMPACT_COST_WEIGHT,
        impact_cost_range: tuple = DEFAULT_IMPACT_COST_RANGE,
        reset_noise_scale: float = DEFAULT_RESET_NOISE_SCALE,
        exclude_current_positions_from_observation: bool = True,
        include_cinert_in_observation: bool = True,
        include_cvel_in_observation: bool = True,
        include_qfrc_actuator_in_observation: bool = True,
        include_cfrc_ext_in_observation: bool = True,
        **kwargs,
    ):
        utils.EzPickle.__init__(self, **locals())

        self.reset_noise_scale = reset_noise_scale

        self.reward_fn = HumanoidStandupReward(
            uph_cost_weight,
            ctrl_cost_weight,
            impact_cost_weight,
            impact_cost_range,
        )

        self.obs_fn = HumanoidStandupObservation(
            exclude_current_positions_from_observation,
            include_cinert_in_observation,
            include_cvel_in_observation,
            include_qfrc_actuator_in_observation,
            include_cfrc_ext_in_observation,
        )



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

        z_pos = self.data.qpos[2]
        reward, reward_info = self.reward_fn.compute(
            z_pos, action, self.model, self.data
        )

        terminated = self.termination_fn.check(self.data)

        info = {
            "x_position": self.data.qpos[0],
            "y_position": self.data.qpos[1],
            "z_distance_from_origin": z_pos - self.init_qpos[2],
            "tendon_length": self.data.ten_length,
            "tendon_velocity": self.data.ten_velocity,
            **reward_info,
        }

        if self.render_mode == "human":
            self.render()

        return self.obs_fn.get(self.data), reward, terminated, False, info

    def reset_model(self):
        noise_low = -self.reset_noise_scale
        noise_high = self.reset_noise_scale

        qpos = self.init_qpos + self.np_random.uniform(
            noise_low, noise_high, size=self.model.nq
        )
        qvel = self.init_qvel + self.np_random.uniform(
            noise_low, noise_high, size=self.model.nv
        )

        self.set_state(qpos, qvel)
        return self.obs_fn.get(self.data)

    def _get_reset_info(self):
        return {
            "x_position": self.data.qpos[0],
            "y_position": self.data.qpos[1],
            "z_distance_from_origin": self.data.qpos[2] - self.init_qpos[2],
            "tendon_length": self.data.ten_length,
            "tendon_velocity": self.data.ten_velocity,
        }
