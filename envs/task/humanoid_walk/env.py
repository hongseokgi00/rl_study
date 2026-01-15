__credits__ = ["Kallinteris-Andreas"]

import numpy as np
from gymnasium import utils
from gymnasium.envs.mujoco import MujocoEnv
from gymnasium.spaces import Box

from config import *
from utils import mass_center
from reward import HumanoidReward
from observation import HumanoidObservation
from termination import HumanoidTermination


class HumanoidEnv(MujocoEnv, utils.EzPickle):
    metadata = {
        "render_modes": ["human", "rgb_array", "depth_array", "rgbd_tuple"],
    }

    def __init__(
        self,
        xml_file="humanoid.xml",
        frame_skip=DEFAULT_FRAME_SKIP,
        default_camera_config=DEFAULT_CAMERA_CONFIG,
        forward_reward_weight=DEFAULT_FORWARD_REWARD_WEIGHT,
        ctrl_cost_weight=DEFAULT_CTRL_COST_WEIGHT,
        contact_cost_weight=DEFAULT_CONTACT_COST_WEIGHT,
        contact_cost_range=DEFAULT_CONTACT_COST_RANGE,
        healthy_reward=DEFAULT_HEALTHY_REWARD,
        terminate_when_unhealthy=DEFAULT_TERMINATE_WHEN_UNHEALTHY,
        healthy_z_range=DEFAULT_HEALTHY_Z_RANGE,
        reset_noise_scale=DEFAULT_RESET_NOISE_SCALE,
        exclude_current_positions_from_observation=True,
        include_cinert_in_observation=True,
        include_cvel_in_observation=True,
        include_qfrc_actuator_in_observation=True,
        include_cfrc_ext_in_observation=True,
        **kwargs,
    ):
        utils.EzPickle.__init__(self, **locals())

        self.reset_noise_scale = reset_noise_scale

        self.reward_fn = HumanoidReward(
            forward_reward_weight,
            ctrl_cost_weight,
            contact_cost_weight,
            contact_cost_range,
            healthy_reward,
        )

        self.obs_fn = HumanoidObservation(
            exclude_current_positions_from_observation,
            include_cinert_in_observation,
            include_cvel_in_observation,
            include_qfrc_actuator_in_observation,
            include_cfrc_ext_in_observation,
        )

        self.termination_fn = HumanoidTermination(
            healthy_z_range, terminate_when_unhealthy
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
        xy_before = mass_center(self.model, self.data)
        self.do_simulation(action, self.frame_skip)
        xy_after = mass_center(self.model, self.data)

        xy_velocity = (xy_after - xy_before) / self.dt
        x_velocity, y_velocity = xy_velocity

        is_healthy = self.termination_fn.is_healthy(self.data)
        terminated = self.termination_fn(self)

        reward, reward_info = self.reward_fn.compute(
            x_velocity, is_healthy, self.data
        )

        info = {
            "x_position": self.data.qpos[0],
            "y_position": self.data.qpos[1],
            "x_velocity": x_velocity,
            "y_velocity": y_velocity,
            "distance_from_origin": np.linalg.norm(self.data.qpos[0:2]),
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
