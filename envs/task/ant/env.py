from gymnasium import utils
from gymnasium.envs.mujoco import MujocoEnv
from gymnasium.spaces import Box
import numpy as np

from config import *
from termination import AntTermination
from reward import AntReward
from observation import AntObservation


class AntEnv(MujocoEnv, utils.EzPickle):
    metadata = {
        "render_modes": ["human", "rgb_array", "depth_array", "rgbd_tuple"],
    }

    def __init__(
        self,
        xml_file="ant.xml",
        frame_skip=DEFAULT_FRAME_SKIP,
        forward_reward_weight=DEFAULT_FORWARD_REWARD_WEIGHT,
        ctrl_cost_weight=DEFAULT_CTRL_COST_WEIGHT,
        contact_cost_weight=DEFAULT_CONTACT_COST_WEIGHT,
        healthy_reward=DEFAULT_HEALTHY_REWARD,
        healthy_z_range=DEFAULT_HEALTHY_Z_RANGE,
        contact_force_range=DEFAULT_CONTACT_FORCE_RANGE,
        reset_noise_scale=DEFAULT_RESET_NOISE_SCALE,
        terminate_when_unhealthy=True,
        exclude_current_positions_from_observation=True,
        include_cfrc_ext_in_observation=True,
        main_body=1,
        **kwargs,
    ):
        utils.EzPickle.__init__(self, **locals())

        self.main_body = main_body
        self.reset_noise_scale = reset_noise_scale

        self.termination_fn = AntTermination(
            healthy_z_range, terminate_when_unhealthy
        )
        self.reward_fn = AntReward(
            forward_reward_weight,
            ctrl_cost_weight,
            contact_cost_weight,
            healthy_reward,
            contact_force_range,
        )
        self.obs_fn = AntObservation(
            exclude_current_positions_from_observation,
            include_cfrc_ext_in_observation,
        )

        MujocoEnv.__init__(
            self,
            xml_file,
            frame_skip,
            observation_space=None,
            **kwargs,
        )

        self.observation_space = Box(
            low=-np.inf,
            high=np.inf,
            shape=(self.obs_fn.obs_size(self.data),),
            dtype=np.float64,
        )
