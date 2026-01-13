import numpy as np
from gymnasium.envs.mujoco import MujocoEnv
from gymnasium import utils
from gymnasium.spaces import Box

from observation import AntObservation
from reward import AntReward
from termination import AntTermination


class AntEnv(MujocoEnv, utils.EzPickle):
    metadata = {"render_modes": ["human", "rgb_array"]}

    def __init__(self, xml_file="ant.xml", frame_skip=5, **kwargs):
        utils.EzPickle.__init__(self, xml_file, frame_skip, **kwargs)

        self.obs_fn = AntObservation()
        self.reward_fn = AntReward()
        self.termination_fn = AntTermination()

        MujocoEnv.__init__(
            self,
            xml_file,
            frame_skip,
            observation_space=None,
            **kwargs,
        )

        obs = self.obs_fn(self)
        self.observation_space = Box(
            low=-np.inf, high=np.inf, shape=obs.shape, dtype=np.float64
        )

    def step(self, action):
        xy_before = self.data.qpos[:2].copy()
        self.do_simulation(action, self.frame_skip)
        xy_after = self.data.qpos[:2].copy()

        x_velocity = (xy_after[0] - xy_before[0]) / self.dt

        obs = self.obs_fn(self)
        reward, reward_info = self.reward_fn(self, action, x_velocity)
        terminated = self.termination_fn(self)

        info = {
            "x_velocity": x_velocity,
            **reward_info,
        }

        return obs, reward, terminated, False, info

    def reset_model(self):
        noise = 0.1
        qpos = self.init_qpos + self.np_random.uniform(-noise, noise, size=self.model.nq)
        qvel = self.init_qvel + noise * self.np_random.standard_normal(self.model.nv)
        self.set_state(qpos, qvel)
        return self.obs_fn(self)
