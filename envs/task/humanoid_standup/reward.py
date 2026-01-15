import numpy as np


class HumanoidStandupReward:
    def __init__(
        self,
        uph_cost_weight: float,
        ctrl_cost_weight: float,
        impact_cost_weight: float,
        impact_cost_range: tuple[float, float],
    ):
        self.uph_cost_weight = uph_cost_weight
        self.ctrl_cost_weight = ctrl_cost_weight
        self.impact_cost_weight = impact_cost_weight
        self.impact_cost_range = impact_cost_range

    def compute(self, z_pos: float, action, model, data):
        dt = model.opt.timestep

        uph_cost = self.uph_cost_weight * (z_pos / dt)

        quad_ctrl_cost = self.ctrl_cost_weight * np.square(data.ctrl).sum()

        quad_impact_cost = self.impact_cost_weight * np.square(data.cfrc_ext).sum()
        quad_impact_cost = np.clip(
            quad_impact_cost,
            self.impact_cost_range[0],
            self.impact_cost_range[1],
        )

        reward = uph_cost - quad_ctrl_cost - quad_impact_cost + 1.0

        info = {
            "reward_linup": uph_cost,
            "reward_quadctrl": -quad_ctrl_cost,
            "reward_impact": -quad_impact_cost,
        }
        return reward, info
