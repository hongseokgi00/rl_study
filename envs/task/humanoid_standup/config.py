import numpy as np

DEFAULT_CAMERA_CONFIG = {
    "trackbodyid": 1,
    "distance": 4.0,
    "lookat": np.array((0.0, 0.0, 0.8925)),
    "elevation": -20.0,
}

DEFAULT_FRAME_SKIP = 5

DEFAULT_UPH_COST_WEIGHT = 1.0
DEFAULT_CTRL_COST_WEIGHT = 0.1
DEFAULT_IMPACT_COST_WEIGHT = 0.5e-6
DEFAULT_IMPACT_COST_RANGE = (-np.inf, 10.0)

DEFAULT_RESET_NOISE_SCALE = 1e-2
