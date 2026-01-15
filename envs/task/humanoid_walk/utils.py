import numpy as np


def mass_center(model, data):
    num = np.einsum("b,bj->j", model.body_mass, data.xipos)
    denom = model.body_mass.sum()
    return (num / denom)[0:2].copy()
