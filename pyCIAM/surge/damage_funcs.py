import numpy as np


def diaz_ddf_i(depth_st, depth_end):
    """Integral of depth / (1 + depth)."""
    return depth_end - depth_st + np.log(depth_st + 1) - np.log(depth_end + 1)


def diaz_dmf_i(depth_st, depth_end, vsl=None, floodmortality=None):
    """Integral of mortality damage function. Here is just constant"""
    return floodmortality * vsl * (depth_end - depth_st)
