"""This module is used to define the fractional damage functions used in pyCIAM that
relate storm surge depth to mortality and physical capital loss. As of April 7, 2022,
the only functions available are those included in Diaz 2016. These define the
"resilience-unadjusted" fractional damages. That is, for a region with resiliendce
factor (:math:`\rho`) of 1, what fraction of the exposed population and/or physical
capital will be lost conditional on a storm surge depth.

At the moment, functions must have an analytical integral and are actually defined by
their integral, as can be seen with the suffix ``_i``.

To add a new damage function, simply define it by its integral where depth is in units
of meters. It must contain two mandatory arguments which refer to the bounds of the
definite integral. Kwargs relevant for this particular damage function are optional.
The output should be "fractional loss" assuming homogeonous distribution of
capital/population within a grid cell defined by the two bounds.

Example
-------
Let's say you wanted a linear damage function in depth where losses increased by 10% per
meter. The integral is this constant, so the damage function would defined simply as:

.. code-block:: python

    def linear_damage_func_i(depth_st, depth_end):
        return 0.1

This function would of course result in losses greater than 100% for depths over 10m and
is just shown here as an example
"""

import numpy as np


def diaz_ddf_i(depth_st, depth_end):
    """Integral of depth / (1 + depth), as used in Diaz 2016, assuming unit resilience
    (:math:`\rho`).

    Parameters
    ----------
    depth_{st,end} : float
        Define the lower and upper bounds of a definite integral

    Returns
    -------
    float :
        Fractional losses of physical capital, assuming homogenous distribution between
        `depth_st` and `depth_end`.
    """
    return depth_end - depth_st + np.log(depth_st + 1) - np.log(depth_end + 1)


def diaz_dmf_i(depth_st, depth_end, floodmortality=0.01):
    """Integral of mortality damage function as used in Diaz 2016, assuming unit
    resilience (:math:`\rho`). It is just a constant fraction conditional on a unit of
    exposure being inundated. Note that kwargs are not optional and will raise an error
    if not specified when called.

    Parameters
    ----------
    depth_{st,end} : float
        Define the lower and upper bounds of a definite integral
    floodmortality : float
        Fractional damage conditional on a unit of exposure being inundated.

    Returns
    -------
    float :
        Fractional mortality, assuming homogenous distribution between `depth_st` and `depth_end`
    """
    return floodmortality * (depth_end - depth_st)
