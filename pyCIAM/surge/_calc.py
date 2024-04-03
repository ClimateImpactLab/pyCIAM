"""This private module contains functions related to specific calculations within pyCIAM
that are called by the ``run`` module."""

import numpy as np
import xarray as xr
from scipy.stats import gumbel_r

from pyCIAM.surge.damage_funcs import diaz_ddf_i, diaz_dmf_i


def _get_surge_heights_probs(
    min_surge_ht, max_surge_ht, gumbel_params, n_surge_heights=100
):
    """Create an array of ``n_surge_heights`` surge heights and associated probabilities
    to apply in CIAM in order to sample an appropriate range of plausible surge
    heights."""

    # get gumbel params
    loc = gumbel_params.sel(params="loc", drop=True)
    scale = gumbel_params.sel(params="scale", drop=True)

    # get array of surge heights to model
    surge_hts = xr.apply_ufunc(
        np.linspace,
        min_surge_ht,
        max_surge_ht,
        n_surge_heights,
        output_core_dims=[["surge_height"]],
        kwargs={"axis": -1},
        dask="parallelized",
        dask_gufunc_kwargs={"output_sizes": {"surge_height": n_surge_heights}},
    )

    # get probability of falling into each surge height bin
    cdf = xr.apply_ufunc(
        gumbel_r.cdf,
        surge_hts,
        kwargs={
            "loc": loc.broadcast_like(surge_hts),
            "scale": scale.broadcast_like(surge_hts),
        },
        dask="parallelized",
    )
    pdf = cdf.diff(dim="surge_height", label="lower")

    # get the midpoint surge height of each bin
    surge_hts_mid = ((surge_hts + surge_hts.shift(surge_height=-1)) / 2).sel(
        surge_height=pdf.surge_height.values
    )

    # add in the final 1/10000th of the cdf and conservatively assign it the 1:10000 yr
    # height
    last_surge = surge_hts_mid.isel(surge_height=0, drop=True).copy()
    last_surge.data = np.ones_like(last_surge) * max_surge_ht
    surge_hts_mid = xr.concat((surge_hts_mid, last_surge), dim="surge_height")
    pdf = xr.concat((pdf, 1 - cdf.isel(surge_height=-1)), dim="surge_height")

    return xr.Dataset({"surge": surge_hts_mid, "p": pdf})


def _calc_storm_capital_and_mortality(
    esl,
    exp_dens,
    bin_bounds,
    R,
    template,
    ddf_i=diaz_ddf_i,
    dmf_i=diaz_dmf_i,
):
    min_ht = np.maximum(bin_bounds.sel(bound="lower", drop=True), R)
    max_ht = np.maximum(bin_bounds.sel(bound="upper", drop=True), R)
    depth_st = np.maximum(esl - max_ht, 0)
    depth_end = np.maximum(esl - min_ht, 0)
    del min_ht, max_ht

    storm_capital = (
        (ddf_i(depth_st, depth_end) * exp_dens.K).where(depth_end > 0).sum("elev")
    )
    storm_mortality = (
        (dmf_i(depth_st, depth_end) * exp_dens.pop).where(depth_end > 0).sum("elev")
    )

    return (
        xr.Dataset({"stormCapital": storm_capital, "stormPopulation": storm_mortality})
        .unstack()
        .reindex_like(template, fill_value=0)
    )


def _calc_storm_damages_no_resilience(
    esl,
    exposure,
    elev_bounds,
    R,
    H,
    ddf_i,
    dmf_i,
    stack=False,
    surge_probs=None,
):
    """Estimate storm-related damages before applying resilience factor."""

    elev_bin_widths = elev_bounds.sel(bound="upper", drop=True) - elev_bounds.sel(
        bound="lower", drop=True
    )

    # turn exposure into exposure density
    this_exposure_dens = exposure / elev_bin_widths

    valid = esl >= H
    esl = esl.where(valid)
    if stack:
        esl = esl.stack(stacked=esl.dims).dropna("stacked")
        if isinstance(R, xr.DataArray):
            R = R.where(valid)
            R = R.stack(stacked=esl.stacked.to_index().names).sel(stacked=esl.stacked)

    if surge_probs is None:
        out = _calc_storm_capital_and_mortality(
            esl,
            this_exposure_dens,
            elev_bounds,
            R,
            valid,
            ddf_i=ddf_i,
            dmf_i=dmf_i,
        )
    else:
        init_da = xr.zeros_like(valid)
        if "surge_height" in init_da.dims:
            init_da = init_da.sel
        out = None
        for h in surge_probs.surge_height.values:
            this_surge_probs = surge_probs.sel(surge_height=h, drop=True)
            if this_surge_probs.sum() == 0:
                continue

            this_esl = esl.sel(surge_height=h, drop=True)
            if isinstance(R, xr.DataArray) and "stacked" in R.dims:
                this_R = R.sel(surge_height=h, drop=True)
            else:
                this_R = R

            this_out = (
                _calc_storm_capital_and_mortality(
                    this_esl,
                    this_exposure_dens,
                    elev_bounds,
                    this_R,
                    valid.sel(surge_height=h, drop=True),
                    ddf_i=ddf_i,
                    dmf_i=dmf_i,
                )
                * this_surge_probs
            )
            if out is None:
                out = this_out
            else:
                out += this_out
    return out
