"""This module contains functions related to creating a storm surge lookup table used
when running pyCIAM in "probabilistic" mode (i.e. running on many thousands of Monte
Carlo samples of sea level rise trajectories). In this mode, calculating storm surge
damages for each elevation slice, each year, each segment, each socioeconomic
trajectory, and each SLR sample is too computationally intensive. This is due to the
need to numerically integrate in each of these cases. Instead, we calculate fractional
losses for a number of storm surge heights between the minimum and maximum that will
occur across the entire SLR ensemble and use linear interpolation to create a linear
spline function that allows for rapid estimation of storm surge losses across a large
ensemble of SLR simulations. As currently configured and used, this lookup table relies
on the assumption of a fixed elevational distribution of capital and population that is
scaled homogeonously over time. However, if one wanted to incorporate within-country
migration into future socioeconomic trajectories, one could do so through the use of
year-specific lookup tables (a.k.a. spline functions).

Public Functions:
    create_surge_lookup
"""

import dask.array as da
import numpy as np
import pandas as pd
import xarray as xr

from pyCIAM.io import _load_lslr_for_ciam, save_to_zarr_region
from pyCIAM.surge._calc import (
    _calc_storm_damages_no_resilience,
    _get_surge_heights_probs,
)
from pyCIAM.surge.damage_funcs import diaz_ddf_i, diaz_dmf_i
from pyCIAM.utils import (
    _get_lslr_plan_data,
    _get_planning_period_map,
    _s2d,
    subset_econ_inputs,
)


def _get_exposure_vars(ds):
    return [
        d
        for d in ds.data_vars
        if (d.startswith("K_") or d.startswith("pop_")) and not d.endswith("_scale")
    ]


def _get_lslr_rhdiff_range(
    sliiders_store,
    slr_stores,
    seg_var,
    seg_vals,
    at_start,
    n_interp_pts_lslr,
    n_interp_pts_rhdiff,
    quantiles=None,
    interp_years=None,
    scen_mc_filter=None,
    include_cc=True,
    include_ncc=True,
    slr_0_years=2005,
    mc_dim="mc_sample_id",
    storage_options={},
):
    """Get the range of lslr and rhdiff that we need to model to cover the full range
    across scenario/mcs.

    The minimum LSLR value we'll need to model for the purposes of
    assessing storm damage is the minimum across sites of: the site-level maximum of "0
    minus the s10000 surge height" and "the minimum projected LSLR for all of the
    scenario/mcs we use in our binned LSL dataset". The maximum LSLR value is the
    maximum experienced at any site in any year for all of the sceanrio/mcs we use in
    the binned LSL dataset.
    """

    if isinstance(slr_0_years, int):
        slr_0_years = [slr_0_years] * len(slr_stores)
    assert len(slr_0_years) == len(slr_stores)
    pc_in = _s2d(
        xr.open_zarr(
            str(sliiders_store), chunks=None, storage_options=storage_options
        ).sel({seg_var: seg_vals})
    )

    if interp_years is None:
        interp_years = pc_in.year.values

    pc_in.surge_height.load()
    pc_in.gumbel_params.load()

    # get max surge height for this seg-adm
    smax = pc_in.surge_height.isel(return_period=-1)

    mins, maxs = [], []
    for sx, slr_store in enumerate(slr_stores):
        lslr = _load_lslr_for_ciam(
            slr_store,
            pc_in[["seg_lon", "seg_lat"]].to_dataframe(),
            interp_years=interp_years,
            scen_mc_filter=scen_mc_filter,
            include_cc=include_cc,
            include_ncc=include_ncc,
            mc_dim=mc_dim,
            slr_0_year=slr_0_years[sx],
            storage_options=storage_options,
            quantiles=quantiles,
        )

        # get the max LSLR experienced
        max_lslr = lslr.max([d for d in lslr.dims if d != seg_var]).sel(
            {seg_var: seg_vals}
        )

        # find the min LSLR we need to model. This is the max of "min lslr" and the
        # lowest elevation in this seg-adm minus the max surge height
        exp_vars = _get_exposure_vars(pc_in)
        assert len(exp_vars) == 2
        min_lslr = pc_in[exp_vars].to_array("tmp").sum("tmp") > 0
        if min_lslr.sum() == 0:
            return None
        min_lslr = pc_in.elev_bounds.sel(bound="lower", drop=True).where(min_lslr).min()
        min_lslr = np.maximum(
            lslr.min([d for d in lslr.dims if d != seg_var]), min_lslr - smax
        )

        # ensure that max > min to enable interpolation even when no damage is possible
        max_lslr = max_lslr.where(
            max_lslr > min_lslr, min_lslr + 0.01 * n_interp_pts_lslr
        )

        mins.append(min_lslr)
        maxs.append(max_lslr)

    min_lslr = xr.concat(mins, dim="tmp").min("tmp")
    max_lslr = xr.concat(maxs, dim="tmp").max("tmp")

    at = _get_planning_period_map(lslr.year, at_start)

    (
        _,
        RH_heights,
        _,
    ) = _get_lslr_plan_data(lslr, pc_in.surge_height, at)

    # allow for some floating point error in rh_diff_max
    rh_diff_max = (
        RH_heights.sel(at=at).isel(return_period=-2, drop=True).max("adapttype") - lslr
    ).max([d for d in lslr.dims if d != seg_var]) + 2 * np.finfo("float32").eps

    # occasionally, the gumbel fit was negative, so we set the 1-year return to 0
    assert (rh_diff_max > 0).all()

    return xr.Dataset(
        {
            "lslr_by_seg": (
                ("lslr", seg_var),
                np.linspace(min_lslr, max_lslr, n_interp_pts_lslr),
            ),
            "rh_diff_by_seg": (
                ("rh_diff", seg_var),
                np.linspace(0, rh_diff_max, n_interp_pts_rhdiff),
            ),
        },
        coords={
            "lslr": np.arange(n_interp_pts_lslr),
            "rh_diff": np.arange(n_interp_pts_lslr),
            seg_var: pc_in[seg_var].values,
        },
    )


def _create_surge_lookup_skeleton_store(
    sliiders_store,
    n_interp_pts_lslr,
    n_interp_pts_rhdiff,
    surge_lookup_store,
    seg_chunksize=1,
    seg_var="seg",
    seg_var_subset=None,
    force_overwrite=True,
    storage_options={},
):
    pc_in = subset_econ_inputs(
        xr.open_zarr(str(sliiders_store), storage_options=storage_options),
        seg_var,
        seg_var_subset,
    )

    to_save = xr.DataArray(
        da.empty(
            (len(pc_in[seg_var]), n_interp_pts_lslr, n_interp_pts_rhdiff, 2, 2),
            chunks=(seg_chunksize, -1, -1, -1, -1),
        ),
        dims=[seg_var, "lslr", "rh_diff", "costtype", "adapttype"],
        coords={
            seg_var: pc_in[seg_var].values,
            "lslr": np.arange(n_interp_pts_lslr),
            "rh_diff": np.arange(n_interp_pts_rhdiff),
            "adapttype": ["retreat", "protect"],
            "costtype": ["stormCapital", "stormPopulation"],
        },
    ).to_dataset(name="frac_losses")
    to_save["rh_diff_by_seg"] = (
        (seg_var, "rh_diff"),
        da.empty(
            (len(to_save[seg_var]), len(to_save.rh_diff)),
            chunks=(to_save.chunks[seg_var], to_save.chunks["rh_diff"]),
        ),
    )
    to_save["lslr_by_seg"] = (
        (seg_var, "lslr"),
        da.empty(
            (len(to_save[seg_var]), len(to_save.rh_diff)),
            chunks=(to_save.chunks[seg_var], to_save.chunks["rh_diff"]),
        ),
    )
    if force_overwrite:
        to_save.to_zarr(
            str(surge_lookup_store),
            compute=False,
            mode="w",
            storage_options=storage_options,
        )
    elif not surge_lookup_store.is_dir():
        to_save.to_zarr(
            str(surge_lookup_store), compute=False, storage_options=storage_options
        )
    return to_save


def _save_storm_dam(
    seg_vals,
    seg_var="seg",
    sliiders_store=None,
    slr_stores=None,
    surge_lookup_store=None,
    at_start=np.arange(2000, 2100, 10),
    n_interp_pts_lslr=100,
    n_interp_pts_rhdiff=100,
    ddf_i=diaz_ddf_i,
    dmf_i=diaz_dmf_i,
    quantiles=None,
    scen_mc_filter=None,
    mc_dim="mc_sample_id",
    start_year=None,
    slr_0_years=2005,
    storage_options={},
):
    """Function to map over each chunk to run through damage calcs."""
    diff_ranges = _get_lslr_rhdiff_range(
        sliiders_store,
        slr_stores,
        seg_var,
        seg_vals,
        at_start,
        n_interp_pts_lslr,
        n_interp_pts_rhdiff,
        quantiles=quantiles,
        scen_mc_filter=scen_mc_filter,
        mc_dim=mc_dim,
        slr_0_years=slr_0_years,
        storage_options=storage_options,
    )

    if diff_ranges is None:
        template = xr.open_zarr(
            str(surge_lookup_store), chunks=None, storage_options=storage_options
        ).sel({seg_var: seg_vals})
        template = xr.zeros_like(template)

        # these must be unique otherwise interp function will raise error
        template["lslr_by_seg"] = (
            (seg_var, "lslr"),
            np.tile(np.arange(len(template.lslr))[np.newaxis, :], (len(seg_vals), 1)),
        )
        template["rh_diff_by_seg"] = (
            (seg_var, "rh_diff"),
            np.tile(
                np.arange(len(template.rh_diff))[np.newaxis, :], (len(seg_vals), 1)
            ),
        )
        if surge_lookup_store is None:
            return template
        save_to_zarr_region(
            template, surge_lookup_store, storage_options=storage_options
        )
        return None

    selectors = {}
    if start_year is not None:
        selectors = {"year": slice(start_year, None)}
    pc_in = xr.open_zarr(
        str(sliiders_store), chunks=None, storage_options=storage_options
    ).sel(selectors)
    exp_vars = _get_exposure_vars(pc_in)
    pc_in = (
        pc_in[
            [
                "surge_height",
                "gumbel_params",
                "elev_bounds",
            ]
            + exp_vars
        ]
        .reset_coords(drop=True)
        .sel({seg_var: seg_vals}, drop=True)
        .load()
        .astype("float64")
    )
    pc_in = pc_in.rename({k: k.split("_")[0] for k in exp_vars})

    exp0 = pc_in[["K", "pop"]]
    exp0_tot = exp0.sum("elev")
    exp_frac = (exp0 / exp0_tot).where(exp0_tot != 0, 0)
    pc_in = pc_in.drop(
        [
            "K",
            "pop",
        ]
    )
    del exp0, exp0_tot

    this_max_surge_height = pc_in.surge_height.isel(return_period=-1, drop=True)

    surge_heights_to_model = _get_surge_heights_probs(
        0,
        this_max_surge_height,
        pc_in.gumbel_params,
    )

    res = []
    for R, H in [
        (diff_ranges.lslr_by_seg + diff_ranges.rh_diff_by_seg, 0),
        (0, diff_ranges.lslr_by_seg + diff_ranges.rh_diff_by_seg),
    ]:
        res.append(
            _calc_storm_damages_no_resilience(
                diff_ranges.lslr_by_seg + surge_heights_to_model.surge,
                exp_frac,
                pc_in.elev_bounds,
                R,
                H,
                ddf_i,
                dmf_i,
                surge_probs=surge_heights_to_model.p,
            ).to_array("costtype")
        )
    res = (
        xr.concat(res, dim=pd.Index(["retreat", "protect"], name="adapttype"))
        .reindex(costtype=["stormCapital", "stormPopulation"])
        .to_dataset(name="frac_losses")
    )

    res = xr.merge((res, diff_ranges))

    if surge_lookup_store is None:
        return res

    # identify which index to save to in template zarr
    save_to_zarr_region(res, surge_lookup_store, storage_options=storage_options)


def create_surge_lookup(
    sliiders_store,
    slr_stores,
    surge_lookup_store,
    seg_var,
    at_start,
    n_interp_pts_lslr,
    n_interp_pts_rhdiff,
    ddf_i,
    dmf_i,
    seg_var_subset=None,
    start_year=None,
    slr_0_years=2005,
    seg_chunksize=1,
    scen_mc_filter=None,
    quantiles=None,
    mc_dim="mc_sample_id",
    force_overwrite=False,
    client=None,
    client_kwargs={},
    storage_options={},
):
    """Create a storm surge lookup table which is used to define a linear spline
    function for each region modeled in pyCIAM. This output is not strictly necessary to
    run pyCIAM but substantially reduces computational expense when running pyCIAM on a
    large probabilistic ensemble of SLR trajectories.

    Parameters
    ----------
    sliiders_store : Path-like
        Path to zarr store that contains the SLIIDERS dataset (or one formatted
        similarly.
    slr_stores : list of Path-like
        List of paths of datasets containing SLR data, indexed by `mc_dim`. The outer
        bounds of these datasets will be used to determine the bounds of the surge
        lookup table
    surge_lookup_store : Path-like
        Path to the output zarr store that will contain the lookup table used to build
        the 2D linear spline function. May be a :py:class:`fsspec.FSMap` object.
    seg_var : str
        The name of the dimension in the SLIIDERS zarr store specifying each
        *region* to be modeled by pyCIAM. Each region must be nested within each coastal
        segment that serves as an independent decision-making agent, and regions may be
        equivalent to segments. The reason you may wish to have nested regions is to
        be able to aggregate impacts to a different regions than those that are defined
        by the segments.
    at_start : list of int
        A list specifying the starting years of each adpatation period. In pyCIAM, each
        segment chooses a new retreat or protection height at the start of each of these
        periods based on the maximum projected extreme sea level heights at various
        return intervals over the period.
    n_interp_pts_{lslr,rhdiff} : int
        The number of interpolation points to estimate within the output lookup table.
        The lookup table enables 2-dimensional linear interpolation where one dimension
        is defined by local sea level height (lslr) and the other by the difference
        between an assumed extreme sea level height and the height of either retreat or
        protection (rhdiff). This function calculates the maximum and minimum of these
        values across all years and SLR trajectories included in SLIIDERS-SLR and
        generates equally spaced interpolation points.
    {ddf,dmf}_i : function
        Functions defining integrals of the depth-damage and depth-mortality functions
        used to calculate losses from extreme sea levels.
    seg_var_subset : str, optional
        If not None (default), will only process segment/admin unit intersection names
        that contain this string. i.e. you can pass "_USA" to only process seg/admin
        intersections in the US.
    start_year : int, optional
        If not None, drop SLIIDERS years prior to this value.
    slr_0_year : int, default 2005
        The reference year for the SLR data, i.e. SLR is assumed to be 0 in this year.
    seg_chunksize : int, default 1
        How many regions to process at once. Larger numbers improve efficiency through
        better vectorization but increase memory footprint.
    scen_mc_filter : :py:class:`pandas.MultiIndex`, optional
        A list of paired `scenario` (str) and Monte Carlo (int) values that specify a
        subset of the individual SLR trajectories contained in the zarr store at
        `slr_store`. If None, run all scenario X MC sample combinations.
    quantiles : array_like, optional
        If not None, run only specified quantiles of SLR per year, per location, and per
        scenario, across the monte carlo dimension of the zarr store located at
        `slr_store`. If None, run all individual Monte Carlo samples.
    mc_dim : str, default "mc_sample_id"
        Name of the dimension that indexes individual SLR scenarios, commonly either
        Monte Carlo samples or pre-computed quantiles of SLR (if you don't want to
        compute quantiles on the fly with the `quantiles` kwarg).
    force_overwrite : bool, default False
        Whether to raise an error (True) or overwrite if `surge_lookup_store` already
        contains an existing zarr store.
    client : :py:class:`distributed.Client`, optional
        A dask.distributed Client object that will be used to parallelize the table
        generation process. If not specifiec, outputs will be generated using a single
        processor in series.
    client_kwargs : dict, optional
        Passed directly to `client.map` when mapping functions over a dask cluster.
    storage_options : dict, optional
        Passed to :py:function:`xarray.open_zarr`

    Returns
    -------
    Returns None, but saves storm surge lookup table to `surge_lookup_store`.
    """

    to_save = _create_surge_lookup_skeleton_store(
        sliiders_store,
        n_interp_pts_lslr,
        n_interp_pts_rhdiff,
        surge_lookup_store,
        seg_chunksize=seg_chunksize,
        seg_var=seg_var,
        seg_var_subset=seg_var_subset,
        force_overwrite=force_overwrite,
        storage_options=storage_options,
    )
    all_segs = to_save[seg_var].values
    seg_grps = [
        all_segs[i : i + seg_chunksize] for i in range(0, len(all_segs), seg_chunksize)
    ]

    if client is None:
        mapper = map
    else:
        mapper = client.map
    return list(
        mapper(
            _save_storm_dam,
            seg_grps,
            seg_var=seg_var,
            sliiders_store=sliiders_store,
            slr_stores=slr_stores,
            surge_lookup_store=surge_lookup_store,
            at_start=at_start,
            n_interp_pts_lslr=n_interp_pts_lslr,
            n_interp_pts_rhdiff=n_interp_pts_rhdiff,
            mc_dim=mc_dim,
            ddf_i=ddf_i,
            dmf_i=dmf_i,
            quantiles=quantiles,
            scen_mc_filter=scen_mc_filter,
            start_year=start_year,
            slr_0_years=slr_0_years,
            storage_options=storage_options,
            **client_kwargs,
        )
    )
