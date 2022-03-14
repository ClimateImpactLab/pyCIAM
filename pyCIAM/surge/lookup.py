import dask.array as da
import numpy as np
import pandas as pd
import xarray as xr

from .calc import calc_storm_damages_no_resilience, get_surge_heights_probs
from .damage_funcs import diaz_ddf_i, diaz_dmf_i
from .io import load_lslr_for_ciam, save_to_zarr_region
from .utils import get_lslr_plan_data, get_planning_period_map, s2d


def _get_exposure_vars(ds):
    return [
        d
        for d in ds.data_vars
        if (d.startswith("K_") or d.startswith("pop_")) and not d.endswith("_scale")
    ]


def _get_lslr_rhdiff_range(
    sliiders_econ_store,
    sliiders_slr_store,
    seg_var,
    seg_vals,
    at_start,
    n_interp_pts_lslr,
    n_interp_pts_rhdiff,
    quantile_minmax=None,
    interp_years=None,
    scen_mc_filter=None,
    include_cc=True,
    include_ncc=True,
    slr_slice=slice(None),
):
    """Get the range of lslr and rhdiff that we need to model to cover the full range
    across scenario/mcs. The minimum LSLR value we'll need to model for the purposes of
    assessing storm damage is the minimum across sites of: the site-level maximum of "0
    minus the s10000 surge height" and "the minimum projected LSLR for all of the
    scenario/mcs we use in our binned LSL dataset". The maximum LSLR value is the
    maximum experienced at any site in any year for all of the sceanrio/mcs we use in
    the binned LSL dataset."""

    pc_in = s2d(xr.open_zarr(sliiders_econ_store, chunks=None).sel({seg_var: seg_vals}))

    if interp_years is None:
        interp_years = pc_in.year.values

    # list out which sites are actually used for CIAM segments
    site_ids = pc_in.SLR_site_id
    unique_site_ids = np.unique(site_ids)

    pc_in.surge_height.load()
    pc_in.gumbel_params.load()

    # get max surge height for this seg-adm
    smax = pc_in.surge_height.isel(return_period=-1)

    lslr = load_lslr_for_ciam(
        sliiders_slr_store,
        unique_site_ids,
        interp_years=interp_years,
        scen_mc_filter=scen_mc_filter,
        slr_slice=slr_slice,
        include_cc=include_cc,
        include_ncc=include_ncc,
    ).sel(site_id=site_ids, drop=True)

    if quantile_minmax is not None:
        lslr = lslr.unstack().quantile(quantile_minmax, dim="mc_sample_id")

    # get the max LSLR experienced
    max_lslr = lslr.max([d for d in lslr.dims if d != seg_var]).sel({seg_var: seg_vals})

    # find the min LSLR we need to model. This is the max of "min lslr" and the lowest
    # elevation in this seg-adm minus the max surge height
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
    max_lslr = max_lslr.where(max_lslr > min_lslr, min_lslr + 0.01 * n_interp_pts_lslr)

    at = get_planning_period_map(lslr.year, at_start)

    (
        _,
        RH_heights,
        _,
    ) = get_lslr_plan_data(lslr, pc_in.surge_height, at)

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
    sliiders_econ_store,
    n_interp_pts_lslr,
    n_interp_pts_rhdiff,
    surge_lookup_store,
    seg_chunksize=1,
    seg_var="seg",
    force_overwrite=True,
):
    pc_in = xr.open_zarr(sliiders_econ_store)

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
        to_save.to_zarr(surge_lookup_store, compute=False, mode="w")
    elif not surge_lookup_store.fs.isdir(surge_lookup_store.root):
        to_save.to_zarr(surge_lookup_store, compute=False)
    return to_save


def save_storm_dam(
    seg_vals,
    seg_var="seg",
    sliiders_econ_store=None,
    sliiders_slr_store=None,
    surge_lookup_store=None,
    at_start=np.arange(2000, 2100, 10),
    n_interp_pts_lslr=100,
    n_interp_pts_rhdiff=100,
    ddf_i=diaz_ddf_i,
    dmf_i=diaz_dmf_i,
    floodmortality=0.01,
    quantile_minmax=None,
    scen_mc_filter=None,
):
    """Function to map over each chunk to run through damage calcs."""
    diff_ranges = _get_lslr_rhdiff_range(
        sliiders_econ_store,
        sliiders_slr_store,
        seg_var,
        seg_vals,
        at_start,
        n_interp_pts_lslr,
        n_interp_pts_rhdiff,
        quantile_minmax=quantile_minmax,
        scen_mc_filter=scen_mc_filter,
    )

    if diff_ranges is None:
        template = xr.open_zarr(surge_lookup_store, chunks=None).sel(
            {seg_var: seg_vals}
        )
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
        save_to_zarr_region(template, surge_lookup_store)
        return None

    pc_in = xr.open_zarr(sliiders_econ_store, chunks=None)
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

    surge_heights_to_model = get_surge_heights_probs(
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
            calc_storm_damages_no_resilience(
                diff_ranges.lslr_by_seg + surge_heights_to_model.surge,
                exp_frac,
                pc_in.elev_bounds,
                R,
                H,
                ddf_i,
                dmf_i,
                ddf_kwargs={},
                dmf_kwargs={
                    "floodmortality": floodmortality,
                    "vsl": 1,
                },
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
    save_to_zarr_region(res, surge_lookup_store)


def create_surge_lookup(
    sliiders_econ_store,
    sliiders_slr_store,
    surge_lookup_store,
    seg_var,
    at_start,
    n_interp_pts_lslr,
    n_interp_pts_rhdiff,
    ddf_i,
    dmf_i,
    floodmortality,
    client,
    client_kwargs={},
    template_save_kwargs={},
    seg_chunksize=1,
    scen_mc_filter=None,
    quantile_minmax=None,
):
    to_save = _create_surge_lookup_skeleton_store(
        sliiders_econ_store,
        n_interp_pts_lslr,
        n_interp_pts_rhdiff,
        surge_lookup_store,
        seg_chunksize=seg_chunksize,
        seg_var=seg_var,
        **template_save_kwargs,
    )
    all_segs = to_save[seg_var].values
    seg_grps = [
        all_segs[i : i + seg_chunksize] for i in range(0, len(all_segs), seg_chunksize)
    ]
    return client.map(
        save_storm_dam,
        seg_grps,
        seg_var=seg_var,
        sliiders_econ_store=sliiders_econ_store,
        sliiders_slr_store=sliiders_slr_store,
        surge_lookup_store=surge_lookup_store,
        at_start=at_start,
        n_interp_pts_lslr=n_interp_pts_lslr,
        n_interp_pts_rhdiff=n_interp_pts_rhdiff,
        ddf_i=ddf_i,
        dmf_i=dmf_i,
        quantile_minmax=quantile_minmax,
        floodmortality=floodmortality,
        scen_mc_filter=scen_mc_filter,
        **client_kwargs,
    )
