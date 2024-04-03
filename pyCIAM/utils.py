"""This private module contains miscellaneous functions to support pyCIAM."""

import shutil
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr
from cloudpathlib import CloudPath
from sklearn.neighbors import BallTree

from pyCIAM.constants import CASE_DICT


def _s2d(ds):
    for v in ds.data_vars:
        if ds[v].dtype == "float32":
            ds[v] = ds[v].astype("float64")
    return ds


def _get_planning_period_map(years, at_start):
    return (
        (years < xr.DataArray(at_start[1:], dims=["at"], coords={"at": at_start[:-1]}))
        .reindex(at=at_start, fill_value=True)
        .idxmax("at")
    )


def _pos(x):
    return np.maximum(x, 0)


def _get_lslr_plan_data(
    lslr,
    surge_heights,
    planning_periods,
    diaz_protect_height=False,
    diaz_lslr_plan=False,
    diaz_negative_retreat=False,
    min_R_noadapt=None,
):
    lslr_plan_noadapt = lslr.copy()
    if min_R_noadapt is not None:
        lslr_plan_noadapt = np.maximum(lslr_plan_noadapt, min_R_noadapt)

    # prohibit negative adaptation
    if diaz_negative_retreat:
        lslr_plan_noadapt = _pos(lslr_plan_noadapt)
    else:
        for i in range(1, len(lslr_plan_noadapt.year)):
            lslr_plan_noadapt[{"year": i}] = lslr_plan_noadapt.isel(
                year=slice(None, i + 1)
            ).max("year")

    # Planning slr height for retreat/protect scenarios. Diaz 2016 uses lslr at the
    # start of the next planning period as the design height for this planning period.
    # Updated model uses maximum of the LSLR within this planning period
    if diaz_lslr_plan:
        plan_years = np.unique(planning_periods)
        design_years = np.concatenate(
            (plan_years[1:], lslr.year.isel(year=[-1]).values)
        )
        lslr_plan = lslr.sel(year=design_years).rename(year="at")
        lslr_plan["at"] = plan_years
    else:
        # hack to handle newer xarray not being able to groupby with multiindex
        lslr_plan = lslr.unstack().groupby(planning_periods).max().rename("lslr_plan")
        if "scen_mc" in lslr.dims:
            lslr_plan = lslr_plan.stack(scen_mc=lslr.xindexes["scen_mc"].index.names)

    # hack to reduce surge height by 50% for protect 10 as in Diaz2016
    if diaz_protect_height:
        surge_heights_p = surge_heights.where(
            surge_heights.return_period != 10, surge_heights / 2
        )
    else:
        surge_heights_p = surge_heights

    surge_heights = xr.concat(
        (surge_heights, surge_heights_p),
        dim=pd.Index(["retreat", "protect"], name="adapttype"),
    )

    # calculate retreat and protect heights
    RH_heights = lslr_plan + surge_heights

    # add in the MSL-level advanced planning return height
    RH_heights = xr.concat(
        (lslr_plan.expand_dims(return_period=[1]), RH_heights), dim="return_period"
    )

    # Prohibit negative adaptation
    if diaz_negative_retreat:
        RH_heights = _pos(RH_heights)
    else:
        for i in range(1, len(RH_heights.at)):
            RH_heights[{"at": i}] = RH_heights.isel(at=slice(None, i + 1)).max("at")

    # set initial RH_heights to 0 (e.g.
    # assuming no retreat or protection anywhere such that both w/ and w/o climate
    # change scenarios are charged for this adaptation). Alternative would be to set
    # protections up to current surge height already exist (e.g. assuming perfect
    # adaptation already).
    RH_heights_prev = RH_heights.shift(at=1, fill_value=0)

    return (
        lslr_plan_noadapt,
        RH_heights,
        RH_heights_prev,
    )


def spherical_nearest_neighbor(df1, df2, x1="lon", y1="lat", x2="lon", y2="lat"):
    ball = BallTree(np.deg2rad(df2[[y2, x2]]), metric="haversine")
    _, ixs = ball.query(np.deg2rad(df1[[y1, x1]]))
    return pd.Series(df2.index[ixs[:, 0]], index=df1.index)


def add_attrs_to_result(ds):
    attr_dict = {
        "case": {
            "long_name": "Adaptation Strategy",
            "description": (
                "Adaptation strategy chosen by all segments.\n"
                "noAdaptation: Reactive retreat\n"
                "protectX: Proactive protection to the 1-in-X year return value\n"
                "retreat1: Proactive retreat to MSL\n"
                "retreatX: Proactive retreat to the 1-in-X year return value"
            ),
        },
        "costtype": {
            "long_name": "Cost Category",
            "description": (
                "stormCapital: Value of capital stock lost due to ESL\n"
                "stormMortality: Monetary value of lost life due to ESL, using Value "
                "of a Statistical Life (VSL)\n"
                "inundation: Value of inundated or abandoned dry land and capital "
                "stock\n"
                "relocation: Value of market and non-market costs of relocation\n"
                "wetland: Value of lost wetland services\n"
                "protection: Construction and maintenance costs of coastal protection"
            ),
        },
        "costs": {
            "long_name": "Annual costs",
            "description": (
                "Costs are valued in the units used in pyCIAM inputs. For Diaz 2016, "
                "this is $2010 USD. For Depsky 2023, this is $2019 USD."
            ),
        },
        "optimal_case": {
            "long_name": "Optimal adaptation strategy for each segment",
            "Description": f"Coded as follows: {CASE_DICT}",
        },
        "seg": {
            "long_name": "Coastal segment",
            "description": (
                "Coastal segment defining an individual decision-making agent when "
                "choosing optimal adaptation strategy"
            ),
        },
        "scenario": {
            "long_name": "SLR scenario",
        },
        "quantile": {
            "long_name": "SLR quantile",
            "description": (
                "Quantile of corresponding probabilistic local SLR projections used at "
                "each segment"
            ),
        },
        "iam": {
            "long_name": "GDP growth model",
            "description": (
                "Affects all values that are predicated on GDP (e.g. capital stock, "
                "land value, etc.)"
            ),
        },
        "ssp": {
            "long_name": "Shared Socioeconomic Pathway",
            "description": "Socioeconomic growth model used",
        },
    }
    extra_vars = [
        v
        for v in ds.variables
        if v not in ["year", "seg_adm", "npv"] + list(attr_dict.keys())
    ]
    assert not len(extra_vars), f"Unexpected variables: {extra_vars}"
    for v in ds.variables:
        if v in attr_dict:
            ds[v].attrs.update(attr_dict[v])
    return ds


def collapse_econ_inputs_to_seg(
    econ_input_path,
    output_path,
    seg_var_subset=None,
    output_chunksize=100,
    seg_var="seg_adm",
    storage_options={},
):
    sliiders = subset_econ_inputs(
        xr.open_zarr(
            str(econ_input_path), storage_options=storage_options, chunks=None
        ).load(),
        seg_var,
        seg_var_subset,
    )

    # clean and cast for double-precision math
    for v in sliiders.variables:
        sliiders[v].encoding = {}
        if sliiders[v].dtype == "float32":
            sliiders[v] = sliiders[v].astype("float64")

    ref_ypcc = sliiders.ypcc.sel(country="USA", drop=True).rename("ref_income")

    grouper = sliiders.seg
    usa_ypcc_ref = (
        sliiders.ypcc.sel(country="USA", drop=True).load().reset_coords(drop=True)
    )

    out = (
        sliiders[["K_2019", "pop_2019", "landarea", "length", "wetland"]]
        .groupby(grouper)
        .sum("seg_adm")
    )

    out[["surge_height", "gumbel_params", "seg_lon", "seg_lat"]] = (
        sliiders[["surge_height", "gumbel_params", "seg_lon", "seg_lat"]]
        .groupby(grouper)
        .first()
    )

    def weighted_avg(varname, wts_in):
        if seg_var not in sliiders[varname].dims:
            sliiders[varname] = sliiders[varname].sel(country=sliiders.seg_country)
        wts_out = wts_in.groupby(grouper).sum(seg_var)
        unweighted = sliiders[varname].groupby(grouper).sum(seg_var)
        out[varname] = (
            (sliiders[varname] * wts_in).groupby(grouper).sum(seg_var) / wts_out
        ).where(wts_out > 0, unweighted)

    for v, w in [
        (
            "mobcapfrac",
            sliiders.K_2019.sum("elev"),
        ),
        ("pop_scale", sliiders.pop_2019.sum("elev")),
        ("K_scale", sliiders.K_2019.sum("elev")),
        ("interior", sliiders.landarea.sum("elev")),
        ("pc", sliiders.length),
        ("ypcc", sliiders.pop_2019.sum("elev")),
        ("wetlandservice", sliiders.wetland.sum("elev")),
    ]:
        weighted_avg(v, w)

    out["rho"] = out.ypcc / (out.ypcc + usa_ypcc_ref.sel(year=2000, drop=True))

    # merge on non-aggregated vars
    other_vars = sliiders[
        [
            d
            for d in sliiders.data_vars
            if seg_var not in sliiders[d].dims and d not in out.data_vars
        ]
    ]
    out = xr.merge((out, other_vars))
    out["ref_income"] = ref_ypcc

    if output_path is None:
        return out
    out.chunk({"seg": output_chunksize}).to_zarr(
        str(output_path), storage_options=storage_options, mode="w"
    )


def subset_econ_inputs(ds, seg_var, seg_var_subset):
    if seg_var_subset is None:
        return ds

    if isinstance(seg_var_subset, str):
        if seg_var == "seg":
            subsetter = ds.seg.str.contains(seg_var_subset)
        else:
            subsetter = ds.seg.isin(
                np.unique(
                    ds.seg.sel({seg_var: ds[seg_var].str.contains(seg_var_subset)})
                )
            )

    return ds.sel({seg_var: subsetter})


def copy(path_src, path_trg):
    if isinstance(path_src, Path):
        if isinstance(path_trg, CloudPath):
            path_trg.upload_from(path_src)
        elif isinstance(path_trg, Path):
            if path_src.is_dir():
                copy_meth = shutil.copytree
                kwargs = {"dirs_exist_ok": True}
            else:
                copy_meth = shutil.copy
                kwargs = {}
            copy_meth(path_src, path_trg, **kwargs)
        else:
            raise TypeError(type(path_trg))
    elif isinstance(path_src, CloudPath):
        if isinstance(path_trg, Path):
            path_src.download_to(path_trg)
        elif isinstance(path_trg, CloudPath):
            path_src.copy(path_trg)
        else:
            raise TypeError(type(path_trg))
    else:
        raise TypeError(type(path_src))
