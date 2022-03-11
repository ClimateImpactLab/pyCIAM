import numpy as np
import pandas as pd
import xarray as xr


def s2d(ds):
    for v in ds.data_vars:
        if ds[v].dtype == "float32":
            ds[v] = ds[v].astype("float64")
    return ds


def d2s(ds):
    for v in ds.data_vars:
        if ds[v].dtype == "float64":
            ds[v] = ds[v].astype("float32")
    return ds


def get_planning_period_map(years, at_start):
    return (
        (years < xr.DataArray(at_start[1:], dims=["at"], coords={"at": at_start[:-1]}))
        .reindex(at=at_start, fill_value=True)
        .idxmax("at")
    )


def pos(x):
    return np.maximum(x, 0)


def get_lslr_plan_data(
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
        lslr_plan_noadapt = pos(lslr_plan_noadapt)
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
        lslr_plan = lslr.groupby(planning_periods).max().rename("lslr_plan")

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
        RH_heights = pos(RH_heights)
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
