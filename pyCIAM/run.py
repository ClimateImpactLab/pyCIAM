"""This module contains the central engine of pyCIAM, in which costs for all adaptation
options are calculated.

Functions
---------
* calc_costs
* select_optimal_case
* execute_pyciam
"""

from collections import OrderedDict
from shutil import rmtree

import numpy as np
import pandas as pd
import xarray as xr
from cloudpathlib import AnyPath, CloudPath
from distributed import Client, wait
from rhg_compute_tools.xarray import dataarray_from_delayed

from pyCIAM.constants import CASE_DICT, CASES, COSTTYPES, PLIST, RLIST, SOLVCASES
from pyCIAM.io import (
    check_finished_zarr_workflow,
    create_template_dataarray,
    load_ciam_inputs,
    load_diaz_inputs,
    save_to_zarr_region,
)
from pyCIAM.surge import damage_funcs, lookup
from pyCIAM.surge._calc import (
    _calc_storm_damages_no_resilience,
    _get_surge_heights_probs,
)
from pyCIAM.surge.damage_funcs import diaz_ddf_i, diaz_dmf_i
from pyCIAM.utils import (
    _get_lslr_plan_data,
    _get_planning_period_map,
    _pos,
    add_attrs_to_result,
    collapse_econ_inputs_to_seg,
    subset_econ_inputs,
)


def calc_costs(
    inputs,
    lslr,
    surge_lookup=None,
    elev_chunksize=1,
    ddf_i=diaz_ddf_i,
    dmf_i=diaz_dmf_i,
    diaz_protect_height=False,
    diaz_construction_freq=False,
    diaz_lslr_plan=False,
    diaz_negative_retreat=False,
    diaz_forward_diff=False,
    diaz_storm_calcs=False,
    diaz_fixed_vars_for_onetime_cost=False,
    diaz_calc_noadapt_damage_w_lslr=False,
    min_R_noadapt=None,
    return_year0_hts=False,
    return_RH_heights=False,
):
    """This is the central engine of pyCIAM. It calculates costs for all adaptation
    options (a.k.a. *cases*). It does not yet calculate the optimal strategy because
    users may have specified `regions` that are more finely resolved than coastal
    *segments*, which are the decision-making agents in pyCIAM. In this case, this
    function may be parallelized over *regions*, and the results must be aggregated to
    the *segment* level before finding the lowest-cost trajectory.

    Parameters
    ----------
    inputs : :py:class:`xarray.Dataset`
        A processed and formatted version of SLIIDERS (or similarly formatted
        dataset), ready to be ingested into pyCIAM. See :py:func:`.load_ciam_inputs`
    lslr : :py:class:`xarray.Dataset`
        A processed and formatted version of SLIIDERS-SLR (or similarly formatted
        dataset), ready to be ingested into pyCIAM. See :py:func:`.load_ciam_inputs`
    surge_lookup : :py:class:`xarray.DataArray`, optional
        If not None, a DataArray containing at least ``lslr`` and ``rhdiff`` dimensions.
        This will be used to build a 2D spline function to interpolate extreme sea
        level-related damages, rather than calculating explicitly. This drastically
        improves execution time.
    elev_chunksize : int, default 1
        Number of elevation slices to process simultaneously. Higher numbers improve
        efficiency through vectorization but result in a larger memory footprint.
    ddf_i, dmf_i : func, default :py:func:`.damage_funcs.ddf_i`,
        :py:func:`.damage_funcs.dmf_i`. Damage functions relating physical capital loss
        and monetized mortality arising from a certain depth of inundation.
    diaz_protect_height : bool, default False
        If True, reduce the 1-in-10-year extreme sea level by 50% as in Diaz 2016. This
        hack should not be necessary when using the ESL heights from CoDEC (as in
        SLIIDERS).
    diaz_construction_freq : bool, default False
        If True, set the lifetime over which the "linear" component of protection
        construction costs (i.e. the component that is proportional to length and not to
        height) is amortized to the length of the adaptation periods. This means that
        the total costs from this linear portion was directly proportional to the number
        of planning periods. Because the default planning periods for pyCIAM are
        shorter, we assume that the lifetime of the seawall foundation that depends on
        this linear part is 50 years (roughly the length of the planning periods in Diaz
        2016). This allows  total seawall construction costs to be roughly independent
        of the number of planning periods.
    diaz_lslr_plan : bool, default False
        If True, use the local sea level height at the start of the next planning period
        to define the design height for retreat or protection actions (as in Diaz 2016).
        If False, use the maximum slr height within the current adaptation planning
        period, which may differ if local SLR is not monotonically increasing through
        the adaptation period.
    diaz_negative_retreat : bool, default False
        If True, only prohibit retreat below present-day MSL but allow migration and/or
        protection to lower SLR levels as long as you don't migrate below. This creates
        inconsistencies in the calculation of retreat, protection, and inundation costs
        for regions that experience negative local SLR. If False, prohibit downward
        migration and/or lowering protection heights to preserve consistency of cost
        calculations.
    diaz_forward_diff : bool, default False
        If True, use a mixture of forward and backward difference techniques to
        calculate rates of change for different cost calculations, as in Diaz 2016. If
        False, use backward difference in all contexts.
    diaz_storm_calcs : bool, default False
        If True, use the pre-calculated exponential functions, parameterized by segment,
        from Diaz 2016 for ESL-related costs. Note that these are as implemented in the
        `associated code <https://github.com/ClimateImpactLab/pyCIAM>`_, which is
        slightly different than described in the paper and assumes constant marginal
        physical capital losses regardless of starting sea level height. If False, use
        either a pre-calculated 2D spline function (if `surge_lookup` is not None) or
        directly calculate ESL costs for each segment, year, elevation, socioeconomic
        scenario, and SLR trajectory. Note that this may only be True if the parameters
        of the exponential function are specified in `inputs`.
    diaz_fixed_vars_for_onetime_cost : bool, default False
        If True, use the price of construction at the start of an adaptation period to
        calculate amortized one-time costs (as in Diaz 2016). If False, account for
        growth in GDP and correlated variables (e.g. land value) over the adaptation
        period.
    diaz_calc_noadapt_damage_w_lslr : bool, default False
        In Diaz 2016, a "spinup" model is used first to calculate starting retreat
        heights assumed for the "reactive retreat" case (a.k.a. "noAdaptation").
        However, when calculating inundation and relocation costs for that reactive
        retreat case, this retreat height is ignored and segments are assumed to only
        retreat to the level of local sea level in a given year (or to the starting sea
        level if future LSLR values become negative). This results in inconsistent
        retreat heights used for inundation and relocation costs than for wetland loss
        and ESL-related losses. This is the approach used if True. If False, stay
        consistent with the retreat height for the reactive retreat scenario case by
        assuming the spinup retreat height for all cost calculations.
    min_R_noadapt : :py:class:`xarray.DataArray`, optional
        If specified, this should be a 1D DataArray indexed by region, which is a
        dimension name that should match that of `inputs`, `lslr`, and/or
        `surge_lookup`. This is the result of having previously run pyCIAM in "spinup"
        mode to determine the initial retreat height for the reactive retreat scenario.
    return_year0_heights : bool, default False
        If True, in addition to costs, output the corresponding retreat or protect
        elevation for each adaptation case in the initial year. This us used when run in
        "spinup" mode. Once the optimal case is chosen for each segment, the
        corresponding height is used as `min_R_noadapt` in a subsequent run of pyCIAM.
    return_RH_heights : bool, default False
        Not typically used, this flag means that results will also include the retreat
        and protection heights for all years. This is useful for the revealed preference
        identification of the non-market costs of relocation performed in Depsky et al.
        2022.

    Returns
    -------
    :py:class:`xarray.Dataset`
        Dataset of costs for each region included in `inputs`, for all scenarios and
        Monte Carlo SLR samples in `lslr`, for all cost types, for all adaptation cases
        *other* than the optimal case, which must be calculated separately. This is to
        allow for arbitrary aggregation of costs from multiple regions within a
        decision-making segment  before calculating the least-cost adaptation pathway.
    :py:class:`xarray.DataArray`
        If `return_year0_heights` is True, also return the first adaptation heights for
        each adaptation case, SLR scenario, and Monte Carlo sample. Used when running
        pyCIAM in "spinup" mode.
    :py:class:`xarray.Dataset`
        If `return_RH_heights` is True, also return the retreat and protection heights
        from all years.

    Notes
    -----
    * Year-specific `surge_lookup` arrays have not yet been implemented, but would be
      necessary if projecting within-segment heterogeneous population, GDPpc, or
      physical capital growth (i.e. differing over elevation slices). This does not
      exist in SLIIDERS, for which growth is homogeonous within each country.
    """
    # You should have already made sure your inputs and slr are in the same years
    assert (lslr.year == inputs.year).all()

    # if elev_chunksize is None, don't chunk over elev dimension at all
    if elev_chunksize is None:
        elev_chunksize = len(inputs.elev)

    # calculate lower bound of elevation bins
    if "elev_bounds" in inputs.data_vars:
        lb_elev = inputs.elev_bounds.sel(bound="lower", drop=True).copy()
        bin_width = inputs.elev_bounds.sel(bound="upper", drop=True) - lb_elev
    else:
        bin_width = inputs.elev.diff("elev").reindex(elev=inputs.elev).bfill("elev")
        lb_elev = (inputs.elev - bin_width / 2).copy()

    # correct potential numerical rounding issue
    lb_elev[0] = 0

    # get planning period info
    at = _get_planning_period_map(lslr.year, inputs.at_start.values)
    tstept = (
        inputs.year.diff("year", label="lower").reindex(year=inputs.year).ffill("year")
    )
    tstep_at = tstept.groupby(at).sum()

    # Calculate annual local SLR rate btw t periods. This is one calculation that diaz
    # does with a forward difference
    if diaz_forward_diff:
        localrate = _pos(
            (lslr.diff("year", label="lower").reindex(year=lslr.year) / tstept).ffill(
                "year"
            )
        )
    else:
        localrate = _pos(
            (lslr.diff("year").reindex(year=lslr.year) / tstept).bfill("year"),
        )

    # get various planning height values
    (lslr_plan_noadapt, RH_heights, RH_heights_prev) = _get_lslr_plan_data(
        lslr,
        inputs.surge_height,
        at,
        diaz_protect_height=diaz_protect_height,
        diaz_lslr_plan=diaz_lslr_plan,
        diaz_negative_retreat=diaz_negative_retreat,
        min_R_noadapt=min_R_noadapt,
    )

    # --------- ELEVATION DISTRIBUTION-INDEPENENT COSTS ----------
    # These are independent of the elevation distribution data so are calculated
    # outside of the below loop

    # SURGE
    # Get different between retreat/protect height and lslr
    rh_diff = (
        RH_heights.sel(at=at).drop("at").isel(return_period=slice(None, -1)) - lslr
    )
    rh_diff_noadapt = lslr_plan_noadapt - lslr

    # rh_diff could be negative if diaz_lslr_plan is true, and in these cases, it is
    # always clipped at 0 for surge damage calculations
    rh_diff = _pos(rh_diff)
    rh_diff_noadapt = _pos(rh_diff_noadapt)

    # calculate fractional storm damage using the coefficients provided by Diaz 2016
    if diaz_storm_calcs:
        tot_landarea = inputs.landarea.sum("elev")
        coefs = inputs.surge_coefs.to_dataset("coef")
        sigma_noadapt = coefs.rsig0 / (
            1 + coefs.rsigA * np.exp(coefs.rsigB * rh_diff_noadapt)
        )
        sigma_r = coefs.rsig0 / (
            1 + coefs.rsigA * np.exp(coefs.rsigB * rh_diff.sel(adapttype="retreat"))
        )
        sigma_p = (coefs.psig0 + coefs.psig0coef * _pos(lslr)) / (
            1 + coefs.psigA * np.exp(coefs.psigB * rh_diff.sel(adapttype="protect"))
        )
        sigma = xr.concat((sigma_r, sigma_p), dim="adapttype")
        surge_cap = sigma / tot_landarea
        surge_pop = surge_cap * 0.01  # floodmortality from diaz
        surge = xr.concat(
            (surge_cap, surge_pop),
            dim=pd.Index(["stormCapital", "stormPopulation"], name="costtype"),
        ).to_dataset("costtype")
        surge_cap_noadapt = sigma_noadapt / tot_landarea
        surge_pop_noadapt = surge_cap_noadapt * 0.01  # floodmortality from diaz
        surge_noadapt = xr.concat(
            (surge_cap_noadapt, surge_pop_noadapt),
            dim=pd.Index(["stormCapital", "stormPopulation"], name="costtype"),
        ).to_dataset("costtype")

        # multiply fractional losses by seg-ir-level total capital and population
        surge["stormCapital"] = surge.stormCapital * inputs.K.sum("elev")
        surge["stormPopulation"] = surge.stormPopulation * inputs.pop.sum("elev")
        surge_noadapt["stormCapital"] = surge_noadapt.stormCapital * inputs.K.sum(
            "elev"
        )
        surge_noadapt["stormPopulation"] = (
            surge_noadapt.stormPopulation * inputs.pop.sum("elev")
        )
    else:
        if surge_lookup is None:
            rh_years = RH_heights.sel(at=at).drop("at")
            min_hts = 0
            max_hts = inputs.surge_height.isel(return_period=-1, drop=True)
            surge_heights_probs = _get_surge_heights_probs(
                min_hts, max_hts, inputs.gumbel_params
            )
            esl = lslr + surge_heights_probs.surge

            surge_noadapt = _calc_storm_damages_no_resilience(
                esl,
                inputs[["K", "pop"]],
                inputs.elev_bounds,
                lslr_plan_noadapt,
                0,
                ddf_i,
                dmf_i,
                surge_probs=surge_heights_probs.p,
            )

            surge = []
            for R, H in (
                (rh_years.sel(adapttype="retreat", drop=True), 0),
                (0, rh_years.sel(adapttype="protect", drop=True)),
            ):
                surge.append(
                    _calc_storm_damages_no_resilience(
                        esl,
                        inputs[["K", "pop"]],
                        inputs.elev_bounds,
                        R,
                        H,
                        ddf_i,
                        dmf_i,
                        surge_probs=surge_heights_probs.p,
                    )
                )
            surge = xr.concat(
                surge, dim=pd.Index(["retreat", "protect"], name="adapttype")
            )
        else:
            # Interpolate to the SLR and rh_diff values for all scenario/mc/year/iam/ssp
            # lslrs that are too low will result in 0 damages, b/c surge isn't high
            # enough to make it to 0 MSL eleveation, so we fill missings from
            # interpolation with 0's
            surge_noadapt = []
            surge = []
            for seg in inputs.seg.values:
                this_surge_lookup = (
                    surge_lookup.sel(seg=seg)
                    .swap_dims(lslr="lslr_by_seg", rh_diff="rh_diff_by_seg")
                    .reset_coords(drop=True)
                    .frac_losses.rename(lslr_by_seg="lslr", rh_diff_by_seg="rh_diff")
                )
                if this_surge_lookup.sum() == 0:
                    continue
                surge_noadapt.append(
                    this_surge_lookup.sel(adapttype="retreat", drop=True)
                    .interp(
                        lslr=lslr.sel(seg=seg),
                        rh_diff=rh_diff_noadapt.sel(seg=seg),
                        assume_sorted=True,
                        kwargs={"fill_value": 0},
                    )
                    .reset_coords(drop=True)
                    .expand_dims(seg=[seg])
                )

                surge_adapt = []
                for adapttype in this_surge_lookup.adapttype.values:
                    surge_adapt.append(
                        this_surge_lookup.sel(adapttype=adapttype)
                        .interp(
                            lslr=lslr.sel(seg=seg),
                            rh_diff=rh_diff.sel(
                                adapttype=adapttype, seg=seg, drop=True
                            ),
                            assume_sorted=True,
                            kwargs={"fill_value": 0},
                        )
                        .reset_coords(drop=True)
                    )
                surge.append(
                    xr.concat(surge_adapt, dim=this_surge_lookup.adapttype).expand_dims(
                        seg=[seg]
                    )
                )
            if len(surge) == 0:
                surge_noadapt = (
                    rh_diff_noadapt.expand_dims(
                        costtype=this_surge_lookup.costtype
                    ).to_dataset("costtype")
                    * 0
                )
                surge = (
                    rh_diff.expand_dims(costtype=this_surge_lookup.costtype).to_dataset(
                        "costtype"
                    )
                    * 0
                )
            else:
                surge = (
                    xr.concat(surge, dim="seg")
                    .reindex(seg=inputs.seg.values, fill_value=0)
                    .to_dataset("costtype")
                )
                surge_noadapt = (
                    xr.concat(surge_noadapt, dim="seg")
                    .reindex(seg=inputs.seg.values, fill_value=0)
                    .to_dataset("costtype")
                )

            # multiply fractional losses by seg-ir-level total capital and population
            surge["stormCapital"] = surge.stormCapital * inputs.K.sum("elev")
            surge["stormPopulation"] = surge.stormPopulation * inputs.pop.sum("elev")
            surge_noadapt["stormCapital"] = surge_noadapt.stormCapital * inputs.K.sum(
                "elev"
            )
            surge_noadapt["stormPopulation"] = (
                surge_noadapt.stormPopulation * inputs.pop.sum("elev")
            )

    surge = surge.stack(tmp=["adapttype", "return_period"])
    surge = (
        surge.assign(case=surge.adapttype.str.cat(surge.return_period.astype(str)))
        .swap_dims(tmp="case")
        .drop(["tmp", "adapttype", "return_period"])
    )

    # merge no adapt and retreat/protect, dropping extra protect1 which we do not allow.
    # Also, skip the last return value where we assume full protection
    surge = xr.concat(
        (surge_noadapt.expand_dims(case=["noAdaptation"]), surge), dim="case"
    ).sel(case=["noAdaptation"] + RLIST[:-1] + PLIST[:-1])
    del surge_noadapt

    # monetize deaths
    surge["stormPopulation"] = surge.stormPopulation * inputs.vsl

    # multiply fractional losses by resilience factor
    surge = surge * (1 - inputs.rho)

    # PROTECTION
    # Construction and maintenance costs of seawall. In Diaz 2016, a linear charge was
    # levied at the start of every planning period. This means that the total costs from
    # this linear portion was directly proportional to the number of planning periods.
    # Because we have shortened the planning period, we assume that the lifetime of the
    # seawall foundation that depends on this linear part is 50 years (roughly the
    # length of the planning periods in Diaz 2016). This allows our total seawall
    # construction costs to be roughly independent of the number of planning periods.
    construction_freq = 50
    if diaz_construction_freq:
        construction_freq = tstep_at

    protect_heights = np.array([p.lstrip("protect") for p in PLIST]).astype(int)
    RH_heights_p = RH_heights.sel(
        adapttype="protect", return_period=protect_heights
    ).drop("adapttype")
    RH_heights_p_prev = RH_heights_prev.sel(
        adapttype="protect", return_period=protect_heights
    ).drop("adapttype")

    protection = (
        (
            (
                # construction (one time cost)
                (
                    inputs.pcfixedfrac / construction_freq
                    + (
                        (1 - inputs.pcfixedfrac)
                        * (RH_heights_p**2 - RH_heights_p_prev**2)
                    )
                    / tstep_at
                )
                # maintenance (annual cost)
                + inputs.wall_maintcost * RH_heights_p
            )
            * inputs.length
            * inputs.pc
        )
        .sel(at=at)
        .drop("at")
    )

    # Half of Land rent value lost due to occupation by seawall (rent is
    # assumed to be half of rent of lowest elevation bin, specified in input data
    # creation)
    landrent0 = inputs.landrent
    if "elev" in landrent0.dims:
        landrent0 = landrent0.isel(elev=0, drop=True)

    if diaz_fixed_vars_for_onetime_cost:
        protection = protection + (
            inputs.wall_width2height
            * RH_heights_p
            * landrent0.rename(year="at").sel(at=RH_heights_p.at)
            / 2
            * inputs.length
        ).sel(at=at).drop("at")
    else:
        protection = protection + (
            inputs.wall_width2height
            * RH_heights_p.sel(at=at).drop("at")
            * landrent0
            / 2
            * inputs.length
        )

    # WETLANDS
    # if protection, all wetlands are lost. For Diaz 2016 inputs, we have more wetlands
    # than land area, so not all wetlands are accounted for in wetlands-by-elev
    if "total_wetland_val" in inputs.data_vars:
        wetland_p = inputs.total_wetland_val
    else:
        wetland_p = (inputs.wetland * inputs.wetlandservice).sum("elev")

    # --------- ELEVATION DISTRIBUTION-DEPENENT COSTS ----------
    def calc_elev_bin_weights(slr, lb_elevs, bin_width):
        """Calculates the fraction of a cell inundated/abandoned given a defined
        slr/retreat height."""
        return _pos(np.minimum(slr - lb_elevs, bin_width)) / bin_width

    # loop over each elevation band to sum up elevation-distribution-dependent costs
    inundation = 0
    relocation_noadapt = 0
    abandonment = 0
    relocation_r = 0
    wetland_r_noadapt = 0

    for e_ix in range(0, len(lb_elev), elev_chunksize):
        this_lb_elev = lb_elev.isel(elev=slice(e_ix, e_ix + elev_chunksize))
        this_elev = this_lb_elev.elev.values

        this_bin_width = bin_width.sel(elev=this_elev)
        this_inputs = inputs.sel(elev=this_elev)
        this_inputs_at_only = this_inputs.rename(year="at").sel(at=RH_heights.at)

        # --------- NO ADAPTATION ----------

        # this is one place where Diaz 2016 employs a forward difference to calculate
        # costs (whereas for retreat/protect scenarios it uses a backwards diff)
        lslr_pos = lslr_plan_noadapt
        if diaz_calc_noadapt_damage_w_lslr:
            lslr_pos = _pos(lslr)
        bin_wt_present = calc_elev_bin_weights(lslr_pos, this_lb_elev, this_bin_width)
        if diaz_forward_diff:
            bin_wt = calc_elev_bin_weights(
                lslr_pos.shift(year=-1).ffill("year"),
                this_lb_elev,
                this_bin_width,
            )
            bin_wt_diff = bin_wt - bin_wt_present
        # else assume no slr before starting year (to help calculate losses at t0)
        else:
            bin_wt = bin_wt_present
            bin_wt_diff = bin_wt - calc_elev_bin_weights(
                lslr_pos.shift(year=1, fill_value=0),
                this_lb_elev,
                this_bin_width,
            )
        inundation += (
            # lost land rent
            (bin_wt * this_inputs.landrent * this_inputs.landarea)
            # one-time loss of non-mobile capital
            + bin_wt_diff * (1 - this_inputs.mobcapfrac) * this_inputs.K / tstept
        ).sum("elev")

        # relocation
        relocation_noadapt += (
            bin_wt_diff
            * (
                # one-time relocation of people cost
                this_inputs.reactive_retreat_factor
                * this_inputs.movefactor
                * this_inputs.ypc
                * this_inputs.pop
                # one-time relocation and demolition of capital
                + (
                    this_inputs.capmovefactor * this_inputs.mobcapfrac
                    + this_inputs.democost * (1 - this_inputs.mobcapfrac)
                )
                * this_inputs.K
            )
        ).sum("elev") / tstept

        # --------- RETREAT ----------
        bin_wt = calc_elev_bin_weights(
            RH_heights.sel(adapttype="retreat", drop=True), this_lb_elev, this_bin_width
        )
        bin_wt_diff = _pos(
            bin_wt
            - calc_elev_bin_weights(
                RH_heights_prev.sel(adapttype="retreat", drop=True),
                this_lb_elev,
                this_bin_width,
            )
        )

        # abandonment
        if diaz_fixed_vars_for_onetime_cost:
            abandonment += (
                (
                    # Value of abandoned land following retreat
                    bin_wt * this_inputs_at_only.landrent * this_inputs_at_only.landarea
                    # one-time loss of abandoned capital
                    + bin_wt_diff
                    * (1 - this_inputs_at_only.mobcapfrac)
                    * (1 - this_inputs_at_only.depr)
                    * this_inputs_at_only.K
                    / tstep_at
                )
                .sum("elev")
                .sel(at=at)
                .drop("at")
            )
        else:
            abandonment += (
                # Value of abandoned land following retreat
                (
                    bin_wt.sel(at=at).drop("at")
                    * this_inputs.landrent
                    * this_inputs.landarea
                ).sum("elev")
                # one-time loss of abandoned capital
                + (
                    bin_wt_diff.sel(at=at).drop("at")
                    * (1 - this_inputs.mobcapfrac)
                    * (1 - this_inputs.depr)
                    * this_inputs.K
                    / tstep_at.sel(at=at).drop("at")
                ).sum("elev")
            )

        # relocation
        if diaz_fixed_vars_for_onetime_cost:
            relocation_r += (
                (
                    (
                        bin_wt_diff
                        * (
                            # one-time relocation of people
                            this_inputs_at_only.pop
                            * this_inputs_at_only.ypc
                            * this_inputs_at_only.movefactor
                            # one-time relocation and demolition of capital
                            + this_inputs_at_only.K
                            * (
                                this_inputs_at_only.mobcapfrac
                                * this_inputs_at_only.capmovefactor
                                + (1 - this_inputs_at_only.mobcapfrac)
                                * this_inputs_at_only.democost
                            )
                        )
                    ).sum("elev")
                    / tstep_at
                )
                .sel(at=at)
                .drop("at")
            )
        else:
            relocation_r += (
                bin_wt_diff.sel(at=at).drop("at")
                * (
                    # one-time relocation of people
                    this_inputs.pop * this_inputs.ypc * this_inputs.movefactor
                    # one-time relocation and demolition of capital
                    + this_inputs.K
                    * (
                        this_inputs.mobcapfrac * this_inputs.capmovefactor
                        + (1 - this_inputs.mobcapfrac) * this_inputs.democost
                    )
                )
            ).sum("elev") / tstep_at.sel(at=at).drop("at")

        # Wetland
        # If no protection, then some fraction of inundated wetlands can accrete if
        # local rate of slr less than max. We define wetlands as covering all land area
        # starting from the bottom of a grid cell. So if there are 5 sqkm of wetlands
        # and 10 sqkm of land area in a grid cell that is half-flooded, all 5 sqkm of
        # wetlands will be lost. If there are more wetlands than land area, the loss
        # is just proportional to inundation percentage. e.g. if that grid cell had 15
        # sqkm of wetlands instead of 5, it would lose 7.5 sqkm when half flooded
        bin_wt = calc_elev_bin_weights(lslr, this_lb_elev, this_bin_width)
        wetland_r_noadapt += (
            this_inputs.wetlandservice
            * np.maximum(
                np.minimum(this_inputs.landarea * bin_wt, this_inputs.wetland),
                this_inputs.wetland * bin_wt,
            )
        ).sum("elev") * np.minimum(1, (localrate / this_inputs.wmaxrate) ** 2)

    # In Diaz 2016, it's the actual costs that are filled for the final year, rather
    # than the LSL heights, for both inundation and relocation
    if diaz_forward_diff:
        yr_mask = inundation.year != inundation.year[-1].item()
        inundation = inundation.where(yr_mask, inundation.isel(year=-2, drop=True))
        relocation_noadapt = relocation_noadapt.where(
            yr_mask, relocation_noadapt.isel(year=-2, drop=True)
        )

    # --------- Aggregate costs ------------

    def caseify(ds, prefix):
        ds = ds.rename(return_period="case")
        ds["case"] = np.char.add(prefix, ds.case.astype(str))
        return ds

    abandonment = caseify(abandonment, "retreat")
    relocation_r = caseify(relocation_r, "retreat")
    protection = caseify(protection, "protect")

    # No Adaptation
    noadapt_cost = (
        xr.merge(
            (
                surge.sel(case="noAdaptation", drop=True),
                xr.Dataset(
                    {
                        "inundation": inundation,
                        "relocation": relocation_noadapt,
                        "wetland": wetland_r_noadapt,
                        "protection": xr.zeros_like(inundation),
                    }
                ),
            )
        )
        .to_array("costtype")
        .expand_dims(case=["noAdaptation"])
    )
    del inundation, relocation_noadapt

    # Retreat
    retreat_cost = xr.merge(
        (
            # this fills retreat10000 with a 0 surge damage
            surge.reindex(case=RLIST, fill_value=0),
            xr.Dataset(
                {
                    "inundation": abandonment,
                    "relocation": relocation_r,
                    "wetland": wetland_r_noadapt,
                    "protection": xr.zeros_like(abandonment),
                }
            ),
        )
    ).to_array("costtype")
    del abandonment, relocation_r, wetland_r_noadapt

    # Protect
    protect_cost = xr.merge(
        (
            # this fills protect10000 with a 0 surge damage
            surge.reindex(case=PLIST, fill_value=0),
            xr.Dataset(
                {
                    "inundation": xr.zeros_like(protection),
                    "relocation": xr.zeros_like(protection),
                    "wetland": wetland_p,
                    "protection": protection,
                }
            ),
        )
    ).to_array("costtype")
    del protection, wetland_p

    # Aggregate costs for all cases and mutliply by length of each timestep
    out = xr.concat(
        (noadapt_cost, retreat_cost, protect_cost),
        dim="case",
    )
    del noadapt_cost, retreat_cost, protect_cost

    if return_year0_hts:
        RH_heights0 = RH_heights.isel(at=0, drop=True)
        out = [
            out,
            xr.concat(
                (
                    lslr_plan_noadapt.isel(year=0, drop=True).expand_dims(
                        case=["noAdaptation"]
                    ),
                    caseify(RH_heights0.sel(adapttype="retreat", drop=True), "retreat"),
                    caseify(RH_heights0.sel(adapttype="protect", drop=True), "protect"),
                ),
                dim="case",
            ).sel(case=out.case),
        ]
    if return_RH_heights:
        RH_heights = RH_heights.stack(case=["adapttype", "return_period"])
        RH_heights = RH_heights.drop("case").assign_coords(
            case=RH_heights.adapttype.str.cat(
                RH_heights.return_period.astype(str)
            ).values
        )
        RH_heights = xr.concat(
            (
                lslr_plan_noadapt.expand_dims(case=["noAdaptation"]),
                RH_heights.sel(at=at).drop("at"),
            ),
            dim="case",
        )
        if isinstance(out, list):
            out.append(RH_heights)
        else:
            out = [out, RH_heights]

    return out


def select_optimal_case(
    all_case_cost_path,
    region,
    seg_regions,
    eps=1,
    region_var="seg_adm",
    storage_options={},
):
    """Calculate the least-cost adaptation path for a given `region`, which is nested
    within a given coastal segment. All regions within a segment must take the same
    adaptation strategy.

    Parameters
    ----------
    all_case_cost_path : Path-like
        Path to Zarr store that contains ``costs`` and ``npv`` variables for each
        adaptation choice for all regions.
    region : str
        Name of region that you will calculate optimal case for
    seg_regions : list of str
        Names of all regions within this segment. NPV across all regions will be summed
        to calculate the segment-level least-cost adaptation choice.
    eps : int, default 1
        Dollars of NPV to shave off of noAdaptation npv when choosing optimal case, in
        order to avoid floating point noise driving decision for some regions. Probably
        only matters in NCC case.
    region_var : str, default "seg_adm"
        Name of dimension corresponding to region name.
    storage_options : dict, optional
        Passed to :py:function:`xarray.open_zarr`

    Returns
    -------
    :py:class:`xarray.Dataset`
        Same as the Dataset stored at `all_case_cost_path` but with only one ``case``,
        ``optimalfixed``, which represents the optimal adaptation choice for this region
        for each socioeconomic and SLR trajectory.
    """

    opt_case = (
        xr.open_zarr(
            str(all_case_cost_path), chunks=None, storage_options=storage_options
        )
        .npv.sel({region_var: seg_regions})
        .drop_sel(case="optimalfixed")
        .sum(region_var)
    )

    # in case of a tie, we don't want floating point precision noise to determine the
    # choice so we artificially shave 1 dollar off of the noAdaptation npv
    opt_case = opt_case.where(opt_case.case != "noAdaptation", opt_case - eps).idxmin(
        "case"
    )

    opt_case_ser = opt_case.to_series()
    opt_val = (
        pd.Series(
            pd.Series(CASE_DICT).loc[opt_case_ser].values,
            index=opt_case_ser.index,
        )
        .to_xarray()
        .astype("uint8")
    )

    out = (
        xr.open_zarr(
            str(all_case_cost_path), chunks=None, storage_options=storage_options
        )[["costs", "npv"]]
        .sel({region_var: [region]})
        .sel(case=opt_case)
        .drop("case")
        .expand_dims(case=["optimalfixed"])
    )
    out["optimal_case"] = opt_val.expand_dims({region_var: [region]})
    return out


def execute_pyciam(
    params_path,
    econ_input_path,
    slr_input_paths,
    slr_names,
    refA_path,
    econ_input_path_seg=None,
    surge_input_paths=None,
    output_path=None,
    tmp_output_path=AnyPath("pyciam_tmp_results.zarr"),
    remove_tmpfile=True,
    overwrite=False,
    mc_dim="quantile",
    seg_var="seg_adm",
    seg_var_subset=None,
    adm_var="adm1",
    quantiles=[0.5],
    extra_attrs={},
    econ_input_seg_chunksize=100,
    surge_batchsize=700,
    surge_seg_chunksize=5,
    refA_seg_chunksize=500,
    pyciam_seg_chunksize=3,
    diaz_inputs=False,
    diaz_config=False,
    dask_client_func=Client,
    storage_options=None,
    **model_kwargs
):
    """Execute the full pyCIAM model. The following inputs are assumed:

    - A socioeconomic input file in the format of `SLIIDERS`, organized by the
      intersection of coastal segments and some form of administrative unit (admin-1 in
      SLIIDERS).
    - A list of one or more sea level rise projections

    In addition, the following intermediate datasets will be created if they do not yet
    exist:

    - A socioeconomic input file collapsed across administrative unit to the coastal
      segment level.
    - A lookup table for mortality and capital stock impacts of extreme sea levels.
    - A dataset of initial adaptation heights calculated by running pyCIAM under an
      assumption of no climate change and allowing each segment to adapt to their
      optimal height

    In addition to these intermediate files, this function will write its output: A
    dataset of costs by administrative unit under all sea level rise scenarios.

    Parameters
    ----------
    params_path : Path-like
        Path to a json file containing model parameters. See the pyCIAM github repo for
        an example.
    econ_input_path : Path-like
        Path to a SLIIDERS-like file.
    slr_input_paths : Iterable of Path-like
        An iterable of sea level rise projection files for which pyCIAM will estimate
        costs. Note that the order matters insomuch as the "no climate change" scenario
        from the first file will be used to calculate initial adaptation heights (see
        refA_path).
    slr_names : Iterable of str
        Names associated with each slr dataset. Must be same length as `slr_input_paths`
    refA_path : Path-like
        Path to intermediate output representing initial adaptation heights. This may be
        pre-calculated. If it does not exist at the designated path, it will be created
        within this function.
    econ_input_path_seg : Path-like, optional
        Path to a version of a SLIIDERS-like file that has been collapsed over the
        administrative unit dimension. If this intermediate zarr store does not yet
        exist, it will be written within this function. This can only be left as None if
        `seg_var="seg"` in which case the data located at `econ_input_path` should
        already be indexed just by segment (as in Diaz 2016).
    surge_input_paths : dict[str, Path-like], optional
        Keys are "seg" and `seg_var`. Values are lookup tables for extreme sea level
        impacts on mortality and capital stock, indexed by either the full intersection
        of coastal segment and administrative unit (`seg_var`) or collapsed over admin
        unit ("seg"). If files do not already exist at designated paths, they will be
        created within this function. If None (default), a lookup table will not be used
        and ESL impacts will be directly estimated. Note that this is slow for large
        numbers of SLR simulations and/or segment/admin units.
    output_path : Path-like, optional
        Path to output cost predictions. If None (default), return the output as an
        xarray Dataset rather than writing to disk. This is only possible when running
        a smaller model (e.g. Diaz 2016)
    tmp_output_path : Path-like, default Path("pyciam_tmp_results.zarr")
        Path to temporary output zarr store that is written to and read from within this
        function. Ignored if `output_path` is not None.
    remove_tmpfile : bool, default True
        If True, remove the intermediate zarr store created before collapsing to
        `adm_var` and rechunking. Setting to False can be useful for debugging if you
        want to examine seg-adm level results.
    ovewrwrite : bool, default False
        If True, overwrite all intermediate output files
    mc_dim : str, default "quantile"
        The dimension of the sea level rise datasets specified at `slr_input_paths` that
        indexes different simulations within the same scenario. This could reflect Monte
        Carlo simulations *or* different quantiles of SLR. Ignored if
        `diaz_inputs=True`.
    seg_var : str, default "seg_adm"
        The coordinate of the socioeconomic input data specified at `econ_input_path`
        that indexes the intersection of coastal segments and administrative units.
    seg_var_subset : str, optional
        If not None (default), will only process segment/admin unit intersection names
        that contain this string. i.e. you can pass "_USA" to only process seg/admin
        intersections in the US.
    adm_var : str, default "adm1"
        The coordinate of the socioeconomic input data specified at `econ_input_path`
        that specifies the administrative unit associated with each admin/seg
        intersection (`seg_var`). Ignored if `seg_var=="seg"`
    quantiles : Optional[Iterable[float]], default [0.5]
        The quantiles of the sea level rise datasets specified at `slr_input_paths` that
        will be used to estimate costs within this function. If `mc_dim=="quantile"`, it
        is expected that these quantiles have been pre-computed and are a subset of the
        values of the "quantile" coordinate in each SLR dataset. If not, then quantiles
        over the simulations indexed by `mc_dim` will be calculated on-the-fly. If None,
        all simulations indexed by `mc_dim` will be used.
    extra_attrs : dict[str, str], optional
        Additional attributes to write to the output dataset.
    econ_input_seg_chunksize : int, default 100
        Chunk size for the admin unit-collapsed version of the econ input data, located
        at `econ_input_data_seg`. Ignored if this path already contains the dataset b/c
        it will not be written by this function.
    surge_batchsize : int, default 700
        Number of simultaneous segment or segment/admin groups to submit to the dask
        client when calculating the ESL lookup table. Most users will not need to modify
        this.
    surge_seg_chunksize : int, default 5
        Number of regions (segment or segment/admin) to process in each group when
        calculating the ESL impacts lookup table. This controls the memory footprint of
        this part of the code. For smaller dask workers, this can be reduced; for larger
        workers, this can be increased.
    refA_seg_chunksize : int, default 500
        Number of segments to process in each group when calculating initial adaptation
        heights. This controls the memory footprint of this part of the code. For
        smaller dask workers, this can be reduced; for larger workers, this can be
        increased.
    pyciam_seg_chunksize : int, default 3
        Number of segment/admin unit intersections to process in each group when
        calculating costs. This controls the memory footprint of this part of the code.
        For smaller dask workers, this can be reduced; for larger workers, this can be
        increased.
    dask_client_func : Callable, default Client
        Function that returns a :py:class:`dask.Client` object. By default, this creates
        a :py:class:`distributed.LocalCluster` object using the default parameters.
        Users will want to modify this based on the computing environment in which they
        are executing this function.
    storage_options : dict, optional
        Storage options that are passed to I/O functions. For example, this may take the
        form `{"token": "/path/to/application-credentials.json"}` if the filesystem on
        which your data resides requires credentials. Note that you may also need to set
        some environment variables in order for the :py:mod:`cloudpathlib` objects to
        function correctly. For example, if your data exists on Google Cloud Storage and
        requires authentication, you would need to set the
        `GOOGLE_APPLICATION_CREDENTIALS` environment variable to the same path as
        reflected in `storage_options["token"]`. Other cloud storage providers will have
        different authentication methods and have not yet been tested with this
        function.
    **model_kwargs
        Passed directly to :py:func:`pyCIAM.calc_costs`
    """

    # convert filepaths to appropriate path representation
    (
        params_path,
        econ_input_path,
        econ_input_path_seg,
        output_path,
        refA_path,
        tmp_output_path,
    ) = [
        AnyPath(f) if f is not None else None
        for f in (
            params_path,
            econ_input_path,
            econ_input_path_seg,
            output_path,
            refA_path,
            tmp_output_path,
        )
    ]
    if surge_input_paths is None:
        surge_input_paths = {k: None for k in {"seg", seg_var}}
    else:
        surge_input_paths = {k: AnyPath(v) for k, v in surge_input_paths.items()}
    slr_input_paths = [AnyPath(f) if f is not None else None for f in slr_input_paths]

    if seg_var == "seg":
        adm_var = "seg"

    # read parameters
    params = pd.read_json(params_path)["values"]

    # determine whether to check for finished jobs
    if output_path is None:
        check = False
        tmp_output_path = None
    else:
        check = True

    attr_dict = {
        "updated": pd.Timestamp.now(tz="US/Pacific").strftime("%c"),
        "planning_period_start_years": params.at_start,
        **extra_attrs,
    }

    # update model kwargs if diaz_config=True
    if diaz_config:
        eps = 0
        model_kwargs.update(
            dict(
                diaz_protect_height=True,
                diaz_construction_freq=True,
                diaz_lslr_plan=True,
                diaz_negative_retreat=True,
                diaz_forward_diff=True,
                diaz_fixed_vars_for_onetime_cost=True,
                diaz_calc_noadapt_damage_w_lslr=True,
                diaz_storm_calcs=True,
            )
        )
    else:
        eps = 1

    # Instantiate dask client
    client = dask_client_func()

    ###########################################
    # create seg-level econ inputs if necessary
    ###########################################
    if seg_var == "seg":
        econ_input_path_seg = econ_input_path
    else:
        if overwrite or not econ_input_path_seg.is_dir():
            collapse_econ_inputs_to_seg(
                econ_input_path,
                econ_input_path_seg,
                seg_var_subset=seg_var_subset,
                output_chunksize=econ_input_seg_chunksize,
                storage_options=storage_options,
                seg_var=seg_var,
            )

    ########################################
    # create surge lookup table if necessary
    ########################################
    surge_futs = {}
    for var, path in surge_input_paths.items():
        if path is None:
            continue
        if overwrite or not path.is_dir():
            if var == seg_var:
                this_econ_input = econ_input_path
            elif var == "seg":
                this_econ_input = econ_input_path_seg
            else:
                raise ValueError(var)
            surge_futs[var] = lookup.create_surge_lookup(
                this_econ_input,
                slr_input_paths,
                path,
                var,
                params.at_start,
                params.n_interp_pts_lslr,
                params.n_interp_pts_rhdiff,
                getattr(damage_funcs, params.ddf + "_i"),
                getattr(damage_funcs, params.dmf + "_i"),
                seg_var_subset=seg_var_subset,
                quantiles=quantiles,
                start_year=params.model_start,
                slr_0_years=params.slr_0_year,
                client=client,
                client_kwargs={"batch_size": surge_batchsize},
                force_overwrite=True,
                seg_chunksize=surge_seg_chunksize,
                mc_dim=mc_dim,
                storage_options=storage_options,
            )
    # block on this calculation
    wait(surge_futs)

    ###############################
    # define temporary output store
    ###############################

    ciam_in = subset_econ_inputs(
        xr.open_zarr(
            str(econ_input_path), chunks=None, storage_options=storage_options
        ),
        seg_var,
        seg_var_subset,
    )
    this_seg = ciam_in[seg_var][0].item()
    if diaz_inputs:
        test_inputs, slr = load_diaz_inputs(
            econ_input_path, [this_seg], params, storage_options=storage_options
        )
    else:
        test_inputs = load_ciam_inputs(
            econ_input_path,
            slr_input_paths,
            params,
            [this_seg],
            slr_names=slr_names,
            seg_var=seg_var,
            surge_lookup_store=None,
            mc_dim=mc_dim,
            quantiles=quantiles,
            storage_options=storage_options,
        )
        slr = test_inputs[1].unstack("scen_mc")
        test_inputs = test_inputs[0]

    if output_path is not None:
        coords = OrderedDict(
            {
                "case": CASES,
                "costtype": COSTTYPES,
                seg_var: ciam_in[seg_var].values,
                "scenario": slr.scenario,
                "quantile": quantiles,
                "year": np.arange(params.model_start, ciam_in.year.max().item() + 1),
                **{
                    dim: ciam_in[dim].values
                    for dim in ["ssp", "iam"]
                    if dim in ciam_in.dims
                },
            }
        )

        chunks = {seg_var: 1, "case": len(coords["case"]) - 1}
        chunks = {k: -1 if k not in chunks else chunks[k] for k in coords}

        # create arrays
        cost_dims = coords.keys()

        out_ds = create_template_dataarray(cost_dims, coords, chunks).to_dataset(
            name="costs"
        )
        out_ds["npv"] = out_ds.costs.isel(year=0, costtype=0, drop=True).astype(
            "float64"
        )
        out_ds["optimal_case"] = out_ds.npv.isel(case=0, drop=True).astype("uint8")

        # add attrs
        out_ds.attrs.update(attr_dict)
        out_ds = add_attrs_to_result(out_ds)

        if overwrite or not tmp_output_path.is_dir():
            out_ds.to_zarr(
                str(tmp_output_path),
                compute=False,
                mode="w",
                storage_options=storage_options,
            )

    ####################################################
    # Create initial adaptaion heights dataset if needed
    ####################################################
    if overwrite or not refA_path.is_dir():
        segs = np.unique(ciam_in.seg)
        seg_grps = [
            segs[i : i + refA_seg_chunksize]
            for i in range(0, len(segs), refA_seg_chunksize)
        ]

        (
            dataarray_from_delayed(
                client.map(
                    get_refA,
                    seg_grps,
                    econ_input_path=econ_input_path_seg,
                    slr_input_path=slr_input_paths[0],
                    params=params,
                    surge_input_path=surge_input_paths["seg"],
                    mc_dim=mc_dim,
                    storage_options=storage_options,
                    quantiles=quantiles,
                    diaz_inputs=diaz_inputs,
                    eps=eps,
                    **model_kwargs
                ),
                dim="seg",
            )
            .to_dataset(name="refA")
            .chunk({"seg": -1})
            .to_zarr(str(refA_path), storage_options=storage_options, mode="w")
        )

    ###############################
    # get groups for running pyCIAM
    ###############################
    groups = [
        ciam_in[seg_var].isel({seg_var: slice(i, i + pyciam_seg_chunksize)}).values
        for i in np.arange(0, len(ciam_in[seg_var]), pyciam_seg_chunksize)
    ]

    # get groups for aggregating seg-adms up to segs
    if seg_var == "seg":
        most_segadm = 1
    else:
        most_segadm = ciam_in.length.groupby("seg").count().max().item()
    i = 0
    agg_groups = []
    while i < len(ciam_in.seg):
        this_group = ciam_in.isel({seg_var: slice(i, i + most_segadm)})
        if len(np.unique(this_group.seg)) == 1:
            i += most_segadm
        else:
            this_group = this_group.isel(
                {seg_var: this_group.seg != this_group.seg.isel(seg_adm=-1, drop=True)}
            )
            i += len(this_group[seg_var])

        agg_groups.append(this_group[seg_var].values)

    groups_ser = (
        pd.Series(groups)
        .explode()
        .reset_index()
        .rename(columns={"index": "group_id", 0: seg_var})
        .set_index(seg_var)
        .group_id
    )

    #########################################################
    # Run 1st stage (estimate costs for each adaptation type)
    #########################################################
    ciam_futs = np.array(
        client.map(
            calc_all_cases,
            groups,
            params=params,
            econ_input_path=econ_input_path,
            slr_input_paths=slr_input_paths,
            slr_names=slr_names,
            output_path=tmp_output_path,
            refA_path=refA_path,
            surge_input_path=surge_input_paths[seg_var],
            seg_var=seg_var,
            mc_dim=mc_dim,
            quantiles=quantiles,
            storage_options=storage_options,
            diaz_inputs=diaz_inputs,
            check=check,
            **model_kwargs
        )
    )

    if output_path is None and seg_var == "seg":
        out = add_attrs_to_result(
            xr.concat(
                client.gather(
                    client.map(
                        optimize_case_seg,
                        ciam_futs,
                        dfact=test_inputs.dfact,
                        npv_start=test_inputs.npv_start,
                    )
                ),
                dim="seg",
            )
        )
        out.attrs.update(attr_dict)
        return out

    ##############################################
    # Run 2nd stage (calculate optimal adaptation)
    ##############################################
    seg_adm_ser = pd.Series(ciam_in[seg_var].values)
    seg_adm_ser.index = ciam_in.seg.values
    seg_grps = seg_adm_ser.groupby(seg_adm_ser.index).apply(list)
    precurser_futs = (
        seg_adm_ser.to_frame(seg_var)
        .join(seg_grps.rename("seg_group"))
        .set_index(seg_var)
        .seg_group.explode()
        .to_frame()
        .join(groups_ser, on="seg_group")
        .groupby(seg_var)
        .group_id.apply(set)
        .apply(list)
        .apply(lambda x: ciam_futs[x])
    )
    ciam_futs_2 = precurser_futs.reset_index(drop=False).apply(
        lambda row: client.submit(
            optimize_case,
            row[seg_var],
            *row.group_id,
            econ_input_path=econ_input_path,
            output_path=tmp_output_path,
            seg_var=seg_var,
            eps=eps,
            check=check,
            storage_options=storage_options
        ),
        axis=1,
    )

    ###############################
    # Rechunk and save final
    ###############################
    wait(ciam_futs_2.tolist())
    assert [f.status == "finished" for f in ciam_futs_2.tolist()]
    client.cancel(ciam_futs_2)
    del ciam_futs_2

    this_chunksize = pyciam_seg_chunksize * 3

    out = (
        xr.open_zarr(
            str(tmp_output_path),
            storage_options=storage_options,
            chunks={"case": -1, seg_var: this_chunksize},
        )
        .drop("npv")
        .chunk({"year": 10})
        .persist()
    )
    if adm_var != seg_var:
        out["costs"] = (
            out.costs.groupby(ciam_in[adm_var]).sum().chunk({adm_var: this_chunksize})
        ).persist()
        out["optimal_case"] = (
            out.optimal_case.load().groupby(ciam_in.seg).first(skipna=False).chunk()
        ).persist()
        out = out.drop(seg_var)
    out = out.unify_chunks()

    for v in out.data_vars:
        out[v].encoding.clear()

    for k, v in out.coords.items():
        if v.dtype == object:
            out[k] = v.astype("unicode")

    out = out.persist()

    out.to_zarr(str(output_path), storage_options=storage_options, mode="w")

    ###############################
    # Final checks and cleanup
    ###############################
    assert (
        xr.open_zarr(str(output_path), storage_options=storage_options)
        .costs.notnull()
        .all()
    )
    client.cluster.close()
    client.close()
    if remove_tmpfile:
        if isinstance(tmp_output_path, CloudPath):
            tmp_output_path.rmtree()
        else:
            rmtree(tmp_output_path)


def get_refA(
    segs,
    econ_input_path,
    slr_input_path,
    params,
    surge_input_path=None,
    mc_dim="quantile",
    storage_options={},
    quantiles=[0.5],
    eps=1,
    diaz_inputs=False,
    **model_kwargs
):
    if diaz_inputs:
        inputs, slr = load_diaz_inputs(
            econ_input_path,
            segs,
            params,
            storage_options=storage_options,
            include_cc=False,
            include_ncc=True,
        )
        surge = None
    else:
        inputs, slr, surge = load_ciam_inputs(
            econ_input_path,
            slr_input_path,
            params,
            segs,
            surge_lookup_store=surge_input_path,
            mc_dim=mc_dim,
            include_cc=False,
            include_ncc=True,
            storage_options=storage_options,
            quantiles=quantiles,
            **params.refA_scenario_selectors
        )
        slr = slr.unstack("scen_mc")
    slr = slr.squeeze(drop=True)

    costs, refA = calc_costs(
        inputs, slr, surge_lookup=surge, return_year0_hts=True, **model_kwargs
    )

    costs = costs.sel(case=SOLVCASES)
    refA = refA.sel(case=SOLVCASES)

    # In our case, we want to assume that agents are weighing the full costs of
    # retreating, not giving them a free "spin-up" step, so we use all model years
    # including the first to calculate NPV rather than starting with "npv_start"
    npv = (
        (costs.sum("costtype") * inputs.dfact)
        .sel(year=slice(inputs.npv_start, None))
        .sum("year")
    )

    # for some distributions of capital and storm surge, there will be no difference
    # between the "no adaptation" costs and the "retreat1" costs. This will results in
    # a choice based on floating point noise. To correct for that, we artificially bump
    # down the noAdaptation npv by one dollar so that we choose a refA of 0 in these
    # ambiguous cases. This was not an issue in diaz 2016 due to assumptions of
    # homogenous population/capital density and lower resolution elevation slices.
    npv = npv.where(npv.case != "noAdaptation", npv - eps)

    lowest = npv.argmin("case").astype("uint8")
    refA = refA.isel(case=lowest)
    refA["case"] = lowest

    return refA


def calc_all_cases(
    seg_adms,
    params,
    econ_input_path,
    slr_input_paths,
    slr_names,
    output_path,
    refA_path,
    surge_input_path=None,
    seg_var="seg_adm",
    mc_dim="quantile",
    quantiles=[0.5],
    storage_options={},
    check=True,
    diaz_inputs=False,
    **model_kwargs
):
    if check_finished_zarr_workflow(
        finalstore=output_path if check else None,
        varname="costs",
        final_selector={seg_var: seg_adms, "case": CASES[:-1]},
        storage_options=storage_options,
    ):
        return None

    segs = ["_".join(seg_adm.split("_")[:2]) for seg_adm in seg_adms]

    if diaz_inputs:
        inputs, slr = load_diaz_inputs(
            econ_input_path, segs, params, storage_options=storage_options
        )
        surge = None
    else:
        inputs, slr, surge = load_ciam_inputs(
            econ_input_path,
            slr_input_paths,
            params,
            seg_adms,
            slr_names=slr_names,
            seg_var=seg_var,
            surge_lookup_store=surge_input_path,
            mc_dim=mc_dim,
            quantiles=quantiles,
            storage_options=storage_options,
        )
    assert inputs.notnull().all().to_array("tmp").all()
    assert slr.notnull().all()
    if surge is not None:
        assert surge.notnull().all().to_array("tmp").all()

    # get initial adaptation height
    refA = (
        xr.open_zarr(str(refA_path), storage_options=storage_options, chunks=None)
        .refA.sel(seg=segs)
        .drop_vars("case")
    )
    refA["seg"] = seg_adms
    if "movefactor" in refA.dims:
        refA = refA.sel(movefactor=params.movefactor, drop=True)

    out = calc_costs(
        inputs,
        slr.unstack(),
        surge_lookup=surge,
        elev_chunksize=None,
        min_R_noadapt=refA,
        **model_kwargs
    ).to_dataset(name="costs")
    if seg_var != "seg":
        out = out.rename(seg=seg_var)

    out["npv"] = (
        (out.costs.sum("costtype") * inputs.dfact)
        .sel(year=slice(inputs.npv_start, None))
        .sum("year")
    )
    if output_path is not None:
        save_to_zarr_region(out, output_path, storage_options=storage_options)
        return None
    return out


def optimize_case(
    seg_adm,
    *wait_futs,
    econ_input_path=None,
    output_path=None,
    seg_var="seg_adm",
    check=True,
    eps=1,
    storage_options={}
):
    # use last fpath to check if this task has already been run
    if check and check_finished_zarr_workflow(
        finalstore=output_path if check else None,
        varname="costs",
        final_selector={seg_var: seg_adm, "case": CASES[-1]},
        storage_options=storage_options,
    ):
        return None

    seg = "_".join(seg_adm.split("_")[:2])
    with xr.open_zarr(
        str(econ_input_path), storage_options=storage_options, chunks=None
    ) as ds:
        all_segs = ds.seg.load()

    this_seg_adms = all_segs[seg_var].isel({seg_var: all_segs.seg == seg}).values

    save_to_zarr_region(
        select_optimal_case(
            output_path,
            seg_adm,
            this_seg_adms,
            eps=eps,
            region_var=seg_var,
            storage_options=storage_options,
        ),
        output_path,
        storage_options=storage_options,
    )

    return None


def optimize_case_seg(costs, dfact, npv_start):
    this_costs = costs.sel(case=SOLVCASES)
    optimal_case = this_costs.npv.argmin("case").astype("uint8")
    costs = xr.concat(
        (
            this_costs.costs,
            this_costs.costs.isel(case=optimal_case)
            .drop("case")
            .expand_dims(case=["optimalfixed"]),
        ),
        dim="case",
    )
    return xr.Dataset({"costs": costs, "optimal_case": optimal_case})
