"""This module contains the central engine of pyCIAM, in which costs for all adaptation
options are calculated.

Functions
    calc_costs
    select_optimal_case
"""

import numpy as np
import pandas as pd
import xarray as xr

from ._utils import _get_lslr_plan_data, _get_planning_period_map, _pos
from .constants import CASE_DICT, PLIST, RLIST
from .surge._calc import _calc_storm_damages_no_resilience, _get_surge_heights_probs
from .surge.damage_funcs import diaz_ddf_i, diaz_dmf_i


def calc_costs(
    inputs,
    lslr,
    surge_lookup=None,
    elev_chunksize=1,
    ddf_i=diaz_ddf_i,
    dmf_i=diaz_dmf_i,
    ddf_kwargs={},
    dmf_kwargs={},
    diaz_protect_height=False,
    diaz_construction_freq=False,
    diaz_lslr_plan=False,
    diaz_negative_retreat=False,
    diaz_forward_diff=False,
    diaz_zero_costs_in_first_year=False,
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
        A processed and formatted version of SLIIDERS-ECON (or similarly formatted
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
    ddf_i, dmf_i : func, default :py:func:`.damage_funcs.ddf_i`, :py:func:`.damage_funcs.dmf_i`
        Damage functions relating physical capital loss and monetized mortality arising
        from a certain depth of inundation.
    dmf_kwargs, ddf_kwargs : dict, optional
        Kwargs passed directly to `ddf_i` and `dmf_i`
    diaz_protect_height : bool, default False
        If True, reduce the 1-in-10-year extreme sea level by 50% as in Diaz 2016. This
        hack should not be necessary when using the ESL heights from CoDEC (as in
        SLIIDERS-ECON).
    diaz_construction_freq : bool, default False
        If True, set the lifetime over which the "linear" component of protection
        construction costs (i.e. the component that is proportional to length and not to
        height) is amortized to the length of the adaptation periods. This means that
        the total costs from this linear portion was directly proportional to the number
        of planning periods. Because the default planning periods for pyCIAM are
        shorter, we assume that the lifetime of the seawall foundation that depends on
        this linear part is 50 years (roughly the length of the planning periods in Diaz 2016). This allows  total seawall construction costs to be roughly independent
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
    diaz_zero_costs_in_first_year : bool, default False
        If True, ignore all costs except wetland loss costs in the first year for all
        cases except "reactive retreat", as in Diaz 2016. If False, account for these
        costs.
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
      exist in SLIIDERS-ECON, for which growth is homogeonous within each country.
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
    (lslr_plan_noadapt, RH_heights, RH_heights_prev,) = _get_lslr_plan_data(
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
        RH_heights.sel(at=at, drop=True).isel(return_period=slice(None, -1)) - lslr
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
        surge_pop = surge_cap * inputs.floodmortality
        surge = xr.concat(
            (surge_cap, surge_pop),
            dim=pd.Index(["stormCapital", "stormPopulation"], name="costtype"),
        ).to_dataset("costtype")
        surge_cap_noadapt = sigma_noadapt / tot_landarea
        surge_pop_noadapt = surge_cap_noadapt * inputs.floodmortality
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
        surge_noadapt[
            "stormPopulation"
        ] = surge_noadapt.stormPopulation * inputs.pop.sum("elev")
    else:
        if surge_lookup is None:
            rh_years = RH_heights.sel(at=at, drop=True)
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
                ddf_kwargs=ddf_kwargs,
                dmf_kwargs=dmf_kwargs,
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
                        ddf_kwargs=ddf_kwargs,
                        dmf_kwargs=dmf_kwargs,
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
            surge_noadapt[
                "stormPopulation"
            ] = surge_noadapt.stormPopulation * inputs.pop.sum("elev")

    surge = surge.stack(case=["adapttype", "return_period"])
    surge["case"] = (surge.adapttype + surge.return_period.astype(str)).values

    # merge no adapt and retreat/protect, dropping extra protect1 which we do not allow.
    # Also, skip the last return value where we assume full protection
    surge = xr.concat(
        (surge_noadapt.expand_dims(case=["noAdaptation"]), surge), dim="case"
    ).sel(case=["noAdaptation"] + RLIST[:-1] + PLIST[:-1])
    del surge_noadapt

    # monetize deaths
    surge["stormPopulation"] *= inputs.vsl

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
    ).sel(at=at, drop=True)

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
        ).sel(at=at)
    else:
        protection = protection + (
            inputs.wall_width2height
            * RH_heights_p.sel(at=at, drop=True)
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
        # else assume no slr before 2000 (to help calculate losses in 2000)
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
                .sel(at=at, drop=True)
            )
        else:
            abandonment += (
                # Value of abandoned land following retreat
                (
                    bin_wt.sel(at=at, drop=True)
                    * this_inputs.landrent
                    * this_inputs.landarea
                ).sum("elev")
                # one-time loss of abandoned capital
                + (
                    bin_wt_diff.sel(at=at, drop=True)
                    * (1 - this_inputs.mobcapfrac)
                    * (1 - this_inputs.depr)
                    * this_inputs.K
                    / tstep_at.sel(at=at, drop=True)
                ).sum("elev")
            )

        # relocation
        if diaz_fixed_vars_for_onetime_cost:
            relocation_r += (
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
            ).sel(at=at, drop=True)
        else:
            relocation_r += (
                bin_wt_diff.sel(at=at, drop=True)
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
            ).sum("elev") / tstep_at.sel(at=at, drop=True)

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

    # Diaz 2016 assigns 0 cost to all with-adaptation scenarios in the first year, with
    # the exception of wetland costs
    if diaz_zero_costs_in_first_year:
        out.where(
            (out.year > out.year[0])
            | (out.case == "noAdaptation")
            | (out.costtype == "wetland"),
            0,
        )

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
        RH_heights["case"] = RH_heights.adapttype + RH_heights.return_period.astype(str)
        RH_heights = xr.concat(
            (
                lslr_plan_noadapt.expand_dims(case=["noAdaptation"]),
                RH_heights.sel(at=at, drop=True),
            ),
            dim="case",
        )
        if isinstance(out, list):
            out.append(RH_heights)
        else:
            out = [out, RH_heights]

    return out


def select_optimal_case(all_case_cost_path, region, seg_regions, region_var="seg_adm"):
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
    region_var : str, default "seg_adm"
        Name of dimension corresponding to region name.

    Returns
    -------
    :py:class:`xarray.Dataset`
        Same as the Dataset stored at `all_case_cost_path` but with only one ``case``,
        ``optimalfixed``, which represents the optimal adaptation choice for this region
        for each socioeconomic and SLR trajectory.
    """

    opt_case = (
        xr.open_zarr(all_case_cost_path, chunks=None)
        .npv.sel({region_var: seg_regions})
        .drop_sel(case="optimalfixed")
        .sum(region_var)
        .idxmin("case")
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
        xr.open_zarr(all_case_cost_path, chunks=None)[["costs", "npv"]]
        .sel({region_var: [region]})
        .sel(case=opt_case, drop=True)
        .expand_dims(case=["optimalfixed"])
    )
    out["optimal_case"] = opt_val.expand_dims({region_var: [region]})
    return out
