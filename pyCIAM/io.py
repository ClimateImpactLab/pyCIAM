"""This module contains functions related to loading and saving inputs and intermediate
outputs used in running pyCIAM.

Functions
    prep_sliiders
    load_ciam_inputs
    load_diaz_inputs
"""

import dask.array as da
import numpy as np
import pandas as pd
import xarray as xr

from ._utils import _s2d


def prep_sliiders(input_store, seg_vals, constants={}, seg_var="seg_adm", selectors={}):
    """Import the SLIIDERS-ECON dataset (or a different dataset formatted analogously),
    format, and calculate derived variables so that it can be used by the functions that
    implement pyCIAM.

    Parameters
    ----------
    input_store : Path-like
        Path to zarr store that contains the SLIIDERS-ECON dataset (or one formatted
        similarly. May be a :py:class:`fsspec.FSMap` object.
    seg_vals : list of str
        Defines the subset of regions (along dimension `seg_var`) that the function
        will prep. Subsets are used to run CIAM in parallel.
    constants : dict, default {}
        Defines a list of constants, typically contained in a JSON file, which will be
        merged as new variables into the :py:class:`xarray.Dataset` along with the
        processed variables from `input_store`.
    seg_var : str, default "seg_var"
        The name of the dimension in `input_store` along which the function will
        subset using `seg_vals`
    selectors : dict
        Defines additional dimensions and values with which to subset `input_store`.
        For example, the `ssp` dimension in SLIIDERS-ECON defines different
        Shared Socioeconomic Pathways (SSPs). If you only wish to proocess SSP2, you
        could specify ``selectors={"ssp": "SSP2"}``.

    Returns
    -------
    :py:class:`xarray.Dataset`
        A subsetted, processed, and reformatted version of `input_store`, with values
        from `constants` having been added.

    Notes
    -----
    * Most often, you will want to use the py:func:`.load_ciam_inputs` function, which
      calls py:func:`.io.prep_sliiders` under the hood.
    * The SLIIDERS-ECON-like dataset may contain cross-sectional variables
      reflecting the geographic and elevational distribution of exposure in a certain
      year (`pop_YYYY` and `K_YYYY`) plus scaling variables to define how these evolve
      over time (`pop_scale` and `K_scale`). This is how  SLIIDERS-ECON 1.0 is
      constructed because it does not consider within-segment migration across elevation slices in its reference socioeconomic trajectories (i.e. in the absence of SLR).
      An alternative specification that this function will handle is one in which `pop`
      and `K` variables within the input dataset explicitly provide estimates of
      exposure for each year, scenario, segment, and elevation. This allows for more flexible assumptions regarding future internal migration but has two drawbacks.
      First, the size of the input dataset is greatly increased. Second, if using pyCIAM
      in "probabilistic" mode, in which you utilize a 2D spline function created by
      :py:func:`.create_surge_lookup`, you must specify a separate lookup table for each scenario and year rather than using just one table.
    """
    inputs_all = xr.open_zarr(input_store, chunks=None).sel(selectors, drop=True)

    inputs = inputs_all.sel({seg_var: seg_vals})
    inputs = _s2d(inputs).assign(constants)

    # assign country level vars to each segment
    for v in inputs.data_vars:
        if "country" in inputs[v].dims:
            inputs[v] = inputs[v].sel(country=inputs.seg_country, drop=True)

    if "vsl" not in inputs.data_vars:
        if "ref_income" in inputs:
            ref_income = inputs.ref_income
        else:
            ref_income = inputs_all.ypcc.sel(country="USA", drop=True)
        ref_income = ref_income.astype("float64").load()
        inputs["vsl"] = (
            inputs.vsl_ypc_mult
            * ref_income
            * (inputs.ypcc / ref_income) ** inputs.vsl_inc_elast
        )

    if "pop" not in inputs.data_vars:
        exp_year = [
            v for v in inputs.data_vars if v.startswith("pop_") and "scale" not in v
        ]
        assert len(exp_year) == 1, exp_year
        exp_year = exp_year[0].split("_")[1]
        pop_var = "pop_" + exp_year
        inputs["pop"] = inputs[pop_var] * inputs.pop_scale
        inputs = inputs.drop(pop_var)
    if "K" not in inputs.data_vars:
        K_var = "K_" + exp_year
        inputs["K"] = inputs[K_var] * inputs.K_scale
        inputs = inputs.drop(K_var)
    if "dfact" not in inputs.data_vars:
        inputs["dfact"] = (1 / (1 + inputs.dr)) ** (inputs.year - inputs.year.min())

    if "landrent" or "ypc" not in inputs.data_vars:
        popdens = (inputs.pop / inputs.landarea).fillna(0)
        if "landrent" not in inputs.data_vars:
            coastland_scale = np.minimum(
                1,
                np.maximum(
                    inputs.min_coastland_scale,
                    np.log(1 + popdens) / np.log(25),
                ),
            )
            inputs["landrent"] = inputs.interior * coastland_scale * inputs.dr

        if "ypc" not in inputs.data_vars:
            ypc_scale = np.maximum(
                inputs.min_ypc_scale,
                (popdens / inputs.ypc_scale_denom) ** inputs.ypc_scale_elast,
            )
            inputs["ypc"] = ypc_scale * inputs.ypcc

    return inputs.drop(
        [
            "pop_scale",
            "K_scale",
            "interior",
            "dr",
            "min_coastland_scale",
            "min_ypc_scale",
            "ypc_scale_denom",
            "ypc_scale_elast",
            "vsl_ypc_mult",
            "vsl_inc_elast",
        ],
        errors="ignore",
    )


def _load_scenario_mc(slr_store, include_ncc=True, include_cc=True):
    scen_mc_filter = (
        xr.open_zarr(slr_store, chunks=None)[["scenario", "mc_sample_id"]]
        .to_dataframe()
        .sort_values(["scenario", "mc_sample_id"])
        .index
    )

    if include_ncc:
        scen_mc_filter = scen_mc_filter.append(
            pd.MultiIndex.from_product(
                (
                    ["ncc"],
                    scen_mc_filter.get_level_values("mc_sample_id")
                    .unique()
                    .sort_values(),
                ),
                names=["scenario", "mc_sample_id"],
            )
        )

    if not include_cc:
        scen_mc_filter = scen_mc_filter[
            scen_mc_filter.get_level_values("scenario") == "ncc"
        ]
    return scen_mc_filter


def _load_lslr_for_ciam(
    slr_store,
    site_id,
    interp_years=None,
    scen_mc_filter=None,
    include_ncc=True,
    include_cc=True,
):

    if scen_mc_filter is None:
        scen_mc_filter = _load_scenario_mc(
            slr_store,
            include_ncc=include_ncc,
            include_cc=include_cc,
        )

    wcc = scen_mc_filter.get_level_values("scenario") != "ncc"
    scen_mc_ncc = scen_mc_filter[~wcc].droplevel("scenario").values
    scen_mc_xr_wcc = (
        scen_mc_filter[wcc]
        .to_frame()
        .reset_index(drop=True)
        .rename_axis(index="scen_mc")
        .to_xarray()
    )

    slr = _s2d(
        xr.open_zarr(slr_store, chunks=None)[["lsl_msl00", "lsl_ncc_msl00"]].sel(
            site_id=site_id, drop=True
        )
    ).drop(["lat", "lon"], errors="ignore")

    # select only the scenarios we wish to model
    if len(scen_mc_xr_wcc.scen_mc):
        slr_out = slr.lsl_msl00.sel(
            scenario=scen_mc_xr_wcc.scenario, mc_sample_id=scen_mc_xr_wcc.mc_sample_id
        ).set_index(scen_mc=["scenario", "mc_sample_id"])
    else:
        slr_out = xr.DataArray([], dims=("scen_mc",), coords={"scen_mc": []})

    if len(scen_mc_ncc):
        slr_ncc = (
            slr.lsl_ncc_msl00.sel(mc_sample_id=scen_mc_ncc)
            .expand_dims(scenario=["ncc"])
            .stack(scen_mc=["scenario", "mc_sample_id"])
        )
        slr_out = xr.concat((slr_out, slr_ncc), dim="scen_mc").sel(
            scen_mc=scen_mc_filter
        )

    # interpolate to yearly
    slr_out = slr_out.reindex(
        year=np.concatenate(([2000], slr.year.values)),
        fill_value=0,
    )

    if interp_years is not None:
        slr_out = slr_out.interp(year=interp_years)
    return slr_out


def create_template_dataarray(dims, coords, chunks, dtype="float32", name=None):
    """A utility function helpful for creatting an empty, dask-backed
    :py:class:`xarray.DataArray` with specific dimensions, coordinates, dtype, name,
    and chunk structure. This is useful for "probabilistic" mode of pyCIAM, in which
    you will run the model on a large ensemble of SLR trajectory realizations. In this
    case, we save an empty Zarr store and then parallelize the model runs across
    regions, with each processor writing to a region within the template store.

    Parameters
    ----------
    dims : list of str
        Dimensions to create.
    coords : dict
        Keys are values in `dims`. Values are a list of values along each dimension.
    chunks : dict
        Keys are values in `dims`. Values are ints defining the chunksize along each
        dimension.
    dtype : str or numpy dtype
        Passed to :py:class:`dask.array.empty`. Defines the dtype of the output array.
    name : str, optional
        Defines the name of the output array.

    Returns
    -------
    :py:class:`xarray.DataArray`
        An empty dask-backed DataArray.
    """
    lens = {k: len(v) for k, v in coords.items()}
    return xr.DataArray(
        da.empty(
            [lens[k] for k in dims], chunks=[chunks[k] for k in dims], dtype=dtype
        ),
        dims=dims,
        coords={k: v for k, v in coords.items() if k in dims},
        name=name,
    )


def create_template_dataset(var_dims, coords, chunks, dtypes):
    """A utility function helpful for creatting an empty, dask-backed
    :py:class:`xarray.Dataset` with specific variables, dimensions, coordinates, dtypes,
    and chunk structure. This is useful for "probabilistic" mode of pyCIAM, in which
    you will run the model on a large ensemble of SLR trajectory realizations. In this
    case, we save an empty Zarr store and then parallelize the model runs across
    regions, with each processor writing to a region within the template store.

    Parameters
    ----------
    var_dims : dict
        Keys are variable names. Values are lists of str, defining the dimensions
        that correspond to each variable.
    coords : dict
        Keys are values in `dims`. Values are a list of values along each dimension.
    chunks : dict
        Keys are values in `dims`. Values are ints defining the chunksize along each
        dimension.
    dtypes : dict of str or numpy dtype
        Defines the dtypes of each variable.

    Returns
    -------
    :py:class:`xarray.Dataset`
        An empty dask-backed Dataset.
    """
    das = {}
    for varname, dims in var_dims.items():
        das[varname] = create_template_dataarray(
            dims,
            {k: v for k, v in coords.items() if k in dims},
            {k: v for k, v in chunks.items() if k in dims},
            dtype=dtypes[varname],
        )
    return xr.Dataset(das)


def check_finished_zarr_workflow(
    finalstore=None,
    tmpstore=None,
    varname=None,
    final_selector={},
    mask=None,
):
    """Check if a workflow that writes to a particular region of a zarr store has
    already run. This is useful when running pyCIAM in "probabilistic" mode across a
    large ensemble of SLR trajectories, using a large, and potentially unstable Dask
    cluster to parallelize computations across regions. If a Dask worker dies, all tasks
    it has completed will be re-computed by another worker. This function is used at the
    start of a mapped computation to check if the work has already been completed and
    saved to disk. If it has, you may wish to skip re-computing it.

    In some cases, we write each worker's individual computations to temporary
    individual zarr stores, and then process different groups of them, writing the
    output of this second stage to regions of a final store. This occurs, for example,
    when the `regions` modeled within CIAM are subsets of decision-making `segment`
    agents. In this case, the first stage of pyCIAM is to calculate costs for all
    adaptation options for each region. In the second stage, we aggregate all regions
    within a segment, sum total costs, and choose an optimal adaptation strategy based
    on the lowest NPV across the segment. We then save the resulting optimal adaptation
    results to a region of a final zarr store.

    This function is designed to handle such a situation, and can check for the
    existence of both the temporary file associated with the first-stage, as well as the
    region of the final zarr store.

    Parameters
    ----------
    finalstore : Path-like, optional
        If specified, path to a final zarr store within which the calling worker is
        going to write to a specific region. If None, do not check for the presence of
        this store.
    tmpstore : Path-like, optional
        If specified, path to a temporary zarr store that will be saved by this task. If
        None, do not check for the presence of this store.
    varname : str
        Name of variable to check within the output zarrs.
    final_selector : dict, optional
        Only necessary if `finalstore` is not None. Keys are dimensions of the Zarr
        store located at `finalstore`. Values are passed to `:py:meth:DataArray.sel`
        before checking for the presence of previously written data.
    mask : DataArray-like of bool
        Mask to use before checking for all non-null values. Mask is applied to both the
        store at `finalstore` and `tmpstore`. `final_selector`, if specified, is
        applied to `finalstore` before `mask`.

    Returns
    -------
    bool :
        True if the `varname` variable within the Zarr store at `finalstore` and/or
        `tmpstore` has all non-null values, after applying `final_selector` and/or
        `mask`. If both `finalstore` and `tmpstore` are specified, both must
        contain null values for this function to return False.
    """
    finished = False
    temp = False
    if finalstore is not None:
        finished = xr.open_zarr(finalstore, chunks=None)[varname].sel(
            final_selector, drop=True
        )
        if mask is not None:
            finished = finished.where(mask, 1)
        finished = finished.notnull().all().item()
    if finished:
        return True
    if tmpstore is not None:
        if tmpstore.fs.isdir(tmpstore.root):
            try:
                temp = xr.open_zarr(tmpstore, chunks=None)
                if mask is not None:
                    temp = temp.where(mask, 1)
                if (
                    varname in temp.data_vars
                    and "year" in temp.dims
                    and len(temp.year) > 0
                ):
                    finished = temp[varname].notnull().all().item()
            except Exception:
                ...
    return finished


def save_to_zarr_region(ds_in, store, already_aligned=False):
    """Wrapper around :py:method:`xarray.Dataset.to_zarr` when specifying the `region`
    kwarg. This function allows you to avoid boilerplate to figure out the integer slice
    objects needed to pass as `region` when calling `:py:meth:xarray.Dataset.to_zarr`.

    Parameters
    ----------
    ds_in : :py:class:`xarray.Dataset` or :py:class:`xarray.DataArray`
        Dataset or DataArray to save to a specific region of a Zarr store
    store : Path-like
        Path to Zarr store
    already_aligned : bool, default False
        If True, assume that the coordinates of `ds_in` are already ordered the same
        way as those of `store`. May save some computation, but will miss-attribute
        values to coordinates if set to True when coords are not aligned.

    Returns
    -------
    None :
        No return value but `ds_in` is saved to the appropriate region of `store`.

    Raises
    ------
        ValueError
            If `ds_in` is an unnamed DataArray and `store` has more than one variable.
        AssertionError
            If any coordinate values of `ds_in` are not contiguous within `store`.
    """
    ds_out = xr.open_zarr(store, chunks=None)

    # convert dataarray to dataset if needed
    if isinstance(ds_in, xr.DataArray):
        if ds_in.name is not None:
            ds_in = ds_in.to_dataset()
        else:
            if len(ds_out.data_vars) != 1:
                raise ValueError(
                    "``ds_in`` is an unnamed DataArray and ``store`` has more than one "
                    "variable."
                )
            ds_in = ds_in.to_dataset(name=list(ds_out.data_vars)[0])

    # align
    for v in ds_in.data_vars:
        ds_in[v] = ds_in[v].transpose(*ds_out[v].dims).astype(ds_out[v].dtype)

    # find appropriate regions
    alignment_dims = {}
    regions = {}
    for r in ds_in.dims:
        if len(ds_in[r]) == len(ds_out[r]):
            alignment_dims[r] = ds_out[r].values
            continue
        alignment_dims[r] = [v for v in ds_out[r].values if v in ds_in[r].values]
        valid_ixs = np.arange(len(ds_out[r]))[ds_out[r].isin(alignment_dims[r]).values]
        n_valid = len(valid_ixs)
        st = valid_ixs[0]
        end = valid_ixs[-1]
        assert (
            end - st == n_valid - 1
        ), f"Indices are not continuous along dimension {r}"
        regions[r] = slice(st, end + 1)

    # align coords
    if not already_aligned:
        ds_in = ds_in.sel(alignment_dims)

    ds_in.drop_vars(ds_in.coords).to_zarr(store, region=regions)


def load_ciam_inputs(
    input_store,
    slr_store,
    params,
    seg_vals,
    seg_var="seg",
    surge_lookup_store=None,
    ssp=None,
    iam=None,
    scen_mc_filter=None,
    include_ncc=True,
    include_cc=True,
):
    """Load, process, and format all inputs needed to run pyCIAM.

    Parameters
    ----------
    input_store : Path-like
        Path to zarr store that contains the SLIIDERS-ECON dataset (or one formatted
        similarly. May be a :py:class:`fsspec.FSMap` object.
    slr_store : Path-like
        Path to zarr store that contains the SLIIDERS-SLR dataset (or one formatted
        similarly. May be a :py:class:`fsspec.FSMap` object.
    params : dict
        Dictionary of model parameters, typically loaded from a JSON file. See
        :file:`../params.json` for an example of the required parameters.
    seg_vals : list of str
        Defines the subset of regions (along dimension `seg_var`) that the function
        will prep. Subsets are used to run CIAM in parallel.
    seg_var : str, default "seg_var"
        The name of the dimension in `input_store` along which the function will
        subset using `seg_vals`
    surge_lookup_store : Path-like, optional
        If not None, will also load and process data from an ESL impacts lookup table
        (see `lookup.create_surge_lookup`). If included in a call to
        :py:func:`.calc_costs`, including this information will allow
        pyCIAM to run more quickly through the use of a 2D linear spline, rather than
        explicitly estimating ESL-related losses for each segment, year, socioeconomic
        scenario, and SLR trajectory.
    ssp : str or list, optional
        If specified, load only the Shared Socioeconomic Pathways socioeconomic
        scenarios from SLIIDERS-ECON. If using a similarly-formatted socioeconomic
        variable dataset, this will raise a ValueError if specified and if `ssp` is
        not a dimension of the dataset. If None, ignore the `ssp` dimension (i.e. if
        the dimension exists in the dataset, load all SSPs).
    iam : {"IIASA", "OECD", None}, default None
        If specified, load only the version of GDP growth for each loaded SSP
        corresponding to the specified Integrated Assessment Model. If not specified and
        if using SLIIDERS-ECON, load both IAMs. If using a different dataset without the
        `iam` dimension, ignore this dimension. If not using SLIIDERS-ECON, this will
        raise a ValueError if specified and if `iam` is not a dimension of the
        dataset.
    scen_mc_filter : :py:class:`pandas.MultiIndex`, optional
        A list of paired `scenario` (str) and `mc_sample_id` (int) values that
        specify a subset of the individual SLR trajectories contained in the zarr store
        at `sliiders_slr_store`. If None, run all scenario X MC sample combinations.
    include_ncc : bool, default True
        If True, include realizations from the "no-climate-change" SLR scenario (i.e.
        scenarios that include no climate change-driven contributions to SLR).
    include_cc : bool, default True
        If True, include realizations from the various climate change-driven SLR
        trajectories in SLIIDERS-SLR (or a similarly formatted input dataset). Set to
        False when running the pyCIAM "spinup" period, in which initial adaptation is
        estimated through optimizing in the no-climate change scenario.

    Returns
    -------
    inputs : :py:class:`xarray.Dataset`
        A processed and formatted dataset of socioeconomic variables used in pyCIAM.
    slr : :py:class:`xarray.Dataset`
        A processed and formatted dataset of sea level rise variables used in pyCIAM.
    surge : :py:class:`xarray.Dataset`
        A processed and formatted dataset of extreme sea level damages used in pyCIAM.
        If `surge_lookup_store` is None, this output will be None

    Raises
    ------
    ValueError
        If `ssp` or `iam` is specified and the corresponding variables are not
        present in the Zarr store located at `input_store`.
    """
    selectors = {}
    if ssp is not None:
        selectors["ssp"] = ssp
    if iam is not None:
        selectors["iam"] = iam
    inputs = prep_sliiders(
        input_store,
        seg_vals,
        constants=params,
        seg_var=seg_var,
        selectors=selectors,
    )

    if seg_var != "seg":
        inputs = inputs.drop("seg", errors="ignore").rename({seg_var: "seg"})
    inputs.load()

    # get surge lookup table
    if surge_lookup_store is not None:
        surge = (
            xr.open_zarr(surge_lookup_store, chunks=None)
            .sel({seg_var: seg_vals})
            .rename({seg_var: "seg"})
            .load()
        )
    else:
        surge = None

    # get SLR
    site_ids = inputs.SLR_site_id.values
    slr = (
        _load_lslr_for_ciam(
            slr_store,
            np.unique(site_ids),
            interp_years=inputs.year.values,
            scen_mc_filter=scen_mc_filter,
            include_ncc=include_ncc,
            include_cc=include_cc,
        )
        .sel(site_id=site_ids)
        .rename(site_id="seg")
    )
    slr["seg"] = seg_vals

    return inputs, slr, surge


def load_diaz_inputs(input_store, seg_vals, params, include_ncc=True, include_cc=True):
    """Load the original inputs used in Diaz 2016.

    Parameters
    ----------
    input_store : Path-like
        Path to zarr store that contains the formatted
        `Diaz 2016 inputs <https://github.com/delavane/CIAM/blob/master/CIAMdata.gdx>`_.
        May be a :py:class:`fsspec.FSMap` object.
    seg_vals : list of str
        Defines the subset of regions (along dimension `seg_var`) that the function
        will prep. Subsets are used to run CIAM in parallel.
    params : dict
        Dictionary of model parameters, typically loaded from a JSON file. See
        :file:`../params.json` in the pyCIAM github repository for an example of the
        required parameters.
    include_ncc : bool, default True
        If True, include realizations from the "no-climate-change" SLR scenario (i.e.
        scenarios that include no climate change-driven contributions to SLR).
    include_cc : bool, default True
        If True, include realizations from the various climate change-driven SLR
        trajectories in SLIIDERS-SLR (or a similarly formatted input dataset). Set to
        False when running the pyCIAM "spinup" period, in which initial adaptation is
        estimated through optimizing in the no-climate change scenario.

    Returns
    -------
    inputs : :py:class:`xarray.Dataset`
        A processed and formatted dataset of socioeconomic variables used in pyCIAM.
    slr : :py:class:`xarray.Dataset`
        A processed and formatted dataset of sea level rise variables used in pyCIAM.

    Notes
    -----
    * `surge` is not returned when running the original Diaz 2016 specification, as
      ESL-related losses are estimated via a pre-calculated exponential function
    """
    inputs = prep_sliiders(input_store, seg_vals, constants=params, seg_var="seg")
    ncc_inputs = inputs.rcp_pt.str.startswith("rcp0")
    lsl_ncc = inputs.lsl.isel(rcp_pt=ncc_inputs)
    lsl_wcc = inputs.lsl.isel(rcp_pt=~ncc_inputs)
    slr = xr.concat(
        [i for i, j in ((lsl_ncc, include_ncc), (lsl_wcc, include_cc)) if j],
        dim="rcp_pt",
    )
    inputs = inputs.drop_dims("rcp_pt")
    return inputs, slr
