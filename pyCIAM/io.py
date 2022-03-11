import dask.array as da
import numpy as np
import pandas as pd
import xarray as xr

from .utils import s2d


def prep_sliiders(input_store, seg_vals, constants={}, seg_var="seg_adm", selectors={}):
    inputs_all = xr.open_zarr(input_store, chunks=None).sel(selectors, drop=True)

    inputs = inputs_all.sel({seg_var: seg_vals})
    inputs = s2d(inputs).assign(constants)

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


def load_scenario_mc(
    slr_store, include_ncc=True, include_cc=True, slr_slice=slice(None)
):
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
        )[slr_slice]

    if not include_cc:
        scen_mc_filter = scen_mc_filter[
            scen_mc_filter.get_level_values("scenario") == "ncc"
        ]
    return scen_mc_filter


def load_lslr_for_ciam(
    slr_store,
    site_id,
    interp_years=None,
    scen_mc_filter=None,
    slr_slice=slice(None),
    include_ncc=True,
    include_cc=True,
):

    if scen_mc_filter is None:
        scen_mc_filter = load_scenario_mc(
            slr_store,
            include_ncc=include_ncc,
            include_cc=include_cc,
            slr_slice=slr_slice,
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

    slr = s2d(
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
    check_final=True,
    check_temp=True,
):
    finished = False
    temp = False
    if check_final:
        finished = xr.open_zarr(finalstore, chunks=None)[varname].sel(
            final_selector, drop=True
        )
        if mask is not None:
            finished = finished.where(mask, 1)
        finished = finished.notnull().all().item()
    if finished:
        return True
    if check_temp:
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
    ds_out = xr.open_zarr(store, chunks=None)

    # convert dataarray to dataset if needed
    if isinstance(ds_in, xr.DataArray):
        if ds_in.name is not None:
            ds_in = ds_in.to_dataset()
        else:
            assert len(ds_out.data_vars) == 1
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
    slr_slice=slice(None),
    include_ncc=True,
    include_cc=True,
):
    if ssp is None:
        ssp = slice(None, None)
    if iam is None:
        iam = slice(None, None)
    inputs = prep_sliiders(
        input_store,
        seg_vals,
        constants=params,
        seg_var=seg_var,
        selectors={"ssp": ssp, "iam": iam},
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
        load_lslr_for_ciam(
            slr_store,
            np.unique(site_ids),
            interp_years=inputs.year.values,
            scen_mc_filter=scen_mc_filter,
            slr_slice=slr_slice,
            include_ncc=include_ncc,
            include_cc=include_cc,
        )
        .sel(site_id=site_ids)
        .rename(site_id="seg")
    )
    slr["seg"] = seg_vals

    return inputs, slr, surge


def load_diaz_inputs(input_store, segs, constants, include_ncc=True, include_cc=True):
    inputs = prep_sliiders(input_store, segs, constants=constants, seg_var="seg")
    ncc_inputs = inputs.rcp_pt.str.startswith("rcp0")
    lsl_ncc = inputs.lsl.isel(rcp_pt=ncc_inputs)
    lsl_wcc = inputs.lsl.isel(rcp_pt=~ncc_inputs)
    lsl = xr.concat(
        [i for i, j in ((lsl_ncc, include_ncc), (lsl_wcc, include_cc)) if j],
        dim="rcp_pt",
    )
    inputs = inputs.drop_dims("rcp_pt")
    return inputs, lsl
