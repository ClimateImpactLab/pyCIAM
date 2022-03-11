import os
import zipfile
from pathlib import Path

import numpy as np
import xarray as xr
from dask.utils import tmpfile
from gcsfs import GCSFileSystem

from pyCIAM import __file__

FS = GCSFileSystem(token="/opt/gcsfuse_tokens/rhg-data.json")

SLIIDERS_VERS = "v1.0"
RES_VERS = "v1.0"

# quantiles of local SLR at which to run analysis
QUANTILES = [0.05, 0.5, 0.95]

# Output dataset attrs
HISTORY = """version 1.0: Version associated with Depsky et al. 2022"""
AUTHOR = "Ian Bolliger"
CONTACT = "ian.bolliger@blackrock.com"

# Filepaths
DIR_HOME = Path("/gcs/rhg-data/impactlab-rhg/coastal")

PATH_PARAMS = Path(__file__).parents[1] / "params.json"
PATH_PARAMS_DIAZ = Path(__file__).parents[1] / "params_diaz.json"

DIR_SLIIDERS = DIR_HOME / "sliiders"
DIR_SLIIDERS_OUTPUT = DIR_SLIIDERS / "output"
PATH_SLIIDERS_ECON = FS.get_mapper(
    DIR_SLIIDERS_OUTPUT.relative_to("/gcs") / f"sliiders-econ-{SLIIDERS_VERS}.zarr"
)
PATH_SLIIDERS_SLR = FS.get_mapper(
    DIR_SLIIDERS_OUTPUT.relative_to("/gcs") / f"sliiders-slr-{SLIIDERS_VERS}.zarr"
)

DIR_CIAM = DIR_HOME / "ciam_paper"
DIR_DATA = DIR_CIAM / "data"
DIR_RAW = DIR_DATA / "raw"
DIR_INT = DIR_DATA / "int"
DIR_RES = DIR_CIAM / "results"
DIR_FIGS = DIR_CIAM / "figures"
DIR_SHP = DIR_HOME / "data" / "raw" / "ciam_inputs" / "shapefiles"

PATH_SLIIDERS_SLR_QUANTILES = FS.get_mapper(
    DIR_INT.relative_to("/gcs") / f"sliiders-slr-{SLIIDERS_VERS}-quantiles.zarr"
)
PATH_SLIIDERS_ECON_SEG = FS.get_mapper(
    DIR_INT.relative_to("/gcs") / f"sliiders-econ-{SLIIDERS_VERS}-seg.zarr"
)

PATH_SURGE_LOOKUP_SEG = FS.get_mapper(
    DIR_INT.relative_to("/gcs") / f"surge_lookup_seg_{SLIIDERS_VERS}.zarr"
)

PATH_SURGE_LOOKUP = FS.get_mapper(
    DIR_INT.relative_to("/gcs") / f"surge_lookup_{SLIIDERS_VERS}.zarr"
)
PATH_REFA = FS.get_mapper(
    DIR_INT.relative_to("/gcs") / f"refA_by_movefactor_{SLIIDERS_VERS}.zarr"
)

PATH_DIAZ_INPUTS_RAW = FS.get_mapper(
    DIR_SLIIDERS.relative_to("/gcs") / "raw" / "CIAM_2016" / "diaz2016_inputs_raw.zarr"
)
PATH_DIAZ_INPUTS_INT = FS.get_mapper(
    DIR_INT.relative_to("/gcs") / "diaz2016_inputs.zarr"
)

PATH_MOVEFACTOR_DATA = FS.get_mapper(
    DIR_RES.relative_to("/gcs")
    / f"suboptimal_capital_by_movefactor_{SLIIDERS_VERS}.zarr"
)

PATH_QUANTILE_RES = FS.get_mapper(
    DIR_RES.relative_to("/gcs") / "pyCIAM_results_quantiles.zarr"
)
PATH_DIAZ_RES = FS.get_mapper(DIR_RES.relative_to("/gcs") / "pyCIAM_results_diaz.zarr")

PATH_PWT_RAW = DIR_SLIIDERS / "raw" / "exposure" / "ypk" / "pwt_100.xlsx"


def get_ncc_slr(inputs, seg_var="seg_adm"):
    site_ids = inputs.SLR_site_id.values
    segs = inputs[seg_var].values
    slr = (
        xr.open_zarr(PATH_SLIIDERS_SLR, chunks=None)
        .lsl_ncc_msl00.sel(site_id=site_ids, drop=True)
        .quantile(QUANTILES, dim="mc_sample_id")
        .rename(site_id="seg")
    )
    slr["seg"] = segs
    slr = slr.reindex(
        year=np.concatenate(([2000], slr.year.values)), fill_value=0
    ).interp(year=inputs.year.values)
    return slr


def upload_pkg(client, pkg_dir):
    with tmpfile(extension="zip") as f:
        zipf = zipfile.ZipFile(f, "w", zipfile.ZIP_DEFLATED)
        for root, dirs, files in os.walk(pkg_dir):
            for file in files:
                zipf.write(
                    os.path.join(root, file),
                    os.path.relpath(
                        os.path.join(root, file), os.path.join(pkg_dir, "..")
                    ),
                )
        zipf.close()
        client.upload_file(f)
