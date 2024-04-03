import os
from pathlib import Path
from zipfile import ZipFile

import geopandas as gpd
import xarray as xr
from dask_gateway import GatewayCluster

from pyCIAM import __file__

DIR_SCRATCH = Path("/tmp/ciam-scratch")

SLIIDERS_VERS = "v1.2"
RES_VERS = "v1.2"

# Cloud Storage tools (will work with local storage as well but may need to be specifiec
# for cloud buckets
STORAGE_OPTIONS = None


def _to_fuse(path):
    return Path(str(path).replace("gs://", "/gcs/"))


# quantiles of local SLR at which to run analysis
QUANTILES = [0.17, 0.5, 0.83]

# Output dataset attrs
HISTORY = """version 1.1: Version associated with Depsky et al. 2023.
version 1.2: Updated to use SLIIDERS v1.2 as input."""
AUTHOR = "Ian Bolliger"
CONTACT = "ian@reask.earth"

# AR5 SLR projections info
LOCALIZESL_COREFILES = {
    "SLRProjections190726core_SEJ_full": ["L", "H"],
    "SLRProjections170113GRIDDEDcore": [None],
    "SLRProjections200204GRIDDEDcore_D20": [None],
    "SLRProjections210628GRIDDEDcore_SROCC": [None],
}
LOCALIZESL_REV = "c9b020a0f9409cde3f6796ca936f229c90f7d5c6"
PATH_LOCALIZESL = "/home/jovyan/git-repos/LocalizeSL"


##################
# ROOT DIRECTORIES
##################
DIR_HOME = Path("/tmp/ciam")
DIR_DATA = DIR_HOME / "data"
DIR_RAW = DIR_DATA / "raw"
DIR_INT = DIR_DATA / "int"
DIR_RES = DIR_HOME / f"results-{RES_VERS}"

##################
# MODEL PARAMS
##################

# NECESSARY FOR EXAMPLE
PATH_PARAMS = Path.home() / "git-repos/pyCIAM/params.json"

PATH_PARAMS_DIAZ = Path.home() / "git-repos/pyCIAM/params_diaz.json"

##################
# SOCIOECON INPUTS
##################

# SLIIDERS

# NECESSARY FOR EXAMPLE
PATH_SLIIDERS = DIR_RAW / f"sliiders-{SLIIDERS_VERS}.zarr"

PATH_SLIIDERS_SEG = PATH_SLIIDERS.parent / (
    PATH_SLIIDERS.stem + "-seg" + PATH_SLIIDERS.suffix
)

# Diaz
PATH_DIAZ_INPUTS_RAW = DIR_RAW / "diaz2016_inputs_raw.zarr"
PATH_DIAZ_INPUTS_INT = DIR_INT / "diaz2016_inputs.zarr"

#####
# SLR
#####

DIR_SLR_RAW = DIR_RAW / "slr"
DIR_SLR_INT = DIR_INT / "slr"

# AR5 SLR Paths
DIR_SLR_AR5_RAW = DIR_SLR_RAW / "ar5"
DIR_SLR_AR5_INT = DIR_SLR_INT / "ar5"
DIR_SLR_AR5_IFILES_RAW = _to_fuse(DIR_SLR_AR5_RAW / "ifiles")
DIR_SLR_AR5_IFILES_INT = _to_fuse(DIR_SLR_AR5_INT / "ifiles")
PATH_SLR_AR5_N_GCMS = DIR_SLR_AR5_INT / "numGCMs.zarr"
PATH_SLR_AR5_FULL = DIR_SLR_AR5_INT / "ar5-msl-rel-2005-full-dist.zarr"
PATH_SLR_AR5_QUANTILES = DIR_SLR_AR5_INT / "ar5-msl-rel-2005-quantiles.zarr"

# Sweet SLR Paths
DIR_SLR_SWEET_RAW = DIR_SLR_RAW / "sweet2022"
PATH_SLR_SWEET = DIR_SLR_INT / "sweet2022-msl-rel-2005.zarr"

# AR6 SLR Paths
# NECESSARY FOR EXAMPLE
DIR_SLR_AR6_RAW = DIR_SLR_RAW / "ar6"
PATH_SLR_AR6 = DIR_SLR_INT / "ar6-msl-rel-2005.zarr"

# MSL Datum Adjustment Paths
PATH_SLR_HIST_TREND_MAP = DIR_SLR_RAW / "msl-altimetry-trend-2000-2006-G2.nc"
PATH_SLR_GMSL_HIST_TIMESERIES = (
    DIR_SLR_RAW / "MSL_Serie_MERGED_Global_AVISO_GIA_Adjust_Filter2m.nc"
)


###########################
# PYCIAM INTERMEDIATE FILES
###########################

# NECESSARY FOR EXAMPLE
PATHS_SURGE_LOOKUP = {}
for seg in ["seg_adm", "seg"]:
    PATHS_SURGE_LOOKUP[seg] = DIR_INT / f"surge-lookup-{SLIIDERS_VERS}-{seg}.zarr"

# NECESSARY FOR EXAMPLE
PATH_REFA = DIR_INT / f"refA_by_movefactor_{SLIIDERS_VERS}.zarr"


###########################
# PYCIAM OUTPUTS
###########################
# NECESSARY FOR EXAMPLE
PATH_OUTPUTS = DIR_RES / "pyCIAM_outputs.zarr"

PATH_DIAZ_RES = DIR_RES / "diaz2016_outputs.zarr"


############################
# FILES FOR PLOTTING RESULTS
############################
PATH_MOVEFACTOR_DATA = DIR_RES / "suboptimal_capital_by_movefactor.zarr"
PATH_SLIIDERS_INCOME_INTERMEDIATE_FILE = DIR_RAW / "ypk_2000_2100_20240222.zarr"

DIR_FIGS = Path("/home/jovyan/ciam-figures")
DIR_SHP = DIR_RAW / "shapefiles"
PATH_PWT = DIR_RAW / "pwt_100.parquet"

PATH_BORDERS = (
    DIR_SHP
    / "ne_10m_admin_0_boundary_lines_land"
    / "ne_10m_admin_0_boundary_lines_land.shp"
)

PATH_COASTLINES = DIR_SHP / "ne_10m_coastline" / "ne_10m_coastline.shp"
PATH_GADM = DIR_SHP / "gadm_410-levels.gpkg"

# Make directories where needed
for p in [
    DIR_SCRATCH,
    DIR_RES,
]:
    p.mkdir(exist_ok=True, parents=True)


def _zipdir(
    path,
    zip_filename,
    skip_files=(
        ".git",
        ".github",
        ".pytest_cache",
        "tests",
        "docs",
        "deploy",
        "notebooks",
        ".ipynb_checkpoints",
        "__pycache__",
        ".coverage",
        "dockerignore",
        ".gitignore",
        ".gitlab-ci.yml",
        ".gitmodules",
        "pyclaw.log",
        "run_tests.sh",
    ),
):

    with ZipFile(zip_filename, "w") as ziph:
        for root, dirs, files in os.walk(path):
            for file in files:
                if any([f in file.split("/") for f in skip_files]):
                    continue
                # Create a relative path for files to preserve the directory structure
                # within the ZIP archive. This relative path is based on the directory
                # being zipped, so files are stored in the same structure.
                relative_path = os.path.relpath(
                    os.path.join(root, file), os.path.join(path, "..")
                )
                ziph.write(os.path.join(root, file), arcname=relative_path)


def upload_pyciam(client):
    package_dir = Path(__file__).parent
    zip_filename = "/tmp/pyCIAM.zip"  # Output ZIP file name
    _zipdir(package_dir, zip_filename)
    client.upload_file(zip_filename)


def save(obj, path, *args, **kwargs):
    if path.suffix == ".zarr":
        meth = "to_zarr"
    elif path.suffix == ".parquet":
        meth = "to_parquet"
    else:
        raise ValueError(type(obj))
    getattr(obj, meth)(str(path), *args, storage_options=STORAGE_OPTIONS, **kwargs)


def open_zarr(path, **kwargs):
    return xr.open_zarr(str(path), storage_options=STORAGE_OPTIONS, **kwargs)


def _generate_parent_fuse_dirs(path):
    return Path(path).parent.mkdir(exist_ok=True, parents=True)


def open_dataset(path, **kwargs):
    _path = str(_to_fuse(path))
    _generate_parent_fuse_dirs(_path)
    return xr.open_dataset(_path, **kwargs)


def save_dataset(ds, path, **kwargs):
    _path = str(_to_fuse(path))
    _generate_parent_fuse_dirs(_path)
    return ds.to_netcdf(_path, **kwargs)


def open_dataarray(path, **kwargs):
    _path = str(_to_fuse(path))
    _generate_parent_fuse_dirs(_path)
    return xr.open_dataarray(_path, **kwargs)


def read_shapefile(path, **kwargs):
    _path = str(path).replace("gs://", "/gcs/")
    _generate_parent_fuse_dirs(_path)
    return gpd.read_file(_path, **kwargs)


def start_dask_cluster(profile="micro", **kwargs):
    cluster = GatewayCluster(profile=profile, **kwargs)
    client = cluster.get_client()
    upload_pyciam(client)
    return client, cluster
