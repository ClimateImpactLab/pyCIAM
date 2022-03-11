from pathlib import Path

import numpy as np
from gcsfs import GCSFileSystem

from pyCIAM.surge import ddf_i, dmf_i

FS = GCSFileSystem(token="/opt/gcsfuse_tokens/rhg-data.json")

SLIIDERS_VERS = "v1.0"

# adaptation time period starting years
AT_START = np.arange(2000, 2100, 10)

# what year to start NPV calculations in for determining optimal adaptation
NPV_START = 2010

# integrals of depth-mortality and depth-damage surge damage functions
DMF_I = dmf_i
DDF_I = ddf_i

# How many interpolation points to make for Surge Lookup Table
N_INTERP_PTS_LSLR = 100
N_INTERP_PTS_RHDIFF = 100

# New percent depreciation of capital under proactive retreat scenario
# (Diaz2016 = 1, Lincke2021=0)
DEPR = 0

# Output dataset attrs
HISTORY = """version 1.0: Version associated with Depsky et al. 2022"""
AUTHOR = "Ian Bolliger"
CONTACT = "ibolliger@rhg.com"

# Filepaths
DIR_HOME = Path("rhg-data/impactlab-rhg/coastal")

DIR_SLIIDERS = DIR_HOME / "data/raw/ciam_inputs"
PATH_SLIIDERS_ECON = FS.get_mapper(DIR_SLIIDERS / f"sliiders-econ-{SLIIDERS_VERS}.zarr")
PATH_SLIIDERS_SLR = FS.get_mapper(DIR_SLIIDERS / f"sliiders-slr-{SLIIDERS_VERS}.zarr")

DIR_CIAM = DIR_HOME / "ciam_paper"
PATH_SURGE_LOOKUP = FS.get_mapper(DIR_CIAM / "surge_lookup.zarr")
PATH_QUANTILE_RES = FS.get_mapper(DIR_CIAM / "pyCIAM_results_quantiles.zarr")
