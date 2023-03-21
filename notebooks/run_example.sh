#!/bin/bash
set -e

: '
NOTES
-----
1. Prior to running this script, you should ensure filepaths in `shared.py` are valid
   and that you have sufficient storage space
2. For the pyCIAM execution notebook (run-pyCIAM-slrquantiles.ipynb), this will
   instantiate a dask cluster. If you are running with different computing resources
   and would like to use a different type of Dask cluster, you must configure that
   yourself. By default, the script runs using a LocalCluster, which may or may not have
   sufficient memory with default configuration.
'

# define alias to run nb's from command line
run_nb() {
    NBPATH=$1.ipynb
    OUTPATH=$(pwd)/nb_logs/$NBPATH
    mkdir -p $(dirname "$OUTPATH")
    shift
    papermill $NBPATH $OUTPATH \
        --cwd $(dirname "$NBPATH") \
        "$@"
}

# Download necessary inputs (ignoring inputs used in Depsky 2023 but not crucial for an
# example run)
run_nb data-acquisition \
    -p DOWNLOAD_DIAZ_INPUTS False \
    -p DOWNLOAD_PLOTTING_DATA False \
    -p DOWNLOAD_SLR_AR5 False \
    -p DOWNLOAD_SLR_SWEET False

# reformat the raw downloaded SLR projections
run_nb data-processing/slr/AR6

# execute pyCIAM
run_nb models/run-pyCIAM-slrquantiles -p SEG_ADM_SUBSET USA -p AR6_ONLY True
