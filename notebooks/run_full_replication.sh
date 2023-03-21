#!/bin/bash
set -e

: '
NOTES
-----
To execute the full replication of Depsky 2023, you will need both Octave and Python
resources. Octave is only necessary for recreating the LocalizeSL SLR projections, so
you may choose to simply use the pre-computed files for this step. To do so, change the
RUN_LOCALIZESL parameter to False.

* The full workflow requires substantial computing resources. You may find a need to
  modify a fair number of default configurations (e.g. xarray chunk sizes) depending on
  your computing platform
* Prior to running this script, you should ensure filepaths in `shared.py` are valid
  and that you have sufficient storage space
* Several notebooks will execute by first instantiating a dask cluster. If you are
  running with different computing resources and would like to use a different type of
  Dask cluster, you must configure that yourself. By default, these scripts run using a
  LocalCluster, which may or may not have sufficient memory with default configuration.
'

RUN_LOCALIZESL=false

if [ "$RUN_LOCALIZESL" = true ]
then
    DOWNLOAD_AR5=False
else
    DOWNLOAD_AR5=True
fi

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
    -p DOWNLOAD_SURGE_LOOKUPS False \
    -p DOWNLOAD_SLR_AR5 $DOWNLOAD_AR5

# reformat the raw downloaded SLR projections
run_nb data-processing/slr/AR6
run_nb data-processing/slr/sweet
if [ $RUN_LOCALIZESL = true]
then
    run_nb data-processing/slr/AR5/1-convert-mat-version
    run_nb data-processing/slr/AR5/2-generate-projected-lsl
    run_nb data-processing/slr/AR5/3-retrieve-num-gcms
    run_nb data-processing/slr/AR5/4-process-localizesl-output
    run_nb data-processing/slr/AR5/5-create-slr-quantile
fi

# collapse SLIIDERS to segment from segment/admin units. This would be done anyways
# within `execute_pyciam` but we do it ahead of time for use in `fit-movefactor.ipynb`
run_nb data-processing/collapse-sliiders-to-seg

# update formatting of Diaz 2016 CIAM inputs
run_nb data-processing/create-diaz-pyCIAM-inputs

# create surge lookup tables. This would also be done anyways within `execute_pyciam`
# but we do it ahead of time for use in `fit-movefactor.ipynb`
run_nb models/create-surge-lookup-tables

# run movefactor analysis
run_nb models/fit-movefactor

# run Diaz 2016 version of pyCIAM
run_nb models/run-pyCIAM-diaz2016

# execute pyCIAM
run_nb models/run-pyCIAM-slrquantiles

# generate results
run_nb post-processing/pyCIAM-results-figures