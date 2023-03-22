# pyCIAM

pyCIAM is a Python port, including numerous model updates, of the [Coastal Impacts and Adaptation Model](https://github.com/delavane/CIAM), as described in [Diaz 2016](https://link.springer.com/article/10.1007/s10584-016-1675-4#Sec13). This code accompanies Depsky et al. 2023. See the manuscript for further details on model functionality and structure.

`pyCIAM`, like `CIAM`, is a tool to estimate global economic impacts of sea level rise at fine resolution, accounting for adaptation and spatial variability and uncertainty in sea level rise. This model requires a number of socioeconomic and sea level rise inputs, organized by coastal "segment" and elevation. In Depsky et al. 2023, we develop the [SLIIDERS](https://doi.org/10.5281/zenodo.6449230) dataset to serve this purpose; however, users may wish to alter and/or replace this dataset for their own purposes, especially if newer input data (used to generate `SLIIDERS`) becomes available.

Model outputs consistent with Depsky et al. 2023 are available on Zenodo, with DOI [10.5281/zenodo.6014085](https://doi.org/10.5281/zenodo.6014085)

At present, this repository contains both the `pyCIAM` package, along with a Jupyter Notebook-based workflow used to create the outputs of Depsky et al. 2023. In the future, we will migrate the notebook workflow into its own repository to isolate the more broadly useful package within this one.

## Status

This code was developed for Depsky et al. 2023, but our intention is for broader use. If and when that broader usage develops, the package will continue to become more formalized and well-tested. In the meantime, please pardon the dust as we continue to finalize the package and develop a comprehensive testing suite. Don't hesitate to reach out by filing an issue if you have any questions about usage.

## Installation

`pyCIAM` is available via PyPI and installable via `pip`.

```bash
pip install python-CIAM
```

Note that currently, package dependencies have not been thoroughly described and must be installed separately by the user. We are working to include dependencies in the package so that all will be included in the `pip install` command.

To run the model, you will need to define parameters, which are located in [params.json](./params.json). This file will need to be either user-generated from scratch or obtained from this repository via cloning or downloading that individual file.

If you wish to use the notebook-based workflow used in Depsky et al. 2023, either to replicate these results or as a template for separate analyses, you should clone this repository. We are working on developing interactive documentation that will also contain these notebooks.

The requisite packages for executing the full Depsky et al. 2023 workflow are identified in [environment.yml](environment/environment.yml) which can be used to build a conda/mamba environment via

```bash
mamba env create -f /path/to/environment.yml
mamba activate pyciam
```

## Quickstart

To run an example set of segments through pyCIAM, execute the following steps:

1. Define a correct set of filepaths in [shared.py](./notebooks/shared.py). All filepaths should be defined as :py:class:`pathlib.Path`-like objects. Because there are many files that are written to or read from in replicating the full Depsky et al. 2023 analysis, the filepaths that are needed explicitly to run the [example script](./notebooks/run_example.sh) are indicated with a `# NEEDED FOR EXAMPLE` comment.

2. Define a `start_dask_cluster` function in [shared.py](./notebooks/shared.py). All notebooks in this repo will use this function to instantiate a dask cluster for executing the model. The type of cluster needed for your use case will depend on your computing environment. The default function simply instantiates a default :py:class:`distributed.LocalCluster` object.

3. Execute [run_example.sh](./notebooks/run_example.sh).

## Usage

To recreate the analysis of Depsky et al. 2023, you will need more inputs and to run the model on a wider range of scenarios and locations. You should ensure all of the paths in [shared.py](.notebooks/shared.py) are valid and then you will want to execute [run_full_replication.sh](./notebooks/run_full_replication.sh). Note that this may require some tuning of parallelization parameters (such as chunk size) within individual notebooks depending on the size of your resources. For reference, in development, these workflows were executed on a Dask cluster with ~6.5 GB memory per worker.

For users wishing to use `pyCIAM` in other contexts, we recommend starting with [the example shell script](./notebooks/run_example.sh) as a template. A full description of the model is available in Depsky et al. 2023, and a description of the workflow contained in each notebook is provided in [notebooks/README.md](./notebooks/README.md).

### API

pyCIAM contains only a handful of public functions that a user will want to employ when executing the model. All are available as top-level imports from pyCIAM.

* `execute_pyciam`: This is the end-to-end function that represents the most likely entrypoint for users. All other public functions are called by this one.
* `create_surge_lookup`: Creates a lookup table that can be leveraged to build a 2D linear spline function for calculating damages from extreme sea levels. This can drastically reduce computational expense required for simulation on large ensembles of sea level rise trajectories.
* `load_ciam_inputs`: An I/O function to load SLIIDERS-like input data, storm surge damage lookup table (if specified), model parameters, and process/format these data for inclusion in pyCIAM.
* `load_diaz_inputs`: A similar function to load a preprocessed SLIIDERS-like input dataset that is generated from the same input data used in Diaz 2016. This is used to generate comparisons to the Diaz 2016 results within Depsky et al. 2023.
* `calc_costs`: This is the main computation engine in pyCIAM. It computes costs for all cost types, regions, years, socioeconomic and SLR trajectories, and adaptation case. It does *not* compute the optimal adaptation case, which must be computed afterward, for reasons described below.
* `select_optimal_case`: This function calculates the optimal adaptation choice for a given region and returns the associated costs and NPV.

### Step-by-step Instructions

The below sections describe the high-level stages of the pyCIAM model. With the exception of the first two (obtaining model inputs and specifying parameters), the `execute_pyciam` wrapper will cover all the remaining steps.

#### Obtaining model inputs

pyCIAM depends on inputs describing a variety of socioeconomic and sea level variables across user-defined regions. The [SLIIDERS](https://doi.org/10.5281/zenodo.6449230) dataset has been developed to contain socioeconomic variables; however, any similarly formatted input dataset will work, if users wish to substitute, for example, alternative projections of economic growth besides the Shared Socioeconomic Pathways. To begin, we recommend that users obtain the SLIIDERS dataset, which can be found at the linked DOI.

All inputs necessary to reproduce the results in Depsky et al. 2023 can be downloaded via the scripts in [notebooks/data-acquisition](./notebooks/data-acquisition).

Additional processing of some of these inputs is needed. See notebooks in [notebooks/data-processing](./notebooks/data-processing) and their execution in [run_example.sh](./notebooks/run_example.sh) and [run_full_replication.sh](./notebooks/run_full_replication.sh).

#### Parameter specification

Parameters for the model run are defined in [params.json](./params.json). These can be modified to reflect alternative assumptions and/or model duration and timesteps. Description of each parameter can be found in the JSON file.

#### Surge lookup table creation

This step is not strictly necessary but provides dramatic performance increases. Rather than calculating the damages from extreme sea levels for all adaptation cases, for all segments, elevation slices, years, and scenarios, you may develop a lookup table that is used to build a 2D spline function to interpolate surge costs for any given combination of sea level height, and difference between storm surge height and retreat or protection height. `create_surge_lookup()` is used for this. See [create-surge-lookup-tables.ipynb](./notebooks/models/create-surge-lookup-tables.ipynb) for an example.

#### Calculation of costs for each adaptation case

In this step, costs for all adaptation options for all segments are calculated. Depending on the number of SLR trajectories modeled, segments may be run in smaller or larger batches to keep memory footprint to a reasonable size. This parallelization is executed via [Dask](https://www.dask.org). `calc_costs()` is the function that executes these calculations. An example of usage can be seen in [fit-movefactor.ipynb](./notebooks/models/fit-movefactor.ipynb).

#### Calculation of the optimal adaptation choice

In this step, we find the optimal adaptation choice for each segment. `select_optimal_case()` is the function that executes this calculation. This is not called directly in any of the workflows in this repository but is called within `execute_pyciam`

## Reliance on Dask

**IMPORTANT**: The notebook-based workflow provided in this repository serves as a set of examples in addition to replicating the analysis in Depsky et al. 2023. It assumes that parallel task execution occurs via [Dask](https://dask.org). During development, a [dask-gateway](https://gateway.dask.org/) server was used on top of a Kubernetes cluster. Users wishing to run these notebooks must specify the type of Dask cluster they wish to use by replacing the default `start_dask_cluster()` function in [shared.py](./notebooks/shared.py). By default, a :py:class:`distributed.LocalCluster` instance will be created.

## Support

Please file an issue for any problems you encounter

## Contributing

We encourage community contributions and hope that the functionality of `pyCIAM` will grow as a result. At the moment, we have no contribution template. Please fork the project and file a Merge Request to propose your addition. Clearly define the contribution that the Merge Request is making and, when any issues have been resolved, we will merge the new code.

## Authors

The original authors of this code include:

* Daniel Allen
* Ian Bolliger
* Junho Choi
* Nicholas Depsky

## License

This code is licensed under the [MIT License](./LICENSE)
