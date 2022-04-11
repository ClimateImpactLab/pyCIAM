# pyCIAM

pyCIAM is a Python port, including numerous model updates, of the [Coastal Impacts and Adaptation Model](https://github.com/delavane/CIAM), as described in [Diaz 2016](https://link.springer.com/article/10.1007/s10584-016-1675-4#Sec13). This code accompanies Depsky et al. 2022 (in prep.). See the manuscript for further details on model functionality and structure.

`pyCIAM`, like `CIAM`, is a tool to estimate global economic impacts of sea level rise at fine resolution, accounting for adaptation and spatial variability and uncertainty in sea level rise. This model requires a number of socioeconomic and sea level rise inputs, organized by coastal "segment" and elevation. In Depsky et al. 2022, we develop the [SLIIDERS](https://doi.org/10.5281/zenodo.6449231) datasets to serve this purpose; however, users may wish to alter and/or replace these datasets for their own purposes, especially if newer input data (used to generate `SLIIDERS`) becomes available.

Model outputs consistent with Depsky et al. 2022 are available on Zenodo, with DOI [10.5281/zenodo.6014086](https://doi.org/10.5281/zenodo.6014086)

At present, this repository contains both the `pyCIAM` package, along with a Jupyter Notebook-based workflow used to create the outputs of Depsky et al. 2022. In the near future, we will likely extract the workflow into its own repository to isolate the more broadly useful package within this one.

## Status

This code was developed for Depsky et al. 2022, but our intention is for broader use. If and when that broader usage develops, the package will continue to become more formalized and well-tested. In the meantime, please pardon the dust as we continue to finalize the package and develop a comprehensive testing suite. Don't hesitate to reach out by filing an issue if you have any questions about usage.

## Installation

`pyCIAM` is available via PyPI and installable via `pip`.

```bash
pip install python-CIAM
```

Note that currently, package dependencies have not been thoroughly described and must be installed separately by the user. We are working to include dependencies in the package so that all will be included in the `pip install` command.

To run the model, you will need to define parameters, which are located in [params.json](./params.json). This file will need to be either user-generated from scratch or obtained from this repository via cloning or downloading that individual file.

If you wish to use the notebook-based workflow used in Depsky et al. 2022, either to replicate these results or as a template for separate analyses, you should clone this repository. We are working on developing interactive documentation that will also contain these notebooks.

## Usage

To recreate the analysis of Depsky et al. 2022, you may follow the workflow contained within [notebooks](./notebooks). See the [README](./notebooks/README.md) file within that directory for more details on notebook execution.

For users wishing to use `pyCIAM` in other contexts, we still recommend starting with that workflow as a template. A full description of the model is available in Depsky et al. 2022.

### API

pyCIAM contains only a handful of public functions that a user will want to employ when executing the model. All are available as top-level imports from pyCIAM.

* `create_surge_lookup`: Creates a lookup table that can be leveraged to build a 2D linear spline function for calculating damages from extreme sea levels. This can drastically reduce computational expense required for simulation on large ensembles of sea level rise trajectories.
* `load_ciam_inputs`: An I/O function to load SLIIDERS-like input data, storm surge damage lookup table (if specified), model parameters, and process/format these data for inclusion in pyCIAM.
* `load_diaz_inputs`: A similar function to load a preprocessed SLIIDERS-like input dataset that is generated from the same input data used in Diaz 2016. This is used to generate comparisons to the Diaz 2016 results within Depsky et al. 2022.
* `calc_costs`: This is the main computation engine in pyCIAM. It computes costs for all cost types, regions, years, socioeconomic and SLR trajectories, and adaptation case. It does *not* compute the optimal adaptation case, which must be computed afterward, for reasons described below.
* `select_optimal_case`: This function calculates the optimal adaptation choice for a given region and returns the associated costs and NPV.

### Step-by-step Instructions

Running pyCIAM takes several stages, outlined below. One additional capability that pyCIAM provides over the original CIAM implementation of Diaz 2016 is the option of specifying administrative regions that intersect with the decision-making coastal segments. This allows for aggregating outputs to arbitrary regions, but increases the complexity of model execution because least-cost adaptation cases must be chosen at the segment level, while results must be aggregated across different regions. The steps described below assume this configuration, matching the analysis in Depsky et al. 2022, which projects costs by "Admin 1" region. Several steps may be simplified if you do not need to aggregate to separate geographies than those defined by the decision-making coastal segments. Im particular, [cost calculation](#calculation-of-costs-for-each-adaptation-case) and [optimal case selection](#calculation-of-costs-for-each-adaptation-case) may not need to be separated. In Diaz 2016, for example, the `region` dimension did not exist. Thus [run-pyCIAM-diaz2016.ipynb](./notebooks/run-pyCIAM-diaz2016.ipynb) provides an example for executing pyCIAM without needing to separate cost calculation and optimal case selection

#### Obtaining model inputs

pyCIAM depends on inputs describing a variety of socioeconomic and sea level variables across user-defined regions. The [SLIIDERS-ECON](https://doi.org/10.5281/zenodo.6010452) and [SLIIDERS-SLR](https://doi.org/10.5281/zenodo.6012027) datasets have been developed to serve this purpose, but any similarly formatted input dataset will work, if users wish to substitute, for example, alternative projections of economic growth besides the Shared Socioeconomic Pathways. To begin, however, we recommend that users obtain the two SLIIDERS datasets, which can be found at the linked DOIs.

#### Parameter specification

Parameters for the model run are defined in [params.json](./params.json). These can be modified to reflect alternative assumptions and/or model duration and timesteps. Description of each parameter can be found in the JSON file.

#### Surge lookup table creation

This step is not strictly necessary but provides dramatic performance increases. Rather than calculating the damages from extreme sea levels for all adaptation cases, for all segments, elevation slices, years, and scenarios, you may develop a lookup table that is used to build a 2D spline function to interpolate surge costs for any given combination of sea level height, and difference between storm surge height and retreat or protection height. `create_surge_lookup()` is used for this. See [create-surge-lookup-tables.ipynb](./notebooks/create-surge-lookup-tables.ipynb) for an example.

#### Calculation of costs for each adaptation case

In this step, costs for all adaptation options for all segments are calculated. Depending on the number of SLR trajectories modeled, segments may be run in smaller or larger batches to keep memory footprint to a reaasonable size. This parallelization is executed via [Dask](https://www.dask.org). `calc_costs()` is the function that executes these calculations. An example of usage can be seen in [run-pyCIAM-slrquantiles.ipynb](./notebooks/run-pyCIAM-slrquantiles.ipynb).

#### Calculation of the optimal adaptation choice

In this step, we find the optimal adaptation choice for each segment. `select_optimal_case()` is the function that executes this calculation. An example of usage can also be seen in [run-pyCIAM-slrquantiles.ipynb](./notebooks/run-pyCIAM-slrquantiles.ipynb).

## Reliance on Dask

**IMPORTANT**: The notebook-based workflow provided in this repository serves as a set of examples in addition to replicating the analysis in Depsky et al. 2022. It assumes that parallel task execution occurs via [Dask](https://dask.org), and in particular via a [dask-gateway](https://gateway.dask.org/) server. Users wishing to run these notebooks must be familiar with Dask and must modify the cluster construction code within each notebook to suit their resources. For example, one could instantiate a `LocalCluster` via [distributed](https://distributed.dask.org) to replace the `GatewayCluster` that is instantiated within the provided notebooks.

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
