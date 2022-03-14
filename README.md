[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.6014086.svg)](https://doi.org/10.5281/zenodo.6014086)

# pyCIAM

A Python port, including numerous model updates, of the [Coastal Impacts and Adaptation Model](https://github.com/delavane/CIAM), as described in [Diaz 2016](https://link.springer.com/article/10.1007/s10584-016-1675-4#Sec13). This code accompanies Depsky et al. 2022 (in prep.). See the manuscript for further details on model functionality and structure.

`pyCIAM`, like `CIAM`, is a tool to estimate global economic impacts of sea level rise at fine resolution, accounting for adaptation and spatial variability and uncertainty in sea level rise. This model requires a number of socioeconomic and sea level rise inputs, organized by coastal "segment" and elevation. In Depsky et al. 2022, we develop the [SLIIDERS](https://github.com/ClimateImpactLab/sliiders) datasets -- [SLIIDERS-ECON](https://doi.org/10.5281/zenodo.6010452) and [SLIIDERS-SLR](https://doi.org/10.5281/zenodo.6012027) -- to serve this purpose. However, users may wish to alter and/or replace these datasets for their own purposes, especially if newer input data (used to generate `SLIIDERS`) becomes available.

Model outputs consistent with Depsky et al. 2022 are available on Zenodo, with DOI [10.5281/zenodo.6014086](https://doi.org/10.5281/zenodo.6014086)

At present, this repository contains both the `pyCIAM` package, along with a Jupyter Notebook-based workflow used to create the outputs of Depsky et al. 2022. In the near future, we will likely extract the workflow into its own repository to isolate the more broadly useful package within this one.

## Status
This code was developed for Depsky et al. 2022, but our intention is for broader use. If and when that broader usage develops, the package will continue to become more formalized, well-tested, and documented. In the meantime, please pardon the dust as we continue to construct the package, and don't hesitate to reach out by filing an issue if you have any questions about usage.

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

**IMPORTANT**: The notebook-based workflow currently assumes that parallel task execution occurs via [Dask](https://dask.org), and in particular via a [dask-gateway](https://gateway.dask.org/) server. Users must be familiar with Dask and must modify the cluster construction code within each notebook to suit their resources. For example, one could instantiate a `LocalCluster` via [distributed](https://distributed.dask.org) to replace the `GatewayCluster` that is instantiated within the provided notebooks.

Parameters for the model run are defined in [params.json](./params.json). These can be modified to reflect alternative assumptions and/or model duration and timesteps.

More complete model documentation is forthcoming.

## Support
Please file an issue for any problems you encounter

## Contributing
We encourage community contributions and hope that the functionality of `pyCIAM` will grow as a result. At the moment, we have no contribution template. Please fork the project and file a Merge Request to propose your addition. Clearly define the contribution that the Merge Request is making and, when any issues have been resolved, we will merge the new code.

## Testing
A more complete testing suite is in development.

## Authors
The original authors of this code include:
- Daniel Allen
- Ian Bolliger
- Junho Choi
- Nicholas Depsky

## License
This code is licensed under the [MIT License](./LICENSE)