History
=======

v1.2.0
------
* Point `data-acquisition.ipynb` to updated Zenodo deposit that fixes the dtype of `subsets` variable in `diaz2016_inputs_raw.zarr.zip` to be bool rather than int8
* Variable name bugfix in `data-acquisition.ipynb`
* Add netcdf versions of SLIIDERS and the pyCIAM results to `upload-zenodo.ipynb`
* Update results in Zenodo record to use SLIIDERS v1.2
  
v1.1.2
------
* Update zenodo-upload.ipynb to include packages
* Update readme to emphasize environment.yml

v1.1.1
------
* Update package dependencies

v1.1.0
------
* Use general Zenodo DOI numbers referencing latest version of each deposit
* Addition of AR6 and Sweet scenarios
* Addition of `execute_pyciam` wrapper function
* Updates to SLIIDERS inputs based on reviewer comments
* General repo hygiene
* Additional/updated figures/tables/results in `post-processing/pyCIAM-results-figures.ipynb`

v1.0.2
------
* Add HISTORY.rst
* Bump patch number to align with Zenodo deposit update
  
v1.0.1
------
* Add docstrings for all public functions
* Update readme with additional step-by-step instructions to serve as user manual
* Add `optimize_case` function
* Refactor various functions to facilitate abstraction

v1.0.0
------
* Initial commit
