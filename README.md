# clear_sky_detection

Python version of the Bright-Sun clear sky detection algorithm: https://github.com/JamieMBright/csd-library

This algorithm requires an clear sky first guess.
Here, the [python-pvlib](https://pvlib-python.readthedocs.io/en/stable/)
package is used to do this, so its required to run one of the examples (see example_pvlib.py).

Alternatively a clear sky model can be selected from 
https://github.com/jonas-witthuhn/clear-sky-models,
which may require additional data (e.g. albedo, aerosol, ozone...). But there are also clear sky models expecting nothing more than sza and date (e.g. ashrae) which also work fine, since the Bright-Sun algorithm optimizes these first guess values (see example_csm.py).

# Requirements
 - numpy
 - scipy
 
 for examples:
 - pandas
 - matplotlib
 - xarray
 - pvlib-python (example_pvlib)
 - rpy2 (example_csm)
 - netcdf4 (example_csm)
 - ` pip install git+https://github.com/jonas-witthuhn/clear-sky-models.git#egg=clear_sky_models `  (example_csm)
 see conda environmets environmet_pvlib.yml and environmet_csm.yml
 
 # Install
 
 ` pip install git+https://github.com/jonas-witthuhn/clear_sky_detection.git#egg=clear_sky_detection
 `
