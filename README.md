## Synopsis

ecco_v4_py is a Python package that includes tools for loading and manipulating the ECCO v4 ocean and sea-ice state estimate (http://ecco-group.org)

Extensive documentation is provided on our readthedocs page: 
http://ecco-v4-python-tutorial.readthedocs.io/index.html#

## Installation

Installation instructions can be found here!

https://ecco-v4-python-tutorial.readthedocs.io/Installing_Python_and_Python_Packages.html


## Contributors

If you would like to contribute, consider forking this repository and making pull requests via git!

## Support 

contact ecco-support@mit.edu or Ian.Fenty at jpl.nasa.gov

## License

MIT License


## Note on version numbers

ecco_v4_py uses the 'semantic versioning' scheme described here:

https://packaging.python.org/guides/distributing-packages-using-setuptools/#semantic-versioning-preferred

The essence of semantic versioning is a 3-part MAJOR.MINOR.MAINTENANCE numbering scheme:

MAJOR version when they make incompatible API changes,

MINOR version when they add functionality in a backwards-compatible manner, and

MAINTENANCE version when they make backwards-compatible bug fixes.

## Note on testing with `pytest`

(credit to Tim Smith)


You can run the tests locally with the pytest package, which is available through conda-forge. With that installed, you can navigate to ECCOv4-py/ecco_v4_py/test and either:

Run all the tests exactly as they are on travis (this takes a while, like 12 minutes!):

```
py.test . -v --cov=ecco_v4_py --cov-config .coveragerc --ignore=ecco_v4_py/test/test_generate_ecco_netcdf_product.py
```

Or you can run any individual module e.g. to run the few tests in ecco_utils:

```
py.test test_ecco_utils.py

```

(and you can add any of the -v or whatever flags you want). 
