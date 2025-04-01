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


## Updating on pypi.org

Note: Publishing a new 'release' will trigger an 'action' and publish the new release to pypi. The version number has to be different otherwise pypi will reject it and say 'version already exists'.

The instructions below outline how to manually push code changes to pypi.

1. Make sure your ```~/.pypirc``` file has entries for [pypi] and [testpypi] with properly-scoped api tokens
```
[distutils]
  index-servers =
    pypi
    ecco_v4_py 

[pypi]
  username = __token__
  password = YOUR-PYPI-TOKEN-HERE

[ecco_v4_py]
  repository = https://upload.pypi.org/legacy/
  username = __token__
  password = YOUR-PYPI-TOKEN-HERE (can be scoped for just the ecco_v4_py package)

[testpypi]
  username = __token__
  password = YOUR-TESTPYPI-TOKEN-HERE
```
2. Verify all code changes are up to date on github, including version number
3. Navigate to ECCOv4_py directory
4. Remove old "distribution" files by deleting the contents of the ```dist/``` directory 
5. Rebuild the ```dist/``` files
```
python3 setup.py sdist bdist_wheel
```
6. Push changes to pypi test platform: test.pypi.org 
```
twine upload --repository testpypi dist/*
```
7. Verify code updates are on test.pypi.org: https://test.pypi.org/project/ecco-v4-py/
8. Push changes to to pypi
```
twine upload dist/* --repository-url https://upload.pypi.org/legacy/
```
9. Verify code updates are on pypi.org: https://pypi.org/project/ecco-v4-py/



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
