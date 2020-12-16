#!/bin/bash

# note pytest package must be installed
py.test . -v --cov=ecco_v4_py --cov-config .coveragerc --ignore=ecco_v4_py/test/test_generate_ecco_netcdf_product.py
