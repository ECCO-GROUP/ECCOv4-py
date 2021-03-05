#!/bin/bash

# note pytest package must be installed

# test on 1 cpu
#py.test . -v --cov=ecco_v4_py --cov-config .coveragerc --ignore=ecco_v4_py/test/test_generate_ecco_netcdf_product.py

# test with 12 cpus
# must have pytest-xdist installed
py.test . -v -n 12 --cov=ecco_v4_py --cov-config .coveragerc --ignore=ecco_v4_py/test/test_generate_ecco_netcdf_product.py

rm -fr /tmp/pytest*
