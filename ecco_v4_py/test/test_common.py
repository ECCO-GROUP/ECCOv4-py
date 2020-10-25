"""
Helper functions for all tests
"""
import pytest
from xmitgcm.test.test_xmitgcm_common import llc_mds_datadirs
import ecco_v4_py as ecco

@pytest.fixture(scope='module')
def get_test_ds(llc_mds_datadirs):

    dirname, expected = llc_mds_datadirs

    # read in array
    ds = ecco.load_ecco_vars_from_mds(dirname,
            model_time_steps_to_load=expected['test_iternum'],
            mds_files=expected['diagnostics'][0])
    return ds

@pytest.fixture(scope='module')
def get_test_array_2d(llc_mds_datadirs):
    """download, unzip and return 2D field"""

    dirname, expected = llc_mds_datadirs

    # read in array
    ds = ecco.load_ecco_vars_from_mds(dirname,
            model_time_steps_to_load=expected['test_iternum'],
            mds_files=expected['diagnostics'][0])

    xda = ds['ETAN']

    if 'time' in xda.dims:
        xda = xda.isel(time=0)
    return xda

@pytest.fixture(scope='module')
def get_test_vectors(llc_mds_datadirs):
    """download, unzip and return zonal/meridional velocity in dataset"""

    dirname, expected = llc_mds_datadirs

    # read in array
    ds = ecco.load_ecco_vars_from_mds(dirname,
            model_time_steps_to_load=expected['test_iternum'],
            mds_files=['U','V'])

    if 'time' in ds.dims:
        ds = ds.isel(time=-1)

    return ds


