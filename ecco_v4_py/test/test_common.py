"""
Helper functions for all tests
"""
from xmitgcm.test.test_xmitgcm_common import llc_mds_datadirs
import ecco_v4_py as ecco

def get_test_array_2d(llc_mds_datadirs,is_xda=True):
    """download, unzip and return 2D field"""

    dirname, expected = llc_mds_datadirs

    # read in array
    ds = ecco.load_ecco_vars_from_mds(dirname,
            model_time_steps_to_load=expected['test_iternum'],
            mds_files=expected['diagnostics'][0])

    if 'time' in ds.dims:
        ds = ds.isel(time=0)

    xda = ds['ETAN']

    if is_xda:
        return xda
    else:
        return xda.values
