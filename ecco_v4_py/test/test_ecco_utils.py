
import warnings
from datetime import datetime
import numpy as np
import xarray as xr
import pytest
import ecco_v4_py

from .test_common import llc_mds_datadirs, get_test_ds

@pytest.mark.parametrize("mytype",['xda','nparr','list','single'])
def test_extract_dates(mytype):

    dints = [[1991,8,9,13,10,15],[1992,10,20,8,30,5]]


    dates = [datetime(year=x[0],month=x[1],day=x[2],
                      hour=x[3],minute=x[4],second=x[5]) for x in dints]
    dates = np.array(dates,dtype='datetime64[s]')
    dates = [np.datetime64(x) for x in dates]
    if mytype=='xda':
        dates = xr.DataArray(np.array(dates))
    elif mytype=='nparr':
        dates = np.array(dates)
    elif mytype=='single':
        dints=dints[0]
        dates = dates[0]

    test_out = ecco_v4_py.extract_yyyy_mm_dd_hh_mm_ss_from_datetime64(dates)
    for test,expected in zip(test_out,np.array(dints).T):
        print('test: ',test)
        print('exp: ',expected)
        test = test.values if mytype=='xda' else test

        assert np.all(test==expected)

def test_get_grid(get_test_ds):
    """make sure we can make a grid ... that's it"""
    grid = ecco_v4_py.get_llc_grid(get_test_ds)
