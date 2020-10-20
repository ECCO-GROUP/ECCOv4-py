"""
Test routines for the vector calculations module
"""
from __future__ import division, print_function
import warnings
from pathlib import Path
import numpy as np
import xarray as xr
import pytest
from ecco_v4_py import vector_calc

from .test_common import llc_mds_datadirs, get_test_vectors

def test_no_angles(get_test_vectors):
    """quick error handling test"""

    ds = get_test_vectors
    ds = ds.drop(['CS','SN'])
    with pytest.raises(KeyError):
        vector_calc.UEVNfromUXVY(ds['U'],ds['V'],ds)

def test_uevn_from_uxvy(get_test_vectors):
    """make sure grid loc is correct... etc...
    test by feeding right combo of 1, -1's
    so that all velocities are positive 1's ... interp should preserve this
    in the lat/lon components"""

    ds = get_test_vectors

    uX = xr.ones_like(ds['U'].isel(k=0)).load();
    vY = xr.ones_like(ds['V'].isel(k=0)).load();
    for t in ds.tile.values:
        if t<6:
            valx = 1
            valy = 1
        elif t == 6:
            valx = 0
            valy = 0
        else:
            valx = -1
            valy = 1

        uX.loc[{'tile':t}] = valx
        vY.loc[{'tile':t}] = valy


    uE,vN = vector_calc.UEVNfromUXVY(uX,vY,ds)

    assert set(('i','j')).issubset(uE.dims)
    assert set(('j','j')).issubset(vN.dims)

    # check the lat/lon tiles
    for t in [1,4,8,11]:
        assert np.allclose(uE.sel(tile=t),1,atol=1e-15)
        assert np.allclose(vN.sel(tile=t),1,atol=1e-15)
