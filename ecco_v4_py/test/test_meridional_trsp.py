"""
Test routines for computing meridional transport
"""
import warnings
import numpy as np
import xarray as xr
import pytest
import ecco_v4_py

from .test_common import llc_mds_datadirs, get_test_ds, get_test_vectors

@pytest.mark.parametrize("lats",[-20,0,10,np.array([-30,-15,20,45])])
def test_trsp_ds(get_test_ds,lats):
    """stupid simple"""
    exp = get_test_ds
    test = ecco_v4_py.calc_meridional_trsp._initialize_trsp_data_array(exp,lats)
    assert np.all(test.lat.values==lats)
    assert np.all(test.time==exp.time)
    assert np.all(test.k == exp.k)

@pytest.mark.parametrize("lats",[-20,0,10,np.array([-30,-15,20,45])])
def test_vol_trsp(get_test_vectors,lats):
    """compute a volume transport"""

    ds = get_test_vectors
    grid = ecco_v4_py.get_llc_grid(ds)

    ds['U'],ds['V'] = get_fake_vectors(ds['U'].load(),ds['V'].load())
    ds = ds.rename({'U':'UVELMASS','V':'VVELMASS'})

    trsp = ecco_v4_py.calc_meridional_vol_trsp(ds,lats,grid=grid)



def get_fake_vectors(fldx,fldy):
    for t in fldx.tile.values:
        if t<6:
            valx = 1
            valy = 1
        elif t == 6:
            valx = 0
            valy = 0
        else:
            valx = -1
            valy = 1

        fldx.loc[{'tile':t}] = valx
        fldy.loc[{'tile':t}] = valy
    return fldx, fldy
