"""
Test routines for computing meridional transport
"""
import warnings
import numpy as np
import xarray as xr
import pytest
import ecco_v4_py

from .test_common import llc_mds_datadirs, get_test_ds, get_test_vectors
from .test_meridional_trsp import get_fake_vectors

def test_trsp_ds(get_test_ds):
    """stupid simple"""
    exp = get_test_ds
    test = ecco_v4_py.calc_section_trsp._initialize_section_trsp_data_array(exp)
    assert np.all(test.time==exp.time)
    assert np.all(test.k == exp.k)

@pytest.mark.parametrize("name",["drakepassage"])
def test_vol_trsp(get_test_vectors,name):
    """compute a volume transport"""

    ds = get_test_vectors
    grid = ecco_v4_py.get_llc_grid(ds)

    ds['U'],ds['V'] = get_fake_vectors(ds['U'].load(),ds['V'].load())
    ds = ds.rename({'U':'UVELMASS','V':'VVELMASS'})

    trsp = ecco_v4_py.calc_section_vol_trsp(ds,section_name=name,grid=grid)

    maskW,maskS = ecco_v4_py.calc_section_trsp._parse_section_trsp_inputs(ds,
                    pt1=None,pt2=None,maskW=None,maskS=None,
                    section_name=name)

    trspx = (ds['drF']*ds['dyG']*np.abs(maskW)).where(ds['maskW']).sum(dim=['i_g','j','tile'])
    trspy = (ds['drF']*ds['dxG']*np.abs(maskS)).where(ds['maskS']).sum(dim=['i','j_g','tile'])
    test = trsp.vol_trsp_z.reset_coords(drop=True)
    expected = (1e-6*(trspx+trspy)).reset_coords(drop=True)
    xr.testing.assert_allclose(test,expected)
