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

@pytest.mark.parametrize("lats",[-20,0,10,np.array([-30,-15,20,45])])
@pytest.mark.parametrize("basin",[None,'atlExt','pacExt','indExt'])
@pytest.mark.parametrize("doFlip",[True,False])
def test_meridional_stf(get_test_vectors,lats,basin,doFlip):
    """compute a meridional streamfunction"""

    ds = get_test_vectors
    grid = ecco_v4_py.get_llc_grid(ds)

    ds['U'],ds['V'] = get_fake_vectors(ds['U'].load(),ds['V'].load())
    ds = ds.rename({'U':'UVELMASS','V':'VVELMASS'})

    trsp = ecco_v4_py.calc_meridional_stf(ds,lats,doFlip=doFlip,basin_name=basin,grid=grid)
    if basin is not None:
        basinW = ecco_v4_py.get_basin_mask(basin,ds['maskW'])
        basinS = ecco_v4_py.get_basin_mask(basin,ds['maskS'])
    else:
        basinW = ds['maskW']
        basinS = ds['maskS']


    lats = [lats] if np.isscalar(lats) else lats
    for lat in lats:
        maskW,maskS = ecco_v4_py.vector_calc.get_latitude_masks(lat,ds['YC'],grid)

        trspx = (ds['drF']*ds['dyG']*np.abs(maskW)).where(basinW).sum(dim=['i_g','j','tile'])
        trspy = (ds['drF']*ds['dxG']*np.abs(maskS)).where(basinS).sum(dim=['i','j_g','tile'])
        test = trsp.sel(lat=lat).psi_moc.reset_coords(drop=True)
        expected = (1e-6*(trspx+trspy)).reset_coords(drop=True)
        if doFlip:
            expected = expected.isel(k=slice(None,None,-1))
        expected=expected.cumsum(dim='k')
        if doFlip:
            expected = -1*expected.isel(k=slice(None,None,-1))
        xr.testing.assert_allclose(test,expected)
