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
@pytest.mark.parametrize("basin",[None,'atlExt','pacExt','indExt'])
def test_vol_trsp(get_test_vectors,lats,basin):
    """compute a volume transport"""

    ds = get_test_vectors
    grid = ecco_v4_py.get_llc_grid(ds)

    ds['U'],ds['V'] = get_fake_vectors(ds['U'].load(),ds['V'].load())
    ds = ds.rename({'U':'UVELMASS','V':'VVELMASS'})

    trsp = ecco_v4_py.calc_meridional_vol_trsp(ds,lats,basin_name=basin,grid=grid)
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
        test = trsp.sel(lat=lat).vol_trsp_z.reset_coords(drop=True)
        expected = (1e-6*(trspx+trspy)).reset_coords(drop=True)
        xr.testing.assert_allclose(test,expected)

@pytest.mark.parametrize("lats",[-20,0,10,np.array([-30,-15,20,45])])
@pytest.mark.parametrize("basin",[None,'atlExt','pacExt','indExt'])
def test_heat_trsp(get_test_vectors,lats,basin):
    """compute heat transport"""

    ds = get_test_vectors
    grid = ecco_v4_py.get_llc_grid(ds)

    ds['U'],ds['V'] = get_fake_vectors(ds['U'].load(),ds['V'].load())
    ds = ds.rename({'U':'ADVx_TH','V':'ADVy_TH'})
    ds['DFxE_TH'] = ds['ADVx_TH'].copy()
    ds['DFyE_TH'] = ds['ADVy_TH'].copy()

    trsp = ecco_v4_py.calc_meridional_heat_trsp(ds,lats,basin_name=basin,grid=grid)
    if basin is not None:
        basinW = ecco_v4_py.get_basin_mask(basin,ds['maskW'])
        basinS = ecco_v4_py.get_basin_mask(basin,ds['maskS'])
    else:
        basinW = ds['maskW']
        basinS = ds['maskS']


    lats = [lats] if np.isscalar(lats) else lats
    for lat in lats:
        maskW,maskS = ecco_v4_py.vector_calc.get_latitude_masks(lat,ds['YC'],grid)

        trspx = (2*np.abs(maskW)).where(basinW).sum(dim=['i_g','j','tile'])
        trspy = (2*np.abs(maskS)).where(basinS).sum(dim=['i','j_g','tile'])
        test = trsp.sel(lat=lat).heat_trsp_z.reset_coords(drop=True)
        expected = (1e-15*1029*4000*(trspx+trspy)).reset_coords(drop=True)
        xr.testing.assert_allclose(test,expected)

@pytest.mark.parametrize("lats",[-20,0,10,np.array([-30,-15,20,45])])
@pytest.mark.parametrize("basin",[None,'atlExt','pacExt','indExt'])
def test_salt_trsp(get_test_vectors,lats,basin):
    """compute salt transport"""

    ds = get_test_vectors
    grid = ecco_v4_py.get_llc_grid(ds)

    ds['U'],ds['V'] = get_fake_vectors(ds['U'].load(),ds['V'].load())
    ds = ds.rename({'U':'ADVx_SLT','V':'ADVy_SLT'})
    ds['DFxE_SLT'] = ds['ADVx_SLT'].copy()
    ds['DFyE_SLT'] = ds['ADVy_SLT'].copy()

    trsp = ecco_v4_py.calc_meridional_salt_trsp(ds,lats,basin_name=basin,grid=grid)
    if basin is not None:
        basinW = ecco_v4_py.get_basin_mask(basin,ds['maskW'])
        basinS = ecco_v4_py.get_basin_mask(basin,ds['maskS'])
    else:
        basinW = ds['maskW']
        basinS = ds['maskS']


    lats = [lats] if np.isscalar(lats) else lats
    for lat in lats:
        maskW,maskS = ecco_v4_py.vector_calc.get_latitude_masks(lat,ds['YC'],grid)

        trspx = (2*np.abs(maskW)).where(basinW).sum(dim=['i_g','j','tile'])
        trspy = (2*np.abs(maskS)).where(basinS).sum(dim=['i','j_g','tile'])
        test = trsp.sel(lat=lat).salt_trsp_z.reset_coords(drop=True)
        expected = (1e-6*(trspx+trspy)).reset_coords(drop=True)
        xr.testing.assert_allclose(test,expected)

def get_fake_vectors(fldx,fldy):
    for t in fldx.tile.values:
        if t<6:
            valx = 1.
            valy = 1.
        elif t == 6:
            valx = 0.
            valy = 0.
        else:
            valx = -1.
            valy = 1.

        fldx.loc[{'tile':t}] = valx
        fldy.loc[{'tile':t}] = valy
    return fldx, fldy
