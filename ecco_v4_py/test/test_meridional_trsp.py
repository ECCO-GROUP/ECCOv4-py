"""
Test routines for computing meridional transport
"""
import warnings
import numpy as np
import xarray as xr
import pytest
import ecco_v4_py

from .test_common import all_mds_datadirs, get_test_ds
from .test_vector_calc import get_fake_vectors

@pytest.mark.parametrize("lats",[-20,0,10,np.array([-30,-15,20,45])])
def test_trsp_ds(get_test_ds,lats):
    """stupid simple"""
    exp = get_test_ds
    test = ecco_v4_py.calc_meridional_trsp._initialize_trsp_data_array(exp,lats)
    assert np.all(test.lat.values==lats)
    assert np.all(test.time==exp.time)
    assert np.all(test.k == exp.k)

@pytest.mark.parametrize("myfunc, tfld, xflds, yflds, factor",
        [   (ecco_v4_py.calc_meridional_vol_trsp,"vol_trsp_z",
                ['UVELMASS'],['VVELMASS'], 1e-6),
            (ecco_v4_py.calc_meridional_heat_trsp,"heat_trsp_z",
                ['ADVx_TH','DFxE_TH'],['ADVy_TH','DFyE_TH'],1e-15*1029*4000),
            (ecco_v4_py.calc_meridional_salt_trsp,"salt_trsp_z",
                ['ADVx_SLT','DFxE_SLT'],['ADVy_SLT','DFyE_SLT'],1e-6)])
@pytest.mark.parametrize("lats",[0,np.array([-20,30,45])])
@pytest.mark.parametrize("basin",[None,'atlExt','pacExt','indExt'])
def test_meridional_trsp(get_test_ds,myfunc,tfld,xflds,yflds,factor,lats,basin):
    """compute a transport"""

    ds = get_test_ds
    grid = ecco_v4_py.get_llc_grid(ds)

    ds['U'],ds['V'] = get_fake_vectors(ds['U'],ds['V'])
    for fx,fy in zip(xflds,yflds):
        ds[fx] = ds['U'].copy()
        ds[fy] = ds['V'].copy()

    if basin is None or len(ds.tile)==13:
        trsp = myfunc(ds,lats,basin_name=basin,grid=grid)
        if basin is not None:
            basinW = ecco_v4_py.get_basin_mask(basin,ds['maskW'].isel(k=0))
            basinS = ecco_v4_py.get_basin_mask(basin,ds['maskS'].isel(k=0))
        else:
            basinW = ds['maskW'].isel(k=0)
            basinS = ds['maskS'].isel(k=0)

        lats = [lats] if np.isscalar(lats) else lats
        expx = (ds['drF']*ds['dyG']).copy() if tfld == 'vol_trsp_z' else 2.*xr.ones_like(ds['hFacW'])
        expy = (ds['drF']*ds['dxG']).copy() if tfld == 'vol_trsp_z' else 2.*xr.ones_like(ds['hFacS'])
        for lat in lats:
            maskW,maskS = ecco_v4_py.vector_calc.get_latitude_masks(lat,ds['YC'],grid)

            trspx = (expx*np.abs(maskW)).where(basinW).sum(dim=['i_g','j','tile'])
            trspy = (expy*np.abs(maskS)).where(basinS).sum(dim=['i','j_g','tile'])

            test = trsp.sel(lat=lat)[tfld].squeeze().reset_coords(drop=True)
            expected = (factor*(trspx+trspy)).reset_coords(drop=True)
            xr.testing.assert_allclose(test,expected)
    else:
        with pytest.raises(NotImplementedError):
            trsp = myfunc(ds,lats,basin_name=basin,grid=grid)

@pytest.mark.parametrize("myfunc, fld, xflds, yflds",
        [   (ecco_v4_py.calc_meridional_vol_trsp,"vol_trsp",
                ['UVELMASS'],['VVELMASS']),
            (ecco_v4_py.calc_meridional_heat_trsp,"heat_trsp",
                ['ADVx_TH','DFxE_TH'],['ADVy_TH','DFyE_TH']),
            (ecco_v4_py.calc_meridional_salt_trsp,"salt_trsp",
                ['ADVx_SLT','DFxE_SLT'],['ADVy_SLT','DFyE_SLT'])])
@pytest.mark.parametrize("lat",[10]) # more is unnecessary
def test_separate_coords(get_test_ds,myfunc,fld,xflds,yflds,lat):
    ds = get_test_ds
    grid = ecco_v4_py.get_llc_grid(ds)

    ds['U'],ds['V'] = get_fake_vectors(ds['U'],ds['V'])
    for fx,fy in zip(xflds,yflds):
        ds[fx] = ds['U'].copy()
        ds[fy] = ds['V'].copy()

    expected = myfunc(ds,lat,grid=grid)
    coords = ds.coords.to_dataset().reset_coords()
    ds = ds.reset_coords(drop=True)

    test = myfunc(ds,lat,coords=coords,grid=grid)
    xr.testing.assert_equal(test[fld].reset_coords(drop=True),
                            expected[fld].reset_coords(drop=True))
