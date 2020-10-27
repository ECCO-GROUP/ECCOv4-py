"""
Test routines for computing meridional transport
"""
import warnings
import numpy as np
import xarray as xr
import pytest
import ecco_v4_py

from .test_common import llc_mds_datadirs, get_test_ds, get_test_vectors
from .test_vector_calc import get_fake_vectors

@pytest.mark.parametrize("lats",[-20,0,10,np.array([-30,-15,20,45])])
@pytest.mark.parametrize("basin",[None,'atlExt','pacExt','indExt'])
@pytest.mark.parametrize("doFlip",[True,False])
def test_meridional_stf(get_test_vectors,lats,basin,doFlip):
    """compute a meridional streamfunction"""

    ds = get_test_vectors
    grid = ecco_v4_py.get_llc_grid(ds)

    ds['U'],ds['V'] = get_fake_vectors(ds['U'],ds['V'])
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

@pytest.mark.parametrize("name, pt1, pt2, maskW, maskS, expArr",
        [
            ("drakepassage",None,None,None,None,None),
            (None,[-173,65.5],[-164,65.5],None,None,None),
            (None,None,None,True,True,None),
            (None,None,None,None,None,TypeError),
            ("drakepassage",[-173,65.5],[-164,65.5],None,None,TypeError),
            ("drakepassage",None,None,True,True,TypeError),
            (None,[-173,65.5],[-164,65.5],True,True,TypeError),
            ("noname",None,None,None,None,TypeError)
        ])
@pytest.mark.parametrize("doFlip",[True,False])
def test_section_stf(get_test_vectors,name,pt1,pt2,maskW,maskS,expArr,doFlip):
    """compute streamfunction across section"""

    ds = get_test_vectors
    grid = ecco_v4_py.get_llc_grid(ds)

    ds['U'],ds['V'] = get_fake_vectors(ds['U'],ds['V'])
    ds = ds.rename({'U':'UVELMASS','V':'VVELMASS'})

    if maskW is not None and maskS is not None:
        if maskW and maskS:
            maskW,maskS = ecco_v4_py.vector_calc.get_latitude_masks(30,ds['YC'],grid)

    if expArr is None:
        trsp = ecco_v4_py.calc_section_stf(ds,
                        pt1=pt1,pt2=pt2,
                        maskW=maskW,maskS=maskS,
                        section_name=name,
                        doFlip=doFlip,grid=grid)

        maskW,maskS = ecco_v4_py.calc_section_trsp._parse_section_trsp_inputs(ds,
                        pt1=pt1,pt2=pt2,maskW=maskW,maskS=maskS,
                        section_name=name)

        trspx = (ds['drF']*ds['dyG']*np.abs(maskW)).where(ds['maskW']).sum(dim=['i_g','j','tile'])
        trspy = (ds['drF']*ds['dxG']*np.abs(maskS)).where(ds['maskS']).sum(dim=['i','j_g','tile'])

        test = trsp.psi_moc.reset_coords(drop=True)
        expected = (1e-6*(trspx+trspy)).reset_coords(drop=True)
        if doFlip:
            expected = expected.isel(k=slice(None,None,-1))
        expected=expected.cumsum(dim='k')
        if doFlip:
            expected = -1*expected.isel(k=slice(None,None,-1))
        xr.testing.assert_allclose(test,expected)

    else:
        with pytest.raises(expArr):
            trsp = ecco_v4_py.calc_section_stf(ds,
                            pt1=pt1,pt2=pt2,
                            maskW=maskW,maskS=maskS,
                            section_name=name,
                            doFlip=doFlip,grid=grid)

            maskW,maskS = ecco_v4_py.calc_section_trsp._parse_section_trsp_inputs(ds,
                            pt1=pt1,pt2=pt2,maskW=maskW,maskS=maskS,
                            section_name=name)

@pytest.mark.parametrize("myfunc, fld, myarg",
        [   (ecco_v4_py.calc_meridional_stf,"vol_trsp", {'lat_vals':10}),
            (ecco_v4_py.calc_section_stf,"vol_trsp",{'section_name':'drakepassage'})])
def test_separate_coords(get_test_vectors,myfunc,fld,myarg):
    ds = get_test_vectors
    grid = ecco_v4_py.get_llc_grid(ds)

    ds['U'],ds['V'] = get_fake_vectors(ds['U'],ds['V'])
    ds = ds.rename({'U':'UVELMASS','V':'VVELMASS'})

    myarg['grid']=grid
    expected = myfunc(ds,**myarg)
    coords = ds.coords.to_dataset().reset_coords()
    ds = ds.reset_coords(drop=True)

    test = myfunc(ds,coords=coords,**myarg)
    xr.test.assert_allclose(test[fld],expected[fld])
