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

_section='floridastrait'
_pt1=[-81,28]
_pt2=[-79,22]

@pytest.mark.parametrize("lats",[0,np.array([-20,30,45])])
@pytest.mark.parametrize("basin",[None,'atlExt','pacExt','indExt'])
@pytest.mark.parametrize("doFlip",[True,False])
def test_meridional_stf(get_test_ds,lats,basin,doFlip):
    """compute a meridional streamfunction"""

    ds = get_test_ds
    grid = ecco_v4_py.get_llc_grid(ds)

    ds['U'],ds['V'] = get_fake_vectors(ds['U'],ds['V'])
    ds = ds.rename({'U':'UVELMASS','V':'VVELMASS'})

    if basin is None or len(ds.tile)==13:
        trsp = ecco_v4_py.calc_meridional_stf(ds,lats,doFlip=doFlip,basin_name=basin,grid=grid)

        basinW = ds['maskW']
        basinS = ds['maskS']
        if basin is not None:
            basinW = ecco_v4_py.get_basin_mask(basin,basinW)
            basinS = ecco_v4_py.get_basin_mask(basin,basinS)

        lats = [lats] if np.isscalar(lats) else lats
        for lat in lats:
            maskW,maskS = ecco_v4_py.vector_calc.get_latitude_masks(lat,ds['YC'],grid)

            trspx = (ds['drF']*ds['dyG']*np.abs(maskW)).where(basinW).sum(dim=['i_g','j','tile'])
            trspy = (ds['drF']*ds['dxG']*np.abs(maskS)).where(basinS).sum(dim=['i','j_g','tile'])
            test = trsp.sel(lat=lat).psi_moc.squeeze().reset_coords(drop=True)
            expected = (1e-6*(trspx+trspy)).reset_coords(drop=True)
            if doFlip:
                expected = expected.isel(k=slice(None,None,-1))
            expected=expected.cumsum(dim='k')
            if doFlip:
                expected = -1*expected.isel(k=slice(None,None,-1))
            xr.testing.assert_allclose(test,expected)
    else:
        with pytest.raises(NotImplementedError):
            trsp = ecco_v4_py.calc_meridional_stf(ds,lats,doFlip=doFlip,basin_name=basin,grid=grid)

@pytest.mark.parametrize("args, mask, error",
        [
            ({'section_name':_section,'pt1':None,'pt2':None},False,None),
            ({'section_name':None,    'pt1':_pt1,'pt2':_pt2},False,None),
            ({'section_name':None,    'pt1':None,'pt2':None},True ,None),
            ({'section_name':None,    'pt1':None,'pt2':None},False,TypeError),
            ({'section_name':_section,'pt1':_pt1,'pt2':_pt2},False,TypeError),
            ({'section_name':_section,'pt1':None,'pt2':None},True ,TypeError),
            ({'section_name':"noname",'pt1':None,'pt2':None},False,TypeError),
        ])
@pytest.mark.parametrize("doFlip",[True,False])
def test_section_stf(get_test_ds,args,mask,error,doFlip):
    """compute streamfunction across section"""

    ds = get_test_ds
    grid = ecco_v4_py.get_llc_grid(ds)

    ds['U'],ds['V'] = get_fake_vectors(ds['U'],ds['V'])
    ds = ds.rename({'U':'UVELMASS','V':'VVELMASS'})

    myargs = args.copy()
    if mask:
        myargs['maskW'],myargs['maskS'] = ecco_v4_py.vector_calc.get_latitude_masks(30,ds['YC'],grid)
    else:
        myargs['maskW']=None
        myargs['maskS']=None

    if error is None:
        trsp = ecco_v4_py.calc_section_stf(ds,doFlip=doFlip,grid=grid,**myargs)

        maskW,maskS = ecco_v4_py.calc_section_trsp._parse_section_trsp_inputs(ds,**myargs)

        trspx = (ds['drF']*ds['dyG']*np.abs(maskW)).where(ds['maskW']).sum(dim=['i_g','j','tile'])
        trspy = (ds['drF']*ds['dxG']*np.abs(maskS)).where(ds['maskS']).sum(dim=['i','j_g','tile'])

        test = trsp.psi_moc.squeeze().reset_coords(drop=True)
        expected = (1e-6*(trspx+trspy)).reset_coords(drop=True)
        if doFlip:
            expected = expected.isel(k=slice(None,None,-1))
        expected=expected.cumsum(dim='k')
        if doFlip:
            expected = -1*expected.isel(k=slice(None,None,-1))
        xr.testing.assert_allclose(test,expected)

    else:
        with pytest.raises(error):
            trsp = ecco_v4_py.calc_section_stf(ds,**myargs)

@pytest.mark.parametrize("myfunc, myarg",
        [   (ecco_v4_py.calc_meridional_stf, {'lat_vals':10}),
            (ecco_v4_py.calc_section_stf,{'section_name':_section})])
def test_separate_coords(get_test_ds,myfunc,myarg):
    ds = get_test_ds
    grid = ecco_v4_py.get_llc_grid(ds)

    ds['U'],ds['V'] = get_fake_vectors(ds['U'],ds['V'])
    ds = ds.rename({'U':'UVELMASS','V':'VVELMASS'})

    myarg['grid']=grid
    expected = myfunc(ds,**myarg)
    coords = ds.coords.to_dataset().reset_coords()
    ds = ds.reset_coords(drop=True)

    test = myfunc(ds,coords=coords,**myarg)
    xr.testing.assert_allclose(test['psi_moc'].reset_coords(drop=True),
                               expected['psi_moc'].reset_coords(drop=True))
