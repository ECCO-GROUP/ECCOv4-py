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

def test_trsp_ds(get_test_ds):
    """stupid simple"""
    exp = get_test_ds
    test = ecco_v4_py.calc_section_trsp._initialize_section_trsp_data_array(exp)
    assert np.all(test.time==exp.time)
    assert np.all(test.k == exp.k)

@pytest.mark.parametrize("myfunc, tfld, xflds, yflds, factor",
        [   (ecco_v4_py.calc_section_vol_trsp,"vol_trsp_z",
                ['UVELMASS'],['VVELMASS'], 1e-6),
            (ecco_v4_py.calc_section_heat_trsp,"heat_trsp_z",
                ['ADVx_TH','DFxE_TH'],['ADVy_TH','DFyE_TH'],1029*4000*1e-15),
            (ecco_v4_py.calc_section_salt_trsp,"salt_trsp_z",
                ['ADVx_SLT','DFxE_SLT'],['ADVy_SLT','DFyE_SLT'],1e-6)])
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
def test_section_trsp(get_test_ds,myfunc,tfld,xflds,yflds,factor,args,mask,error):
    """compute a volume transport,
    within the lat/lon portion of the domain"""

    ds = get_test_ds
    grid = ecco_v4_py.get_llc_grid(ds)

    ds['U'],ds['V'] = get_fake_vectors(ds['U'],ds['V'])
    for fx,fy in zip(xflds,yflds):
        ds[fx] = ds['U'].copy()
        ds[fy] = ds['V'].copy()

    myargs = args.copy()
    if mask:
        myargs['maskW'],myargs['maskS'] = ecco_v4_py.vector_calc.get_latitude_masks(30,ds['YC'],grid)
    else:
        myargs['maskW']=None
        myargs['maskS']=None

    if error is None:
        trsp = myfunc(ds,grid=grid,**myargs)

        maskW,maskS = ecco_v4_py.calc_section_trsp._parse_section_trsp_inputs(ds,
                        grid=grid,**myargs)

        expx = (ds['drF']*ds['dyG']).copy() if tfld == 'vol_trsp_z' else 2.*xr.ones_like(ds['hFacW'])
        expy = (ds['drF']*ds['dxG']).copy() if tfld == 'vol_trsp_z' else 2.*xr.ones_like(ds['hFacS'])
        trspx = (expx*np.abs(maskW)).where(ds['maskW']).sum(dim=['i_g','j','tile'])
        trspy = (expy*np.abs(maskS)).where(ds['maskS']).sum(dim=['i','j_g','tile'])

        test = trsp[tfld].squeeze().reset_coords(drop=True)
        expected = (factor*(trspx+trspy)).reset_coords(drop=True)
        xr.testing.assert_equal(test,expected)

    else:
        with pytest.raises(error):
            trsp = myfunc(ds,**myargs)

@pytest.mark.parametrize("myfunc, tfld, xflds, yflds",
        [   (ecco_v4_py.calc_section_vol_trsp,"vol_trsp",
                ['UVELMASS'],['VVELMASS']),
            (ecco_v4_py.calc_section_heat_trsp,"heat_trsp",
                ['ADVx_TH','DFxE_TH'],['ADVy_TH','DFyE_TH']),
            (ecco_v4_py.calc_section_salt_trsp,"salt_trsp",
                ['ADVx_SLT','DFxE_SLT'],['ADVy_SLT','DFyE_SLT'])])
@pytest.mark.parametrize("section_name",["beringstrait"]) # more is unnecessary
def test_separate_coords(get_test_ds,myfunc,tfld,xflds,yflds,section_name):
    ds = get_test_ds
    grid = ecco_v4_py.get_llc_grid(ds)

    ds['U'],ds['V'] = get_fake_vectors(ds['U'],ds['V'])
    for fx,fy in zip(xflds,yflds):
        ds[fx] = ds['U']
        ds[fy] = ds['V']

    expected = myfunc(ds,section_name=section_name,grid=grid)
    coords = ds.coords.to_dataset().reset_coords()
    ds = ds.reset_coords(drop=True)

    test = myfunc(ds,section_name=section_name,coords=coords,grid=grid)
    xr.testing.assert_equal(test[tfld].reset_coords(drop=True),
                            expected[tfld].reset_coords(drop=True))

@pytest.mark.parametrize("section_name",["beringstrait"])
def test_trsp_masking(get_test_ds,section_name):
    """make sure internal masking is legit"""

    ds = get_test_ds
    grid = ecco_v4_py.get_llc_grid(ds)

    ds['U'],ds['V'] = get_fake_vectors(ds['U'],ds['V'])
    ds['U'] = ds['U'].where(ds['maskW'],0.)
    ds['V'] = ds['V'].where(ds['maskS'],0.)

    pt1,pt2 = ecco_v4_py.get_section_endpoints(section_name)
    _, maskW,maskS = ecco_v4_py.get_section_line_masks(pt1,pt2,ds)

    expected = ecco_v4_py.section_trsp_at_depth(ds['U'],ds['V'],maskW,maskS,ds)

    ds = ds.drop_vars(['maskW','maskS'])
    test = ecco_v4_py.section_trsp_at_depth(ds['U'],ds['V'],maskW,maskS)

    xr.testing.assert_equal(test['trsp_z'].reset_coords(drop=True),
                            expected['trsp_z'].reset_coords(drop=True))
