"""
Test routines for the vector calculations module
"""
from __future__ import division, print_function
import warnings
from pathlib import Path
import numpy as np
import xarray as xr
import pytest
from ecco_v4_py import vector_calc, get_llc_grid

from .test_common import all_mds_datadirs, get_test_ds

def test_no_angles(get_test_ds):
    """quick error handling test"""

    ds = get_test_ds
    ds = ds.drop_vars(['CS','SN'])
    with pytest.raises(KeyError):
        vector_calc.UEVNfromUXVY(ds['U'],ds['V'],ds)

def test_optional_grid(get_test_ds):
    """simple, make sure we can optionally provide grid..."""

    ds = get_test_ds
    grid = get_llc_grid(ds)

    uX = xr.ones_like(ds['U'].isel(k=0)).load();
    vY = xr.ones_like(ds['V'].isel(k=0)).load();

    u1,v1 = vector_calc.UEVNfromUXVY(uX,vY,ds)
    u2,v2 = vector_calc.UEVNfromUXVY(uX,vY,ds,grid)
    assert (u1==u2).all()
    assert (v1==v2).all()

def test_uevn_from_uxvy(get_test_ds):
    """make sure grid loc is correct... etc...
    test by feeding right combo of 1, -1's
    so that all velocities are positive 1's ... interp should preserve this
    in the lat/lon components"""

    ds = get_test_ds

    uX,vY = get_fake_vectors(ds['U'],ds['V'])
    uE,vN = vector_calc.UEVNfromUXVY(uX,vY,ds)

    assert set(('i','j')).issubset(uE.dims)
    assert set(('i','j')).issubset(vN.dims)

    # check the lat/lon tiles
    tilelist = [1,4,8,11] if len(uE.tile)==13 else [0,5]
    for t in tilelist:
        assert np.allclose(uE.where(ds.maskC,0.).sel(tile=t),
                           1.*ds.maskC.sel(tile=t),atol=1e-12)
        assert np.allclose(vN.where(ds.maskC,0.).sel(tile=t),
                           1.*ds.maskC.sel(tile=t),atol=1e-12)

def test_latitude_masks(get_test_ds):
    """run through lats, and ensure we're grabbing the closest
    "south-grid-cell-location" (whether that's in the x or y direction!)
    to each latitude value"""

    ds = get_test_ds

    grid = get_llc_grid(ds)
    yW = grid.interp(ds['YC'],'X',boundary='fill')
    yS = grid.interp(ds['YC'],'Y',boundary='fill')
    wetW = ds['maskW'].isel(k=0)
    wetS = ds['maskS'].isel(k=0)

    dLat = 0.5 # is this robust?
    nx = 90

    for lat in np.arange(-89,89,10):
        print('lat: ',lat)
        maskW,maskS = vector_calc.get_latitude_masks(lat,ds['YC'],grid)

        maskW = maskW.where((maskW!=0) & wetW,0.)
        maskS = maskS.where((maskS!=0) & wetS,0.)

        arctic = int((len(maskW.tile)-1)/2)
        assert not (maskW>0).sel(tile=slice(arctic+1,None)).any()
        assert not (maskS<0).sel(tile=slice(arctic-1)).any()

        assert (yW-lat < dLat).where(ds['maskW'].isel(k=0) & (maskW!=0)).all().values
        assert (yS-lat < dLat).where(ds['maskS'].isel(k=0) & (maskS!=0)).all().values

def get_fake_vectors(fldx,fldy):
    fldx.load();
    fldy.load();
    arctic = int((len(fldx.tile)-1)/2)
    for t in fldx.tile.values:
        if t<arctic:
            valx = 1.
            valy = 1.
        elif t == arctic:
            valx = 0.
            valy = 0.
        else:
            valx = -1.
            valy = 1.

        fldx.loc[{'tile':t}] = valx
        fldy.loc[{'tile':t}] = valy
    return fldx, fldy
