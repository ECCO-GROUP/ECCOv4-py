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

from .test_common import llc_mds_datadirs, get_test_vectors

def test_no_angles(get_test_vectors):
    """quick error handling test"""

    ds = get_test_vectors
    ds = ds.drop_vars(['CS','SN'])
    with pytest.raises(KeyError):
        vector_calc.UEVNfromUXVY(ds['U'],ds['V'],ds)

def test_optional_grid(get_test_vectors):
    """simple, make sure we can optionally provide grid..."""

    ds = get_test_vectors
    grid = get_llc_grid(ds)

    uX = xr.ones_like(ds['U'].isel(k=0)).load();
    vY = xr.ones_like(ds['V'].isel(k=0)).load();

    u1,v1 = vector_calc.UEVNfromUXVY(uX,vY,ds)
    u2,v2 = vector_calc.UEVNfromUXVY(uX,vY,ds,grid)
    assert (u1==u2).all()
    assert (v1==v2).all()

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

def test_latitude_masks(get_test_vectors):
    """run through lats, and ensure we're grabbing the closest
    "south-grid-cell-location" (whether that's in the x or y direction!)
    to each latitude value"""

    ds = get_test_vectors

    grid = get_llc_grid(ds)
    yW = grid.interp(ds['YC'],'X',boundary='fill')
    yS = grid.interp(ds['YC'],'Y',boundary='fill')
    wetW = ds['maskW'].isel(k=0)
    wetS = ds['maskS'].isel(k=0)

    dLat = 0.5 # is this robust?
    nx = 90

    for lat in np.arange(-89,89):
        print('lat: ',lat)
        maskW,maskS = vector_calc.get_latitude_masks(lat,ds['YC'],grid)

        maskW = maskW.where((maskW!=0) & wetW,0.)
        maskS = maskS.where((maskS!=0) & wetS,0.)

        assert not (maskW>0).sel(tile=slice(7,None)).any()
        assert not (maskS<0).sel(tile=slice(5)).any()

        assert (yW-lat < dLat).where(ds['maskW'].isel(k=0) & (maskW!=0)).all().values
        assert (yS-lat < dLat).where(ds['maskS'].isel(k=0) & (maskS!=0)).all().values
