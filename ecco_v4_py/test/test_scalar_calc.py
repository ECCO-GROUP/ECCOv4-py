"""
Test routines for the vector calculations module
"""
from __future__ import division, print_function
import warnings
from pathlib import Path
import numpy as np
import xarray as xr
import pytest
from ecco_v4_py import scalar_calc, get_llc_grid

from .test_common import all_mds_datadirs, get_test_ds

def test_latitude_mask(get_test_ds):
    """run through lats, and ensure we're grabbing the closest
    "south-grid-cell-location" (whether that's in the x or y direction!)
    to each latitude value"""

    ds = get_test_ds

    grid = get_llc_grid(ds)
    wetC = ds['maskC'].isel(k=0)

    dLat = 0.5 # is this robust?
    nx = 90

    for lat in np.arange(-89,89,10):
        print('lat: ',lat)
        maskC = scalar_calc.get_latitude_mask(lat,ds['YC'],grid)

        maskC = maskC.where((maskC!=0) & wetC,0.)

        assert (ds['YC']-lat < dLat).where((maskC!=0)).all().values
