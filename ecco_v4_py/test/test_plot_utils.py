"""
Test routines for the tile plotting
"""
from __future__ import division, print_function
import warnings
from pathlib import Path
import numpy as np
import pytest
import ecco_v4_py as ecco

from ecco_v4_py.plot_utils import assign_colormap

# Define bin directory for test reading
_PKG_DIR = Path(__file__).resolve().parent.parent.parent
_DATA_DIR = _PKG_DIR.joinpath('binary_data')

_TEST_FILES = ['basins.data', 'hFacC.data', 'state_3d_set1.0000000732.data']

def get_test_array(is_xda=False):
    """define a numpy and xarray DataArray for testing"""

    # read in as numpy array
    test_arr = ecco.read_llc_to_tiles(fdir=_DATA_DIR,
                                      fname=_TEST_FILES[0],
                                      less_output=True)

    if is_xda:
        test_arr = ecco.llc_tiles_to_xda(test_arr,var_type='c',less_output=True)

    return test_arr

# ----------------------------------------------------------------
# tests
# ----------------------------------------------------------------
@pytest.mark.parametrize("arr",[
        get_test_array(is_xda=False),
        get_test_array(is_xda=True)
    ])
def test_cmap_override(arr):
    """make sure if user provides cmap, it overrides default"""

    # run plotting routine
    cmap_expected = 'inferno'

    cmap_test, _ = assign_colormap(arr,user_cmap=cmap_expected)
    assert cmap_test == cmap_expected

@pytest.mark.parametrize("arr",[
        get_test_array(is_xda=False),
        get_test_array(is_xda=True)
    ])
def test_cminmax_dtype(arr):
    """make cmin/cmax are floats"""

    _, (cmin,cmax) = assign_colormap(arr,user_cmap=None)
        
    assert isinstance(cmin,float)
    assert isinstance(cmax,float)
