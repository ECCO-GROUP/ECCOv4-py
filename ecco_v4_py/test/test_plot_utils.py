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

def get_test_array(is_xda=False,sequential_data=True):
    """define a numpy and xarray DataArray for testing,
    For divergent data, set sequential_data=False
    """

    # read in as numpy array
    if sequential_data:
        test_arr = ecco.read_llc_to_tiles(fdir=_DATA_DIR,
                                          fname=_TEST_FILES[0],
                                          less_output=True)
    else:
        test_arr = ecco.read_llc_to_tiles(fdir=_DATA_DIR,
                                          fname=_TEST_FILES[2],
                                          less_output=True,
                                          skip=2)

    if is_xda:
        test_arr = ecco.llc_tiles_to_xda(test_arr,var_type='c',less_output=True)

    return test_arr

# ----------------------------------------------------------------
# tests
# ----------------------------------------------------------------
@pytest.mark.parametrize("test_arr,cmap_expected",
    [(get_test_array(is_xda=True,sequential_data=False),'RdBu_r'),
     (get_test_array(is_xda=True,sequential_data=True),'viridis')])
def test_cmap_defaults(test_arr,cmap_expected):
    cmap_test,_ = assign_colormap(test_arr)
    assert cmap_test==cmap_expected


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

    assert isinstance(cmin,float) or isinstance(cmin,np.float32)
    assert isinstance(cmax,float) or isinstance(cmin,np.float32)
