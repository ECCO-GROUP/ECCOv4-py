"""
Test routines for the tile plotting
"""
from __future__ import division, print_function
import warnings
from pathlib import Path
import numpy as np
import pytest
import ecco_v4_py as ecco

from .test_common import llc_mds_datadirs,get_test_array_2d
from ecco_v4_py.plot_utils import assign_colormap

@pytest.mark.parametrize("is_xda",[True,False])
@pytest.mark.parametrize("sequential_data, cmap_expected",
    [(True,'viridis'),
     (False,'RdBu_r'),
     (False,'inferno')])
def test_cmap(get_test_array_2d,is_xda,sequential_data,cmap_expected):

    test_arr = get_test_array_2d
    test_arr = test_arr if is_xda else test_arr.values

    if sequential_data:
        test_arr = np.abs(test_arr)

    if set(cmap_expected).issubset(set(['viridis','RdBu_r'])):
        cmap_test,_ = assign_colormap(test_arr)
    else:
        cmap_test,_ = assign_colormap(test_arr,cmap_expected)

    assert cmap_test==cmap_expected

@pytest.mark.parametrize("is_xda",[True,False])
def test_cminmax_dtype(get_test_array_2d,is_xda):
    """make cmin/cmax are floats"""

    test_arr = get_test_array_2d
    test_arr = test_arr if is_xda else test_arr.values
    _, (cmin,cmax) = assign_colormap(test_arr)

    assert isinstance(cmin,float) or isinstance(cmin,np.float32)
    assert isinstance(cmax,float) or isinstance(cmax,np.float32)
