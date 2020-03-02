"""
Test routines for the tile plotting
"""
from __future__ import division, print_function
import warnings
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import pytest
import ecco_v4_py as ecco

from .test_plot_utils import get_test_array

# Define bin directory for test reading
_PKG_DIR = Path.cwd().resolve().parent.parent
_DATA_DIR = _PKG_DIR.joinpath('binary_data')

_TEST_FILES = ['basins.data', 'hFacC.data', 'state_3d_set1.0000000732.data']
_TEST_NK = [1, 50, 50]
_TEST_RECS = [1, 1, 3]

@pytest.mark.parametrize("vdict",[
        {}, #defaults
        {'cmap':'plasma'},
        {'layout':'latlon', 
          'rotate_to_latlon':False},
        {'layout':'latlon',
         'rotate_to_latlon':False,
         'Arctic_cap_tile_location':5},
        {'layout':'latlon',
         'rotate_to_latlon':False,
         'Arctic_cap_tile_location':7},
        {'layout':'latlon',
         'rotate_to_latlon':False,
         'Arctic_cap_tile_location':10},
        {'show_colorbar':True},
        {'show_colorbar':True,
         'show_cbar_label':True,
         'cbar_label':'something good'},
        {'show_tile_labels':False},
        {'fig_size':20},
        {'less_output':False}
    ])
def test_plot_tiles(vdict):
    """Run through various options and make sure nothing is broken"""

    # read in array
    nparr = get_test_array(is_xda=False)
    xda = get_test_array(is_xda=True)

    # run plotting routine
    for arr in [nparr,xda]:
        ecco.plot_tiles(arr, **vdict)
        plt.close()

@pytest.mark.parametrize("vdict",[
        {}, #defaults
        {'cmap':'plasma'},
        {'show_colorbar':True},
        {'show_colorbar':True,
         'show_cbar_label':True,
         'cbar_label':'something good'},
        {'show_tile_labels':False},
        {'less_output':False}
    ])
def test_plot_single_tile(vdict):
    """plot a single tile"""

    # read in array
    nparr = get_test_array(is_xda=False)
    xda = get_test_array(is_xda=True)

    # run plotting routine on each tile
    for t in xda.tile.values:

        ecco.plot_tile(nparr[t,...], **vdict)
        plt.close()

        ecco.plot_tile(xda.sel(tile=t), **vdict)
        plt.close()


def test_plot_tiles_array():
    """a crude test to make sure the array being created
    matches the original"""

    nparr = get_test_array(is_xda=False)
    xda = get_test_array(is_xda=True)

    for arr_expected in [nparr,xda]:
        _,arr_test = ecco.plot_tiles(arr_expected)
        assert np.nansum(arr_test)==float(np.nansum(arr_expected))
        plt.close()
