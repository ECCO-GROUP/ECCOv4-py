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

from xmitgcm.test.test_xmitgcm_common import llc_mds_datadirs
from .test_common import get_test_array_2d

@pytest.mark.parametrize("vdict",[
        {}, #defaults
        {'cmap':'plasma'},
        {'layout':'latlon', 
          'rotate_to_latlon':True},
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
        {'less_output':False},
        {'show_tile_labels':True,
         'show_colorbar':True},
        {'show_tile_labels':True,
         'show_colorbar':False}
    ])
@pytest.mark.parametrize("is_xda",[True,False])
def test_plot_tiles(llc_mds_datadirs,vdict,is_xda):
    """Run through various options and make sure nothing is broken"""

    test_arr = get_test_array_2d(llc_mds_datadirs,is_xda=is_xda)
    ecco.plot_tiles(test_arr,**vdict)
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
@pytest.mark.parametrize("is_xda",[True,False])
def test_plot_single_tile(llc_mds_datadirs,vdict,is_xda):
    """plot a single tile"""

    # read in array
    test_arr = get_test_array_2d(llc_mds_datadirs,is_xda=is_xda)

    # run plotting routine on each tile
    for t in np.arange(test_arr.shape[0]):
        ecco.plot_tile(test_arr[t,...], **vdict)
        plt.close()


@pytest.mark.parametrize("is_xda",[True,False])
def test_plot_tiles_array(llc_mds_datadirs,is_xda):
    """a crude test to make sure the array being created
    matches the original"""

    arr_expected = get_test_array_2d(llc_mds_datadirs,is_xda=is_xda)
    _,arr_test = ecco.plot_tiles(arr_expected)
    assert np.allclose(np.nansum(arr_test),float(np.nansum(arr_expected)))
    plt.close()
