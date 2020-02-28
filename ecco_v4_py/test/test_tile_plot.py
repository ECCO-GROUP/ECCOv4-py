"""
Test routines for the tile plotting
"""
from __future__ import division, print_function
import warnings
from pathlib import Path
import numpy as np
import pytest
import ecco_v4_py as ecco

# Define bin directory for test reading
_PKG_DIR = Path.cwd().resolve().parent.parent
_DATA_DIR = _PKG_DIR.joinpath('binary_data')

_TEST_FILES = ['basins.data', 'hFacC.data', 'state_3d_set1.0000000732.data']
_TEST_NK = [1, 50, 50]
_TEST_RECS = [1, 1, 3]

def get_default_plot_tiles_input():
    """make a dict with default inputs"""
    return {'cmap':None,
            'layout':'llc',
            'rotate_to_latlon':False,
            'show_colorbar':False,
            'show_cbar_label':False,
            'show_tile_labels':True,
            'fig_size':9,
            'less_output':True}

@pytest.mark.parametrize("fname,vdict",[
        (_TEST_FILES[0],{}), # test defaults
        (_TEST_FILES[0],{'layout':'latlon', 
                         'rotate_to_latlon':False),
        (_TEST_FILES[0],{'layout':'latlon',
                         'rotate_to_latlon':False,
                         'Arctic_cap_tile_location':5}),
    ])
def test_plot_tiles(fname,vdict):
    """Run through various options and make sure nothing is broken"""

    #get defaults
    inputs=get_default_plot_tiles_input()
    for key, val in vdict.items():
        inputs[key] = val

    # 
