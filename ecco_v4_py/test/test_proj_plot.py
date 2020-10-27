"""
Test routines for the tile plotting
"""
from __future__ import division, print_function
import warnings
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import pytest
from ecco_v4_py import plot_proj_to_latlon_grid

from .test_common import llc_mds_datadirs,get_test_array_2d

@pytest.mark.parametrize("kwargs",
        [   {'projection_type':'Mercator'},
            {'projection_type':'LambertConformal'},
            {'projection_type':'AlbersEqualArea'},
            {'projection_type':'PlateCarree'},
            {'projection_type':'cyl'},
            {'projection_type':'ortho'},
            {'projection_type':'ortho','lat_lim':-50},
            {'projection_type':'ortho','user_lat_0':60},
            {'projection_type':'ortho','user_lat_0':-60},
            {'projection_type':'InterruptedGoodeHomolosine'},
            {'projection_type':'blah'},
            {'plot_type':'contourf'},
            {'user_lon_0':-180},
            {'user_lon_0':180},
            {'show_colorbar':True},
            {'subplot_grid':[1,2,1]},
            {'subplot_grid':{'nrows':1,'ncols':2,'index':1}},
            {'cmap':'hot'},
            {'cmin':-1,'cmax':1},
            {'cmin':1,'cmax':10},
            {'cmin':-10,'cmax':-1},
            {'show_land':False,'show_coastline':False},
            {'show_grid_lines':False}])
@pytest.mark.parametrize("dx, dy",[(1,1)])
def test_plot_proj(get_test_array_2d,kwargs,dx,dy):
    """Run through various options and make sure nothing is broken"""

    test_arr = get_test_array_2d
    kwargs['dx']=dx
    kwargs['dy']=dy
    if 'blah' in kwargs.values():

        with pytest.raises(NotImplementedError):
            plot_proj_to_latlon_grid(test_arr.XC,test_arr.YC,test_arr,**kwargs)

    else:
        plot_proj_to_latlon_grid(test_arr.XC,test_arr.YC,test_arr,**kwargs)
        plt.close()
