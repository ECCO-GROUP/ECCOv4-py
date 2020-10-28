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

from .test_common import all_mds_datadirs, get_test_ds

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
def test_plot_proj(get_test_ds,kwargs,dx,dy):
    """Run through various options and make sure nothing is broken"""

    ds = get_test_ds
    kwargs['dx']=dx
    kwargs['dy']=dy
    print(ds)
    if 'blah' in kwargs.values():

        with pytest.raises(NotImplementedError):
            plot_proj_to_latlon_grid(ds.XC,ds.YC,ds.ETAN,**kwargs)

    else:
        plot_proj_to_latlon_grid(ds.XC,ds.YC,ds.ETAN,**kwargs)
        plt.close()
