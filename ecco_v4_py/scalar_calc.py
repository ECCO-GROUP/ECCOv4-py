"""
generic functions for scalar valued (cell centered) quantities
"""

import xarray as xr

def get_latitude_masks(lat_val,yc,grid):
    """Compute maskCedge which grabs the grid cell center points along 
    the desired latitude

    This mirrors the MATLAB function  gcmfaces/gcmfaces_calc/gcmfaces_lines_zonal.m

    Parameters
    ----------

    lat_val : int
        latitude at which to compute mask 
    yc : xarray DataArray
        Contains latitude values at cell centers
    grid : xgcm Grid object
        llc grid object generated via get_llc_grid

    Returns
    -------

    maskCedge : xarray DataArray
        contains mask of latitude at grid cell tracer points
    """

    # Compute difference in X, Y direction. 
    # multiply by 1 so that "True" -> 1, 2nd arg to "where" puts False -> 0 
    ones = xr.ones_like(yc)
    lat_maskC = ones.where(yc>=lat_val,0)

    maskCedge = get_edge_mask(lat_maskC,grid)

    return maskCedge

def get_edge_mask(maskC,grid):
    """From a given mask with points at cell centers, compute the 
    boundary between 1's and 0's

    Parameters
    ----------
    
    maskC : xarray DataArray
        containing 1's at interior points, 0's outside. We want the 
        boundary between them
    grid : xgcm Grid object

    Returns
    -------

    maskCedge : xarray DataArray
        with same dimensions as input maskC, with 1's at boundary 
        between 1's and 0's
    """

    # This first interpolation gets 0.5 at boundary points
    # however, the result lives on West and South grid cell edges
    maskX = grid.interp(maskC,'X', boundary='fill')
    maskY = grid.interp(maskC,'Y', boundary='fill')

    # Now interpolate these to get back on to cell centers
    # edge will now be at locations where values are 0.75
    maskXY= grid.interp_2d_vector({'X' : maskX, 'Y' : maskY}, boundary='fill')

    # Now wherever this is > 0 and the original mask is 0 is the boundary
    maskCedge = xr.ones_like(maskC).where( ((maskXY['X'] + maskXY['Y']) > 0) & (maskC==0.) , 0)

    return maskCedge
