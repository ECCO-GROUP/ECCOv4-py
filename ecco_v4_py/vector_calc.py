"""
Functions defined on vector valued fields
"""

import xarray as xr
import xgcm
from .ecco_utils import get_llc_grid


def UEVNfromUXVY(xfld,yfld,coords,grid=None):
    """Compute east, north facing vector field components from x, y components
    by interpolating to cell centers and rotating by grid cell angle

    Note: this mirrors gcmfaces_calc/calc_UEVNfromUXVY.m

    Parameters
    ----------
    xfld, yfld : xarray DataArray
        fields living on west and south grid cell edges, e.g. UVELMASS and VVELMASS 
    coords : xarray Dataset
        must contain CS (cosine of grid orientation) and 
        SN (sine of grid orientation) 
    grid : xgcm Grid object, optional
        see ecco_utils.get_llc_grid and xgcm.Grid

    Returns
    -------
    u_east, v_north : xarray DataArray
        eastward and northward components of input vector field at 
        grid cell center/tracer points
    """

    # Check to make sure 'CS' and 'SN' are in coords
    # before doing calculation
    required_fields = ['CS','SN']
    for var in required_fields:
        if var not in coords.variables:
            raise KeyError('Could not find %s in coords DataSet' % var)

    # If no grid, establish it
    if grid is None:
        grid = get_llc_grid(coords)

    # First, interpolate velocity fields from cell edges to cell centers
    velc = grid.interp_2d_vector({'X': xfld, 'Y': yfld},boundary='fill')

    # Compute UE VN using cos(), sin()
    u_east = velc['X']*coords['CS'] - velc['Y']*coords['SN']
    v_north= velc['X']*coords['SN'] + velc['Y']*coords['CS']

    return u_east, v_north


def get_latitude_masks(lat_val,yc,grid):
    """Compute maskW/S which grabs vector field grid cells along specified latitude
    band and corrects the sign associated with X-Y LLC grid

    This mirrors the MATLAB function gcmfaces/gcmfaces_calc/gcmfaces_lines_zonal.m

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

    maskWedge, maskSedge : xarray DataArray
        contains masks of latitude band at grid cell west and south grid edges
    """

    # Compute difference in X, Y direction. 
    # multiply by 1 so that "True" -> 1, 2nd arg to "where" puts False -> 0 
    ones = xr.ones_like(yc)
    maskC = ones.where(yc>=lat_val,0)

    maskWedge = grid.diff( maskC, 'X', boundary='fill')
    maskSedge = grid.diff( maskC, 'Y', boundary='fill')

    return maskWedge, maskSedge

