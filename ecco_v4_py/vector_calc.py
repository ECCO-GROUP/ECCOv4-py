"""
Functions defined on vector valued fields
"""

import xarray as xr
import xgcm
from .ecco_utils import get_llc_grid


def UEVNfromUXVY(x_fld,y_fld, coords, grid=None):
    """Compute the zonal and meridional components of a vector field defined
    by its x and y components with respect to the model grid.

    The x and y components of the vector can be defined on model grid cell edges ('u' and 'v' points),
    or on model grid cell centers ('c' points). If the vector components are defined on the grid cell edges then the function will first interpolate them to the grid cell centers. 
    
    Once both x and y vector components are at the cell centers, they are rotated to the zonal and meridional components using the cosine and sine of the grid cell angle. The grid cell angle is defined as the angle between the Earth's parallels and the line connecting the center of a grid cell to the center of its neighbor in the x-direction. The cosine and sine of this angle are provided in the 'coords' input field.

    The function will raise an error if the grid cell angle terms (CS, SN) are not provided in the coords

    Example vector fields provided at the grid cell edges are UVEL and VVEL. Example fields provided at the grid cell centers are EXFuwind and EXFvwind. 

    Note: this routine is inspired by gcmfaces_calc/calc_UEVNfromUXVY.m

    Parameters
    ----------
    x_fld, y_fld : xarray DataArray
        x and y components of a vector field provided at the model grid cell edges or centers
    coords : xarray Dataset
        must contain CS (cosine of grid orientation) and SN (sine of grid orientation)
    grid : xgcm Grid object, optional
        see ecco_utils.get_llc_grid and xgcm.Grid
        If not provided, the function will create one using the coords dataset.
    
    Returns
    -------
    e_fld, n_fld : xarray DataArray
        zonal (positive east) and meridional (positive north) components of input vector field 
        at grid cell centers/tracer points
    """

    # Check to make sure 'CS' and 'SN' are in coords
    # before doing calculation
    required_fields = ['CS','SN']
    for var in required_fields:
        if var not in coords.variables:
            raise KeyError('Could not find %s in coords Dataset' % var)

    # If no grid, establish it
    if grid is None:
        grid = get_llc_grid(coords)

    # Determine if vector fields are at cell edges or cell centers
    # by checking the dimensions of the x and y fields
    # If the vector components are at cell edges, we need to interpolate to cell centers
    # If the vector components are at cell centers, we don't need to interpolate
    # Create a set with all of the dimensions of the x and y fields
    vector_dims = set(x_fld.dims + y_fld.dims)
    
    # if neither 'i_g' and 'j_g' are present then the vector components are at cell centers
    vector_components_at_cell_center = 'i_g' not in vector_dims and 'j_g' not in vector_dims

    if vector_components_at_cell_center:
        # If the vector components are at cell centers, we don't need to interpolate
        # vec_field is a dictionary with the x and y fields
        vec_field = {'X': x_fld, 'Y': y_fld}
    else:
        # If the vector components are at cell edges, we need to interpolate to cell centers
        # vec_field is a dictionary with the x and y fields
        vec_field = grid.interp_2d_vector({'X': x_fld, 'Y': y_fld},boundary='fill')
        
    # Compute the zonal "e" and meridional components "n" using cos(), sin()
    e_fld = vec_field['X']*coords['CS'] - vec_field['Y']*coords['SN']
    n_fld = vec_field['X']*coords['SN'] + vec_field['Y']*coords['CS']

    return e_fld, n_fld



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

