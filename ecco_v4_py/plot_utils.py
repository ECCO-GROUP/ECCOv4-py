"""
Some helper functions for plotting
"""

import numpy as np
import xarray as xr

def assign_colormap(fld,user_cmap=None):
    """assign a default colormap based on input field
    following xarray defaults

    Sequential fld: viridis
    Divergent fld : RdBu_r
    

    Parameters
    ----------
    fld : xarray.DataArray or numpy.ndarray
        must be in tiled format

    user_cmap : str or None, optional
        if user_cmap is specified, use this
        None if not specified, and default cmaps selected

    Returns
    -------
    cmap : str
        assigned colormap depending on diverging or sequential
        data in field

    (cmin,cmax) : tuple of floats
        minimum and maximum values for colormap
    """

    #%%    
    # If fld span positive/negative, default to normalize about 0
    # otherwise, regular (sequential). Assign cmap accordingly.
    cmin = np.nanmin(fld)
    cmax = np.nanmax(fld)
    if cmin*cmax<0:
        cmax=np.nanmax(np.abs(fld))
        cmin=-cmax
        cmap = 'RdBu_r'
    else:
        cmap = 'viridis'

    # override if user_cmap is not None
    cmap = cmap if user_cmap is None else user_cmap

    return cmap, (cmin,cmax)
