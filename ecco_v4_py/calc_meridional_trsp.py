"""
Module for computing meridional transport of quantities
"""

import numpy as np
import xarray as xr
# xarray compatibility
try:
    from xarray.core.pycompat import OrderedDict
except ImportError:
    from collections import OrderedDict

from .get_basin import get_basin_mask
from .ecco_utils import get_llc_grid
from .vector_calc import get_latitude_masks

# Define constants
METERS_CUBED_TO_SVERDRUPS = 10**-6
WATTS_TO_PETAWATTS = 10**-15
RHO_CONST = 1000
HEAT_CAPACITY = 4000

def calc_meridional_vol_trsp(ds,lat_vals,basin_name=None,grid=None):
    """Compute volumetric transport across latitude band in Sverdrups

    Parameters
    ----------
    ds : xarray Dataset
        must contain UVELMASS,VVELMASS, drF, dyG, dxG
    lat_vals : float or list
        latitude value(s) specifying where to compute transport
    basin_name : string, optional
        denote ocean basin over which to compute streamfunction
        If not specified, compute global quantity
        see get_basin.get_available_basin_names for options
    grid : xgcm Grid object, optional
        denotes LLC90 operations for xgcm, see utils.get_llc_grid

    Returns
    -------
    vol_trsp : xarray DataArray
        meridional freshwater transport in Sv
        with dimensions 'time' (if in given dataset) and 'lat' 
    """

    u_vol = ds['UVELMASS'] * ds['drF'] * ds['dyG']
    v_vol = ds['VVELMASS'] * ds['drF'] * ds['dxG']

    # Computes salt transport in m^3/s at each depth level
    vol_trsp = meridional_trsp_at_depth(u_vol,v_vol,
                                       cds=ds.coords.to_dataset(),
                                       lat_vals=lat_vals,
                                       basin_name=basin_name,
                                       grid=grid)

    # Sum over depth
    vol_trsp = vol_trsp.sum('k')

    # Convert to Sv
    vol_trsp = METERS_CUBED_TO_SVERDRUPS * vol_trsp
    vol_trsp.attrs['units'] = 'Sv'

    return vol_trsp


def calc_meridional_heat_trsp(ds,lat_vals,basin_name=None,grid=None):
    """Compute heat transport across latitude band in Petwatts
    see calc_meridional_vol_trsp for argument documentation. 
    The only differences are:

    Parameters
    ----------
    ds : xarray Dataset
        must contain fields 'ADVx_TH','ADVy_TH','DFxE_TH','DFyE_TH'

    Returns
    -------
    heat_trsp : xarray DataArray
        meridional heat transport in petawatts
        with dimensions 'time' (if in given dataset) and 'lat' 
    """

    u_heat = ds['ADVx_TH'] + ds['DFxE_TH']
    v_heat = ds['ADVy_TH'] + ds['DFyE_TH']

    # Computes heat transport in degC * m^3/s at each depth level
    heat_trsp = meridional_trsp_at_depth(u_heat,v_heat,
                                         cds=ds.coords.to_dataset(),
                                         lat_vals=lat_vals,
                                         basin_name=basin_name,
                                         grid=grid)

    # Sum over depth
    heat_trsp = heat_trsp.sum('k')

    # Convert to PetaWatts
    heat_trsp = WATTS_TO_PETAWATTS * RHO_CONST * HEAT_CAPACITY * heat_trsp

    heat_trsp.attrs['units'] = 'PW'

    return heat_trsp

def calc_meridional_salt_trsp(ds,lat_vals,basin_name=None,grid=None):
    """Compute salt transport across latitude band in psu * Sv
    see calc_meridional_vol_trsp for argument documentation. 
    The only differences are:

    Parameters
    ----------
    ds : xarray Dataset
        must contain fields 'ADVx_SLT','ADVy_SLT','DFxE_SLT','DFyE_SLT'

    Returns
    -------
    salt_trsp : xarray DataArray
        meridional salt transport in psu*Sv
        with dimensions 'time' (if in given dataset) and 'lat' 
    """

    u_salt = ds['ADVx_SLT'] + ds['DFxE_SLT']
    v_salt = ds['ADVy_SLT'] + ds['DFyE_SLT']

    # Computes salt transport in psu * m^3/s at each depth level
    salt_trsp = meridional_trsp_at_depth(u_salt,v_salt,
                                         cds=ds.coords.to_dataset(),
                                         lat_vals=lat_vals,
                                         basin_name=basin_name,
                                         grid=grid)
    # Sum over depth
    salt_trsp = salt_trsp.sum('k')

    # Convert to psu * Sv
    salt_trsp = METERS_CUBED_TO_SVERDRUPS * salt_trsp
    salt_trsp.attrs['units'] = 'psu.Sv'

    return salt_trsp


# ---------------------------------------------------------------------

def meridional_trsp_at_depth(ufld, vfld, lat_vals, cds, 
                             basin_name=None, grid=None):
    """
    Compute transport of vector quantity at each depth level 
    across latitude(s) defined in lat_vals

    Parameters
    ----------
    ufld, vfld : xarray DataArray
        3D spatial (+ time, optional) field at west and south grid cell edges
    lat_vals : float or list
        latitude value(s) specifying where to compute transport
    cds : xarray Dataset
        with all LLC90 coordinates, including: maskW, maskS, YC
    basin_name : string, optional
        denote ocean basin over which to compute streamfunction
        If not specified, compute global quantity
        see get_basin.get_available_basin_names for options
    grid : xgcm Grid object, optional
        denotes LLC90 operations for xgcm, see ecco_utils.get_llc_grid
        see also xgcm.Grid

    Returns
    -------
    lat_trsp : xarray DataArray
        transport of vector quantity across denoted latitude band at
        each depth level with dimensions 'time' (if in given dataset),
        'k' (depth), and 'lat' 
    """

    if grid is None:
        grid = get_llc_grid(cds)

    if np.isscalar(lat_vals):
        lat_vals = [lat_vals]

    # Initialize empty DataArray with coordinates and dims
    lat_trsp = _initialize_trsp_data_array(cds, lat_vals)

    # Get basin mask
    if basin_name is not None:
        basin_maskW = get_basin_mask(basin_name,cds['maskW'].isel(k=0))
        basin_maskS = get_basin_mask(basin_name,cds['maskS'].isel(k=0))
    else:
        basin_maskW = cds['maskW'].isel(k=0)
        basin_maskS = cds['maskS'].isel(k=0)

    for lat in lat_vals:

        # Compute mask for particular latitude band
        lat_maskW, lat_maskS = get_latitude_masks(lat, cds['YC'], grid)

        # Sum horizontally
        lat_trsp_x = (ufld * lat_maskW * basin_maskW).sum(dim=['i_g','j','tile'])
        lat_trsp_y = (vfld * lat_maskS * basin_maskS).sum(dim=['i','j_g','tile'])

        lat_trsp.loc[{'lat':lat}] = lat_trsp_x + lat_trsp_y

    return lat_trsp


def _initialize_trsp_data_array(cds, lat_vals):
    """Create an xarray DataArray with time, depth, and latitude dims

    Parameters
    ----------
    cds : xarray Dataset
        contains LLC coordinates 'k' and (optionally) 'time'
    lat_vals : int or array of ints
        latitude value(s) rounded to the nearest degree
        specifying where to compute transport

    Returns
    -------
    da : xarray DataArray
        zero-valued DataArray with time (optional), depth, and latitude dimensions
    """

    coords = OrderedDict()
    dims = ()

    if 'time' in cds:
        coords.update( {'time': cds['time'].values} )
        dims += ('time',)
        zeros = np.zeros((len(cds['time'].values),
                          len(cds['k'].values),
                          len(lat_vals)))
    else:
        zeros = np.zeros((len(cds['k'].values),
                          len(lat_vals)))


    coords.update( {'k': cds['k'].values} )
    coords.update( {'lat': lat_vals} )

    dims += ('k','lat')

    return xr.DataArray(data=zeros, coords=coords, dims=dims)

