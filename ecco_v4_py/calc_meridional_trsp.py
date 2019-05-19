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
from ecco_v4_py import vector_calc

# Define constants
# These are chosen (for now) to match gcmfaces
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
        denotes LLC90 operations for xgcm, see ecco_utils.get_llc_grid
        see also the [xgcm documentation](https://xgcm.readthedocs.io/en/latest/grid_topology.html)


    Returns
    -------
    ds_out : xarray Dataset
        Dataset with the following variables
            'vol_trsp' :
                meridional volume transport in Sv
                with dimensions 'time' (if in given dataset) and 'lat'
            'vol_trsp_z' :
                meridional volume transport in Sv at each depth level
                in the given input fields
                dimensions: 'time' (if provided), 'lat', and 'k'
    """

    x_vol = ds['UVELMASS'] * ds['drF'] * ds['dyG']
    y_vol = ds['VVELMASS'] * ds['drF'] * ds['dxG']

    # Computes salt transport in m^3/s at each depth level
    vol_trsp_z = meridional_trsp_at_depth(x_vol,y_vol,
                                          cds=ds.coords.to_dataset(),
                                          lat_vals=lat_vals,
                                          basin_name=basin_name,
                                          grid=grid)

    # Make a datset with this result
    ds_out = vol_trsp_z.to_dataset(name='vol_trsp_z')

    # Sum over depth for total transport
    ds_out['vol_trsp'] = ds_out['vol_trsp_z'].sum('k')

    # Convert both fields to Sv
    for fld in ['vol_trsp','vol_trsp_z']:
        ds_out[fld] = METERS_CUBED_TO_SVERDRUPS * ds_out[fld]
        ds_out[fld].attrs['units'] = 'Sv'

    return ds_out


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
    ds_out : xarray Dataset
        Dataset with the following variables
            'heat_trsp' :
                meridional heat transport in PW
                with dimensions 'time' (if in given dataset) and 'lat'
            'heat_trsp_z' :
                meridional heat transport in PW at each depth level
                in the given input fields
                dimensions: 'time' (if provided), 'lat', and 'k'
    """

    x_heat = ds['ADVx_TH'] + ds['DFxE_TH']
    y_heat = ds['ADVy_TH'] + ds['DFyE_TH']

    # Computes heat transport in degC * m^3/s at each depth level
    heat_trsp_z = meridional_trsp_at_depth(x_heat,y_heat,
                                           cds=ds.coords.to_dataset(),
                                           lat_vals=lat_vals,
                                           basin_name=basin_name,
                                           grid=grid)

    # Make a datset with this result
    ds_out = heat_trsp_z.to_dataset(name='heat_trsp_z')

    # Sum over depth for total transport
    ds_out['heat_trsp'] = ds_out['heat_trsp_z'].sum('k')

    # Convert both fields to PW
    for fld in ['heat_trsp','heat_trsp_z']:
        ds_out[fld] = WATTS_TO_PETAWATTS * RHO_CONST * HEAT_CAPACITY * ds_out[fld]
        ds_out[fld].attrs['units'] = 'PW'

    return ds_out

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
    ds_out : xarray Dataset
        Dataset with the following variables
            'salt_trsp' :
                meridional salt transport in psu*Sv
                with dimensions 'time' (if in given dataset) and 'lat'
            'salt_trsp_z' :
                meridional salt transport in psu*Sv at each depth level
                in the given input fields
                dimensions: 'time' (if provided), 'lat', and 'k'
    """

    x_salt = ds['ADVx_SLT'] + ds['DFxE_SLT']
    y_salt = ds['ADVy_SLT'] + ds['DFyE_SLT']

    # Computes salt transport in psu * m^3/s at each depth level
    salt_trsp_z = meridional_trsp_at_depth(x_salt,y_salt,
                                           cds=ds.coords.to_dataset(),
                                           lat_vals=lat_vals,
                                           basin_name=basin_name,
                                           grid=grid)
    # Make a datset with this result
    ds_out = salt_trsp_z.to_dataset(name='salt_trsp_z')

    # Sum over depth for total transport
    ds_out['salt_trsp'] = ds_out['salt_trsp_z'].sum('k')

    # Convert both fields to psu.Sv
    for fld in ['salt_trsp','salt_trsp_z']:
        ds_out[fld] = METERS_CUBED_TO_SVERDRUPS * ds_out[fld]
        ds_out[fld].attrs['units'] = 'psu.Sv'

    return ds_out

# ---------------------------------------------------------------------

def meridional_trsp_at_depth(xfld, yfld, lat_vals, cds, 
                             basin_name=None, grid=None):
    """
    Compute transport of vector quantity at each depth level 
    across latitude(s) defined in lat_vals

    Parameters
    ----------
    xfld, yfld : xarray DataArray
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
        see also the [xgcm documentation](https://xgcm.readthedocs.io/en/latest/grid_topology.html)

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
        lat_maskW, lat_maskS = vector_calc.get_latitude_masks(lat, cds['YC'], grid)

        # Sum horizontally
        lat_trsp_x = (xfld * lat_maskW * basin_maskW).sum(dim=['i_g','j','tile'])
        lat_trsp_y = (yfld * lat_maskS * basin_maskS).sum(dim=['i','j_g','tile'])

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

