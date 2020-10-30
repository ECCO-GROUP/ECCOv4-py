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
RHO_CONST = 1029
HEAT_CAPACITY = 4000

def calc_meridional_vol_trsp(ds,lat_vals,
                             basin_name=None,coords=None,grid=None):
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
    coords : xarray Dataset
        separate dataset containing the coordinate information
        YC, Z, drF, dyG, dxG, optionally maskW, maskS
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

    coords = _parse_coords(ds,coords,['Z','YC','drF','dyG','dxG'])

    x_vol = ds['UVELMASS'] * coords['drF'] * coords['dyG']
    y_vol = ds['VVELMASS'] * coords['drF'] * coords['dxG']

    # Computes salt transport in m^3/s at each depth level
    ds_out = meridional_trsp_at_depth(x_vol,y_vol,
                                      coords=coords,
                                      lat_vals=lat_vals,
                                      basin_name=basin_name,
                                      grid=grid)

    # Rename to useful data array name
    ds_out = ds_out.rename({'trsp_z': 'vol_trsp_z'})

    # Sum over depth for total transport
    ds_out['vol_trsp'] = ds_out['vol_trsp_z'].sum('k')

    # Convert both fields to Sv
    for fld in ['vol_trsp','vol_trsp_z']:
        ds_out[fld] = METERS_CUBED_TO_SVERDRUPS * ds_out[fld]
        ds_out[fld].attrs['units'] = 'Sv'

    return ds_out


def calc_meridional_heat_trsp(ds,lat_vals,
                              basin_name=None,coords=None,grid=None):
    """Compute heat transport across latitude band in Petwatts
    see calc_meridional_vol_trsp for argument documentation.
    The only differences are:

    Parameters
    ----------
    ds : xarray Dataset
        must contain fields 'ADVx_TH','ADVy_TH','DFxE_TH','DFyE_TH'
    coords : xarray Dataset, optional
        in case coordinates are in a separate dataset
        only needs field 'YC' and optionally maskW, maskS

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

    coords = _parse_coords(ds,coords,['Z','YC'])

    x_heat = ds['ADVx_TH'] + ds['DFxE_TH']
    y_heat = ds['ADVy_TH'] + ds['DFyE_TH']

    # Computes heat transport in degC * m^3/s at each depth level
    ds_out = meridional_trsp_at_depth(x_heat,y_heat,
                                      coords=coords,
                                      lat_vals=lat_vals,
                                      basin_name=basin_name,
                                      grid=grid)

    # Rename to useful data array name
    ds_out = ds_out.rename({'trsp_z': 'heat_trsp_z'})

    # Sum over depth for total transport
    ds_out['heat_trsp'] = ds_out['heat_trsp_z'].sum('k')

    # Convert both fields to PW
    for fld in ['heat_trsp','heat_trsp_z']:
        ds_out[fld] = WATTS_TO_PETAWATTS * RHO_CONST * HEAT_CAPACITY * ds_out[fld]
        ds_out[fld].attrs['units'] = 'PW'

    return ds_out

def calc_meridional_salt_trsp(ds,lat_vals,
                              basin_name=None,coords=None,grid=None):
    """Compute salt transport across latitude band in psu * Sv
    see calc_meridional_vol_trsp for argument documentation.
    The only differences are:

    Parameters
    ----------
    ds : xarray Dataset
        must contain fields 'ADVx_SLT','ADVy_SLT','DFxE_SLT','DFyE_SLT'
    coords : xarray Dataset, optional
        in case coordinates are in a separate dataset
        only needs field 'YC' and optionally maskW, maskS

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

    coords = _parse_coords(ds,coords,['Z','YC'])

    x_salt = ds['ADVx_SLT'] + ds['DFxE_SLT']
    y_salt = ds['ADVy_SLT'] + ds['DFyE_SLT']

    # Computes salt transport in psu * m^3/s at each depth level
    ds_out = meridional_trsp_at_depth(x_salt,y_salt,
                                      coords=coords,
                                      lat_vals=lat_vals,
                                      basin_name=basin_name,
                                      grid=grid)

    # Rename to useful data array name
    ds_out = ds_out.rename({'trsp_z': 'salt_trsp_z'})

    # Sum over depth for total transport
    ds_out['salt_trsp'] = ds_out['salt_trsp_z'].sum('k')

    # Convert both fields to psu.Sv
    for fld in ['salt_trsp','salt_trsp_z']:
        ds_out[fld] = METERS_CUBED_TO_SVERDRUPS * ds_out[fld]
        ds_out[fld].attrs['units'] = 'psu.Sv'

    return ds_out

# ---------------------------------------------------------------------

def meridional_trsp_at_depth(xfld, yfld, lat_vals, coords,
                             basin_name=None, grid=None, less_output=True):
    """
    Compute transport of vector quantity at each depth level
    across latitude(s) defined in lat_vals

    Parameters
    ----------
    xfld, yfld : xarray DataArray
        3D spatial (+ time, optional) field at west and south grid cell edges
    lat_vals : float or list
        latitude value(s) specifying where to compute transport
    coords : xarray Dataset
        only needs YC, and optionally maskW, maskS (defining wet points)
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
        with the main variable
            'trsp_z'
                transport of vector quantity across denoted latitude band at
                each depth level with dimensions 'time' (if in given dataset),
                'k' (depth), and 'lat'
    """

    if grid is None:
        grid = get_llc_grid(coords)

    # Initialize empty DataArray with coordinates and dims
    ds_out = _initialize_trsp_data_array(coords, lat_vals)

    # Get basin mask
    maskW = coords['maskW'] if 'maskW' in coords else xr.ones_like(xfld)
    maskS = coords['maskS'] if 'maskS' in coords else xr.ones_like(yfld)
    if basin_name is not None:
        maskW = get_basin_mask(basin_name,maskW)
        maskS = get_basin_mask(basin_name,maskS)

    # These sums are the same for all lats, therefore precompute to save
    # time
    tmp_x = xfld.where(maskW)
    tmp_y = yfld.where(maskS)

    for lat in ds_out.lat.values:
        if not less_output:
            print ('calculating transport for latitutde ', lat)

        # Compute mask for particular latitude band
        lat_maskW, lat_maskS = vector_calc.get_latitude_masks(lat, coords['YC'], grid)

        # Sum horizontally
        lat_trsp_x = (tmp_x * lat_maskW).sum(dim=['i_g','j','tile'])
        lat_trsp_y = (tmp_y * lat_maskS).sum(dim=['i','j_g','tile'])

        ds_out['trsp_z'].loc[{'lat':lat}] = lat_trsp_x + lat_trsp_y

    return ds_out


def _initialize_trsp_data_array(coords, lat_vals):
    """Create an xarray DataArray with time, depth, and latitude dims

    Parameters
    ----------
    coords : xarray Dataset
        contains LLC coordinates 'k' and (optionally) 'time'
    lat_vals : int or array of ints
        latitude value(s) rounded to the nearest degree
        specifying where to compute transport

    Returns
    -------
    ds_out : xarray Dataset
        Dataset with the variables
            'trsp_z'
                zero-valued DataArray with time (optional),
                depth, and latitude dimensions
            'Z'
                the original depth coordinate
    """

    lat_vals = np.array(lat_vals) if isinstance(lat_vals,list) else lat_vals
    lat_vals = np.array([lat_vals]) if np.isscalar(lat_vals) else lat_vals
    lat_vals = xr.DataArray(lat_vals,coords={'lat':lat_vals},dims=('lat',))

    xda = xr.zeros_like(coords['k']*lat_vals)
    xda = xda if 'time' not in coords.dims else xda.broadcast_like(coords['time']).copy()

    # Convert to dataset to add Z coordinate
    xds = xda.to_dataset(name='trsp_z')
    xds['Z'] = coords['Z']
    xds = xds.set_coords('Z')

    return xds

def _parse_coords(ds,coords,coordlist):
    if coords is not None:
        return coords
    else:
        for f in set(['maskW','maskS']).intersection(ds.reset_coords().keys()):
            coordlist.append(f)

        if 'time' in ds.dims:
            coordlist.append('time')

        dsout = ds[coordlist]
        if 'domain' in ds.attrs:
            dsout.attrs['domain'] = ds.attrs['domain']
        return dsout
