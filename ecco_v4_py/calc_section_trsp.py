"""
Compute transport (freshwater, heat, salt) across a section, e.g. Drake Passage
"""

import numpy as np
import xarray as xr
# xarray compatibility
try:
    from xarray.core.pycompat import OrderedDict
except ImportError:
    from collections import OrderedDict

from .ecco_utils import get_llc_grid
from .get_section_masks import get_available_sections, \
        get_section_endpoints, get_section_line_masks

# -------------------------------------------------------------------------------
# Main functions for computing standard transport quantities
# -------------------------------------------------------------------------------

# Define constants
METERS_CUBED_TO_SVERDRUPS = 10**-6
WATTS_TO_PETAWATTS = 10**-15
RHO_CONST = 1029
HEAT_CAPACITY = 4000

def calc_section_vol_trsp(ds,
                          pt1=None, pt2=None,
                          section_name=None,
                          maskW=None, maskS=None,
                          coords=None, grid=None):
    """Compute volumetric transport across section in Sverdrups
    There are 3 ways to call this function:

    1. Provide pre-defined section_name, e.g.

        >> trsp = calc_section_vol_trsp(ds,'Drake Passage')

            * Computes volumetric trsp across predefined Drake Passage line
            * See get_available_sections for available definitions

    2. Provide lat/lon pairs to compute transport across, e.g.

        >> pt1 = [lon1, lat1]
        >> pt2 = [lon2, lat2]
        >> trsp = calc_section_vol_trsp(ds,pt1,pt2)

            * Computes volumetric transport across a band between pt1 -> pt2
            * If section name is provided, it gets added to returned DataArray

    3. Provide maskW, maskS, e.g.

        >> _, maskW, maskS = get_section_line_masks(pt1, pt2, ds)
        >> trsp = calc_section_vol_trsp(ds,maskW,maskS)

            * Compute trsp across band defined by masks
            * If section name is provided, it gets added to returned DataArray

    Parameters
    ----------
    ds : xarray Dataset
        must contain UVELMASS,VVELMASS, drF, dyG, dxG
    pt1, pt2 : list or tuple with two floats, optional
        end points for section line as [lon lat] or (lon, lat)
    maskW, maskS : xarray DataArray, optional
        masks denoting the section, created by get_section_line_masks
    section_name: string, optional
        name for the section. If predefined value, section mask is defined
        via get_section_endpoints
        otherwise, adds name to returned DataArray
    coords : xarray Dataset
        separate dataset containing the coordinate information
        XC, YC, drF, dyG, dxG
    grid : xgcm Grid object, optional
        denotes LLC90 operations for xgcm, see ecco_utils.get_llc_grid
        see also the [xgcm documentation](https://xgcm.readthedocs.io/en/latest/grid_topology.html)

    Returns
    -------
    ds_out : xarray Dataset
        includes variables as xarray DataArrays
            vol_trsp
                volumetric transport across section in Sv
                with dimensions 'time' (if in given dataset) and 'lat'
            vol_trsp_z
                volumetric transport across section at each depth level in Sv
                with dimensions 'time' (if in given dataset), 'lat', and 'k'
            maskW, maskS
                defining the section
        and the section_name as an attribute if it is provided
    """

    coords = coords if coords is not None else ds[['drF','dyG','dxG','XC','YC','Z']]
    maskW, maskS = _parse_section_trsp_inputs(coords,pt1,pt2,maskW,maskS,section_name,
                                              grid=grid)

    # Define volumetric transport
    x_vol = ds['UVELMASS'] * coords['drF'] * coords['dyG']
    y_vol = ds['VVELMASS'] * coords['drF'] * coords['dxG']

    # Computes salt transport in m^3/s at each depth level
    ds_out = section_trsp_at_depth(x_vol,y_vol,maskW,maskS,
                                   coords=coords)

    # Rename to useful data array name
    ds_out = ds_out.rename({'trsp_z': 'vol_trsp_z'})

    # Sum over depth for total transport
    ds_out['vol_trsp'] = ds_out['vol_trsp_z'].sum('k')

    # Convert both fields to Sv
    for fld in ['vol_trsp','vol_trsp_z']:
        ds_out[fld] = METERS_CUBED_TO_SVERDRUPS * ds_out[fld]
        ds_out[fld].attrs['units'] = 'Sv'

    # Add section name and masks to Dataset
    ds_out['maskW'] = maskW
    ds_out['maskS'] = maskS
    if section_name is not None:
        ds_out.attrs['name'] = section_name

    return ds_out

def calc_section_heat_trsp(ds,
                           pt1=None, pt2=None,
                           section_name=None,
                           maskW=None, maskS=None,
                           coords=None,grid=None):
    """Compute heat transport across section in PW
    Inputs and usage are same as calc_section_vol_trsp.
    The only differences are:

    Parameters
    ----------
    ds : xarray Dataset
        must contain ADVx_TH, ADVy_TH, DFxe_TH, DFyE_TH

    Returns
    -------
    heat_trsp_ds : xarray Dataset
        includes variables as xarray DataArrays
            heat_trsp
                heat transport across section in PW
                with dimensions 'time' (if in given dataset) and 'lat'
            heat_trsp_z
                heat transport across section at each depth level in PW
                with dimensions 'time' (if in given dataset), 'lat', and 'k'
            maskW, maskS
                defining the section
        and the section_name as an attribute if it is provided
    """

    coords = coords if coords is not None else ds[['XC','YC','Z']]
    maskW, maskS = _parse_section_trsp_inputs(ds,pt1,pt2,maskW,maskS,section_name,
                                              grid=grid)

    # Define heat transport
    x_heat = ds['ADVx_TH'] + ds['DFxE_TH']
    y_heat = ds['ADVy_TH'] + ds['DFyE_TH']

    # Computes salt transport in degC * m^3/s at each depth level
    ds_out = section_trsp_at_depth(x_heat,y_heat,maskW,maskS,
                                   coords=coords)

    # Rename to useful data array name
    ds_out = ds_out.rename({'trsp_z': 'heat_trsp_z'})

    # Sum over depth for total transport
    ds_out['heat_trsp'] = ds_out['heat_trsp_z'].sum('k')

    # Convert both fields to PW
    for fld in ['heat_trsp','heat_trsp_z']:
        ds_out[fld] = WATTS_TO_PETAWATTS * RHO_CONST * HEAT_CAPACITY * ds_out[fld]
        ds_out[fld].attrs['units'] = 'PW'

    # Add section name and masks to Dataset
    ds_out['maskW'] = maskW
    ds_out['maskS'] = maskS
    if section_name is not None:
        ds_out.attrs['name'] = section_name

    return ds_out

def calc_section_salt_trsp(ds,
                           pt1=None, pt2=None,
                           section_name=None,
                           maskW=None, maskS=None,
                           coords=None, grid=None):
    """Compute salt transport across section in psu*Sv
    Inputs and usage are same as calc_section_vol_trsp.
    The only differences are:

    Parameters
    ----------
    ds : xarray Dataset
        must contain ADVx_SLT, ADVy_SLT, DFxe_SLT, DFyE_SLT

    Returns
    -------
    salt_trsp_ds : xarray Dataset
        includes variables as xarray DataArrays
            salt_trsp
                salt transport across section in psu*Sv
                with dimensions 'time' (if in given dataset) and 'lat'
            salt_trsp_z
                salt transport across section at each depth level in psu*Sv
                with dimensions 'time' (if in given dataset), 'lat', and 'k'
            maskW, maskS
                defining the section
        and the section_name as an attribute if it is provided
    """

    coords = coords if coords is not None else ds[['XC','YC','Z']]
    maskW, maskS = _parse_section_trsp_inputs(ds,pt1,pt2,maskW,maskS,section_name,
                                              grid=grid)

    # Define salt transport
    x_salt = ds['ADVx_SLT'] + ds['DFxE_SLT']
    y_salt = ds['ADVy_SLT'] + ds['DFyE_SLT']

    # Computes salt transport in psu * m^3/s at each depth level
    ds_out = section_trsp_at_depth(x_salt,y_salt,maskW,maskS,
                                   coords=coords)

    # Rename to useful data array name
    ds_out = ds_out.rename({'trsp_z': 'salt_trsp_z'})

    # Sum over depth for total transport
    ds_out['salt_trsp'] = ds_out['salt_trsp_z'].sum('k')

    # Convert both fields to psu.Sv
    for fld in ['salt_trsp','salt_trsp_z']:
        ds_out[fld] = METERS_CUBED_TO_SVERDRUPS * ds_out[fld]
        ds_out[fld].attrs['units'] = 'psu.Sv'

    # Add section name and masks to Dataset
    ds_out['maskW'] = maskW
    ds_out['maskS'] = maskS
    if section_name is not None:
        ds_out.attrs['name'] = section_name

    return ds_out

# -------------------------------------------------------------------------------
# Main function for computing standard transport quantities
# -------------------------------------------------------------------------------

def section_trsp_at_depth(xfld, yfld, maskW, maskS, coords):
    """
    Compute transport of vector quantity at each depth level
    across latitude(s) defined in lat_vals

    Parameters
    ----------
    xfld, yfld : xarray DataArray
        3D spatial (+ time, optional) field at west and south grid cell edge
    maskW, maskS : xarray DataArray
        defines the section to define transport across
    coords : xarray Dataset
        with all LLC90 coordinates, including: maskW/S, YC
    grid : xgcm Grid object, optional
        denotes LLC90 operations for xgcm, see utils.get_llc_grid

    Returns
    -------
    ds_out : xarray Dataset
        with the main variable
            'trsp_z'
                transport of vector quantity across denoted section at
                each depth level with dimensions 'time' (if in given dataset),
                and 'k' (depth)
    """

    # Initialize empty DataArray with coordinates and dims
    ds_out = _initialize_section_trsp_data_array(coords)

    # Apply section mask and sum horizontally
    maskW = maskW.where(cds['maskW']) if 'maskW' in cds else maskW
    maskS = maskS.where(cds['maskS']) if 'maskS' in cds else maskS
    sec_trsp_x = (xfld * maskW).sum(dim=['i_g','j','tile'])
    sec_trsp_y = (yfld * maskS).sum(dim=['i','j_g','tile'])

    ds_out['trsp_z'] = sec_trsp_x + sec_trsp_y

    return ds_out


# -------------------------------------------------------------------------------
#
# All functions below are non-user facing
#
# -------------------------------------------------------------------------------
# Helper functions for the computing volume, heat, and salt transport
# -------------------------------------------------------------------------------

def _parse_section_trsp_inputs(ds,pt1,pt2,maskW,maskS,section_name,grid=None):
    """Handle inputs for computing volume, heat, or salt transport across
    a section

    Parameters
    ----------
    see calc_section_vol_trsp

    Returns
    -------
    maskW, maskS : xarray DataArray
        masks defining the section
    """

    use_predefined_section = False
    use_endpoints = False
    use_masks = False

    # Test if section name is in available basins
    if section_name is not None:
        if get_section_endpoints(section_name) is not None:
            use_predefined_section = True

    # Test if endpoints provided
    if (pt1 is not None and pt2 is not None):
        use_endpoints = True

    # Test if masks provided
    if (maskW is not None and maskS is not None):
        use_masks = True

    # Test to make sure section is defined by at least one method
    if not use_predefined_section and not use_endpoints and not use_masks:
        raise TypeError('Must provide one method for defining section')

    # First, try to use predefined section
    if use_predefined_section:
        if use_endpoints or use_masks:
            raise TypeError('Cannot provide more than one method for defining section')
        pt1, pt2 = get_section_endpoints(section_name)
    else:
        # Secondly, try to use endpoints or mask
        if use_endpoints and use_masks:
            raise TypeError('Cannot provide more than one method for defining section')
    _, maskW, maskS = get_section_line_masks(pt1, pt2, ds, grid=grid)

    return maskW, maskS

def _initialize_section_trsp_data_array(coords):
    """Create an xarray DataArray with time, depth, and latitude dims

    Parameters
    ----------
    coords : xarray Dataset
        contains LLC coordinates 'k' and (optionally) 'time'

    Returns
    -------
    ds_out : xarray Dataset
        Dataset with the variables
            'trsp_z'
                zero-valued DataArray with time (optional) and
                depth dimensions
            'Z'
                the original depth coordinate
    """

    ddict = OrderedDict()
    dims = ()

    xda = xr.zeros_like(cds['k'])
    xda = xda if 'time' not in cds.dims else xda.broadcast_like(cds['time'])

    # Convert to dataset to add Z coordinate
    xds = xda.to_dataset(name='trsp_z')
    xds['Z'] = coords['Z']
    xds = xds.set_coords('Z')

    return xds
