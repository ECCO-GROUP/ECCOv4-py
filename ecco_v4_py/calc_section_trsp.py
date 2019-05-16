"""
Compute transport (freshwater, heat, salt) across a section, e.g. Drake Passage
"""

import warnings
import numpy as np
import xarray as xr
# xarray compatibility
try:
    from xarray.core.pycompat import OrderedDict
except ImportError:
    from collections import OrderedDict

from .utils import get_llc_grid, remake_llc_xda
from . import scalar_calc

# -------------------------------------------------------------------------------
# Main functions for computing standard transport quantities
# -------------------------------------------------------------------------------

# Define constants
METERS_CUBED_TO_SVERDRUPS = 10**-6
WATTS_TO_PETAWATTS = 10**-15
RHO_CONST = 1000
HEAT_CAPACITY = 4000

def calc_section_vol_trsp(ds,
                          pt1=None, pt2=None,
                          section_name=None,
                          maskW=None, maskS=None, 
                          grid=None):
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
    pt1, pt2 : array_like or tuple with two floats, optional
        end points for section line as [lon lat] or (lon, lat)
    maskW, maskS : xarray DataArray, optional
        masks denoting the section, created by get_section_line_masks
    section_name: string, optional
        name for the section. If predefined value, section mask is defined 
        via get_section_endpoints
        otherwise, adds name to returned DataArray
    grid : xgcm Grid object, optional
        denotes LLC90 operations for xgcm, see utils.get_llc_grid

    Returns
    -------
    vol_trsp_ds : xarray Dataset
        includes variables as xarray DataArrays
            vol_trsp
                freshwater transport across section in Sv
                with dimensions 'time' (if in given dataset) and 'lat' 
            maskW, maskS
                defining the section
        and the section_name as an attribute if it is provided
    """

    maskW, maskS = _parse_section_trsp_inputs(ds,pt1,pt2,maskW,maskS,section_name)

    # Define volumetric transport
    u_vol = ds['UVELMASS'] * ds['drF'] * ds['dyG'] 
    v_vol = ds['VVELMASS'] * ds['drF'] * ds['dxG'] 

    # Computes salt transport in m^3/s at each depth level
    vol_trsp = section_trsp_at_depth(u_vol,v_vol,maskW,maskS,
                                     cds=ds.coords.to_dataset(),
                                     grid=grid)
    # Sum over depth
    vol_trsp = vol_trsp.sum('k')

    # Convert to Sv
    vol_trsp = METERS_CUBED_TO_SVERDRUPS * vol_trsp
    vol_trsp.attrs['units'] = 'Sv'

    # Add this to a Dataset
    vol_trsp_ds = vol_trsp.to_dataset(name='vol_trsp')

    # Add section name and masks to Dataset
    vol_trsp_ds['maskW'] = maskW
    vol_trsp_ds['maskS'] = maskS
    if section_name is not None:
        vol_trsp_ds.attrs['name'] = section_name

    return vol_trsp_ds

def calc_section_heat_trsp(ds,
                           pt1=None, pt2=None,
                           section_name=None,
                           maskW=None, maskS=None, 
                           grid=None):
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
            maskW, maskS
                defining the section
        and the section_name as an attribute if it is provided
    """

    maskW, maskS = _parse_section_trsp_inputs(ds,pt1,pt2,maskW,maskS,section_name)

    # Define heat transport
    u_heat = ds['ADVx_TH'] * ds['DFxE_TH']
    v_heat = ds['ADVy_TH'] * ds['DFyE_TH']

    # Computes salt transport in degC * m^3/s at each depth level
    heat_trsp = section_trsp_at_depth(u_heat,v_heat,maskW,maskS,
                                      cds=ds.coords.to_dataset(),
                                      grid=grid)
    # Sum over depth
    heat_trsp = heat_trsp.sum('k')

    # Convert to PW
    heat_trsp = WATTS_TO_PETAWATTS * RHO_CONST * HEAT_CAPACITY * heat_trsp
    heat_trsp.attrs['units'] = 'PW'

    # Add this to a Dataset
    heat_trsp_ds = heat_trsp.to_dataset(name='heat_trsp')

    # Add section name and masks to Dataset
    heat_trsp_ds['maskW'] = maskW
    heat_trsp_ds['maskS'] = maskS
    if section_name is not None:
        heat_trsp_ds.attrs['name'] = section_name

    return heat_trsp_ds

def calc_section_salt_trsp(ds,
                           pt1=None, pt2=None,
                           section_name=None,
                           maskW=None, maskS=None, 
                           grid=None):
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
            maskW, maskS
                defining the section
        and the section_name as an attribute if it is provided
    """

    maskW, maskS = _parse_section_trsp_inputs(ds,pt1,pt2,maskW,maskS,section_name)

    # Define salt transport
    u_salt = ds['ADVx_SLT'] * ds['DFxE_SLT']
    v_salt = ds['ADVy_SLT'] * ds['DFyE_SLT']

    # Computes salt transport in psu * m^3/s at each depth level
    salt_trsp = section_trsp_at_depth(u_salt,v_salt,maskW,maskS,
                                      cds=ds.coords.to_dataset(),
                                      grid=grid)
    # Sum over depth
    salt_trsp = salt_trsp.sum('k')

    # Convert to PW
    salt_trsp = METERS_CUBED_TO_SVERDRUPS * salt_trsp
    salt_trsp.attrs['units'] = 'psu.Sv'

    # Add this to a Dataset
    salt_trsp_ds = salt_trsp.to_dataset(name='salt_trsp')

    # Add section name and masks to Dataset
    salt_trsp_ds['maskW'] = maskW
    salt_trsp_ds['maskS'] = maskS
    if section_name is not None:
        salt_trsp_ds.attrs['name'] = section_name

    return salt_trsp_ds

# -------------------------------------------------------------------------------
# Main functions for computing standard transport quantities
# -------------------------------------------------------------------------------

def section_trsp_at_depth(ufld, vfld, maskW, maskS, cds, 
                          grid=None):
    """
    Compute transport of vector quantity at each depth level 
    across latitude(s) defined in lat_vals

    Parameters
    ----------
    ufld, vfld : xarray DataArray
        3D spatial (+ time, optional) field at west and south grid cell edge
    maskW, maskS : xarray DataArray
        defines the section to define transport across
    cds : xarray Dataset
        with all LLC90 coordinates, including: maskW/S, YC
    grid : xgcm Grid object, optional
        denotes LLC90 operations for xgcm, see utils.get_llc_grid

    Returns
    -------
    lat_trsp : xarray DataArray
        transport of vector quantity across denoted latitude band at
        each depth level with dimensions 'time' (if in given dataset),
        'k' (depth), and 'lat' 
    """

    if grid is None:
        grid = get_llc_grid(cds)

    # Initialize empty DataArray with coordinates and dims
    sec_trsp = _initialize_section_trsp_data_array(cds)

    # Apply section mask and sum horizontally
    sec_trsp_x = (ufld * maskW).sum(dim=['i_g','j','tile'])
    sec_trsp_y = (vfld * maskS).sum(dim=['i','j_g','tile'])

    return sec_trsp_x + sec_trsp_y


# -------------------------------------------------------------------------------
# Functions for generating pre-defined section masks
# -------------------------------------------------------------------------------

def get_section_endpoints(section_name):
    """Get the [lon, lat] endpoints associated with a pre-defined section
    e.g.

        >> pt1, pt2 = get_section_endpoints('Drake Passage')
        pt1 = [-68, -54]
        pt2 = [-63, -66]


    Parameters
    ----------
    section_name : str
        name of the section to compute transport across

    Returns
    -------
    pt1, pt2 : array_like
        array with two values, [lon, lat] of each endpoint

    or 

    None  
        if section_name is not in the pre-defined list of sections
    """

    # Set to input lower case and remove spaces/tabs
    section_name = ''.join(section_name.lower().split())

    # Test to see if name exists in list
    section_list = get_available_sections()
    section_list = [''.join(name.lower().split()) for name in section_list]
    if section_name not in section_list:
        warnings.warn('\nSection name %s unavailable as pre-defined section' % section_name)
        return None

    if section_name == 'drakepassage':
        pt1 = [-68, -54]
        pt2 = [-63, -66]
    elif section_name == 'beringstrait':
        pt1 = [-173, 65.5]
        pt2 = [-164, 65.5]
    elif section_name == 'gibraltar':
        pt1 = [-5, 34]
        pt2 = [-5, 40]
    elif section_name == 'floridastrait':
        pt1 = [-81, 28]
        pt2 = [-77, 26]
    elif section_name == 'floridastraitw1':
        pt1 = [-81, 28]
        pt2 = [-79, 22]
    elif section_name == 'floridastraits1':
        pt1 = [-76, 21]
        pt2 = [-76,  8]
    elif section_name == 'floridastraite1':
        pt1 = [-77, 26]
        pt2 = [-77, 24]
    elif section_name == 'floridastraite2':
        pt1 = [-77, 24]
        pt2 = [-77, 22]
    elif section_name == 'floridastraite3':
        pt1 = [-76, 21]
        pt2 = [-72, 18.5]
    elif section_name == 'floridastraite4':
        pt1 = [-72, 18.5]
        pt2 = [-72, 10]
    elif section_name == 'davisstrait':
        pt1 = [-65, 66]
        pt2 = [-50, 66]
    elif section_name == 'denmarkstrait':
        pt1 = [-35, 67]
        pt2 = [-20, 65]
    elif section_name == 'icelandfaroe':
        pt1 = [-16, 65]
        pt2 = [ -7, 62.5]
    elif section_name == 'scotlandnorway':
        pt1 = [-4, 57]
        pt2 = [ 8, 62]
    elif section_name == 'indonesiaw1':
        pt1 = [103, 4]
        pt2 = [103,-1]
    elif section_name == 'indonesiaw2':
        pt1 = [104, -3]
        pt2 = [109, -8]
    elif section_name == 'indonesiaw3':
        pt1 = [113, -8.5]
        pt2 = [118, -8.5]
    elif section_name == 'indonesiaw4':
        pt1 = [118, -8.5]
        pt2 = [127, -15]
    elif section_name == 'australiaantarctica':
        pt1 = [127, -25]
        pt2 = [127, -68]
    elif section_name == 'madagascarchannel':
        pt1 = [38, -10]
        pt2 = [46, -22]
    elif section_name == 'madagascarantarctica':
        pt1 = [46, -22]
        pt2 = [46, -69]
    elif section_name == 'southafricaantarctica':
        pt1 = [20, -30]
        pt2 = [20, -69.5]

    return pt1, pt2

def get_available_sections():
    """Return pre-defined section names for computing transports across this section

    Returns
    -------
    section_list : list of str
        list of available pre-defined sections
    """
    section_list = ['Bering Strait',
    'Gibraltar',
    'Florida Strait',
    'Florida Strait W1',
    'Florida Strait S1',
    'Florida Strait E1',
    'Florida Strait E2',
    'Florida Strait E3',
    'Florida Strait E4',
    'Davis Strait',
    'Denmark Strait',
    'Iceland Faroe',
    'Faroe Scotland',
    'Scotland Norway',
    'Drake Passage',
    'Indonesia W1',
    'Indonesia W2',
    'Indonesia W3',
    'Indonesia W4',
    'Australia Antarctica',
    'Madagascar Channel',
    'Madagascar Antarctica',
    'South Africa Antarctica']


    return section_list

# -------------------------------------------------------------------------------
# Main function to compute section masks 
# -------------------------------------------------------------------------------

def get_section_line_masks(pt1, pt2, cds):
    """Compute 2D mask with 1's along great circle line between lat/lon1 -> lat/lon2

    Parameters
    ----------
    pt1, pt2 : tuple or array with 2 floats
        [longitude, latitude] or (longitude, latitude) of endpoints
    cds : xarray Dataset
        containing grid coordinate information, at least XC, YC

    Returns
    -------
    section_mask : xarray DataArray
        2D mask along section
    """

    # Get cartesian coordinates of end points 
    x1, y1, z1 = _convert_latlon_to_cartesian(pt1[0],pt1[1])
    x2, y2, z2 = _convert_latlon_to_cartesian(pt2[0],pt2[1])

    # Compute rotation matrices
    # 1. Rotate around x-axis to put first point at z = 0
    theta_1 = np.arctan2(-z1, y1)
    rot_1 = np.vstack(( [1, 0, 0],
                        [0, np.cos(theta_1),-np.sin(theta_1)],
                        [0, np.sin(theta_1), np.cos(theta_1)]))

    x1, y1, z1 = _apply_rotation_matrix(rot_1, (x1,y1,z1))
    x2, y2, z2 = _apply_rotation_matrix(rot_1, (x2,y2,z2))

    # 2. Rotate around z-axis to put first point at y = 0
    theta_2 = np.arctan2(x1,y1)
    rot_2 = np.vstack(( [np.cos(theta_2),-np.sin(theta_2), 0],
                        [np.sin(theta_2), np.cos(theta_2), 0],
                        [0, 0, 1]))

    x1, y1, z1 = _apply_rotation_matrix(rot_2, (x1,y1,z1))
    x2, y2, z2 = _apply_rotation_matrix(rot_2, (x2,y2,z2))

    # 3. Rotate around y-axis to put second point at z = 0
    theta_3 = np.arctan2(-z2, -x2)
    rot_3 = np.vstack(( [ np.cos(theta_3), 0, np.sin(theta_3)],
                        [ 0, 1, 0],
                        [-np.sin(theta_3), 0, np.cos(theta_3)]))

    x1, y1, z1 = _apply_rotation_matrix(rot_3, (x1,y1,z1))
    x2, y2, z2 = _apply_rotation_matrix(rot_3, (x2,y2,z2))

    # Now apply rotations to the grid 
    # and get cartesian coordinates at cell centers 
    xc, yc, zc = _rotate_the_grid(cds.XC, cds.YC, rot_1, rot_2, rot_3)

    # Interpolate for x,y to west and south edges
    grid = get_llc_grid(cds)
    xw = grid.interp(xc, 'X', boundary='fill')
    yw = grid.interp(yc, 'X', boundary='fill')
    xs = grid.interp(xc, 'Y', boundary='fill')
    ys = grid.interp(yc, 'Y', boundary='fill')

    # Compute the great circle mask, covering the entire globe
    maskC = scalar_calc.get_edge_mask(zc>0,grid) 
    maskW = grid.diff( 1*(zc>0), 'X', boundary='fill')
    maskS = grid.diff( 1*(zc>0), 'Y', boundary='fill')

    # Get section of mask pt1 -> pt2 only
    maskC = _calc_section_along_full_arc_mask(maskC, x1, y1, x2, y2, xc, yc)
    maskW = _calc_section_along_full_arc_mask(maskW, x1, y1, x2, y2, xw, yw)
    maskS = _calc_section_along_full_arc_mask(maskS, x1, y1, x2, y2, xs, ys)

    return maskC, maskW, maskS


# -------------------------------------------------------------------------------
#
# All functions below are non-user facing
#
# -------------------------------------------------------------------------------
# Helper functions for computing section masks 
# -------------------------------------------------------------------------------

def _calc_section_along_full_arc_mask( mask, x1, y1, x2, y2, xg, yg ):
    """Given a mask which has a great circle passing through 
    pt1 = (x1, y1) and pt2 = (x2,y2), grab the section just connecting pt1 and pt2

    Parameters
    ----------
    mask : xarray DataArray
        2D LLC mask with 1's along great circle across globe, crossing pt1 and pt2
    x1,y1,x2,y2 : scalars
        cartesian coordinates of rotated pt1 and pt2. Note that z1 = z2 = 0
    xg, yg : xarray DataArray
        cartesian coordinates of the rotated horizontal grid

    Returns
    -------
    mask : xarray DataArray
        mask with great arc passing from pt1 -> pt2
    """

    theta_1 = np.arctan2(y1,x1)
    theta_2 = np.arctan2(y2,x2)
    theta_g = np.arctan2(yg,xg)

    if theta_2 < 0:
        theta_g = theta_g.where( theta_g > theta_2, theta_g + 2*np.pi )
        theta_2 = theta_2 + 2 * np.pi

    if (theta_2 - theta_1) <= np.pi:
        mask = mask.where( (theta_g <= theta_2) & (theta_g >= theta_1), 0)
    else:
        mask = mask.where( (theta_g > theta_2) | (theta_g < theta_1), 0)

    return mask

def _rotate_the_grid(lon, lat, rot_1, rot_2, rot_3):
    """Rotate the horizontal grid at lon, lat, via rotation matrices rot_1/2/3

    Parameters
    ----------
    lon, lat : xarray DataArray
        giving longitude, latitude in degrees of LLC horizontal grid
    rot_1, rot_2, rot_3 : np.ndarray
        rotation matrices

    Returns
    -------
    xg, yg, zg : xarray DataArray
        cartesian coordinates of the horizontal grid
    """

    # Get cartesian of 1D view of lat/lon
    xg, yg, zg = _convert_latlon_to_cartesian(lon.values.ravel(),lat.values.ravel())

    # These rotations result in:
    #   xg = 0 at pt1
    #   yg = 1 at pt1
    #   zg = 0 at pt1 and pt2 (and the great circle that crosses pt1 & pt2)
    xg, yg, zg = _apply_rotation_matrix(rot_1, (xg,yg,zg))
    xg, yg, zg = _apply_rotation_matrix(rot_2, (xg,yg,zg))
    xg, yg, zg = _apply_rotation_matrix(rot_3, (xg,yg,zg))

    # Remake into LLC xarray DataArray
    xg = remake_llc_xda(xg,lon)
    yg = remake_llc_xda(yg,lat)
    zg = remake_llc_xda(zg,lon)

    return xg, yg, zg

def _apply_rotation_matrix(rot_mat,xyz):
    """Apply a rotation matrix to a tuple x,y,z (each x,y,z possibly being arrays)

    Parameters
    ----------
    rot_mat : numpy matrix
        2D matrix defining rotation in 3D cartesian coordinates
    xyz : tuple of arrays
        with cartesian coordinates

    Returns
    -------
    xyz_rot : tuple of arrays
        rotated a la rot_mat
    """

    # Put tuple into matrix form
    xyz_mat = np.vstack( (xyz[0],xyz[1],xyz[2]) )

    # Perform rotation
    xyz_rot_mat = np.matmul( rot_mat, xyz_mat )

    # Either return as scalar or array
    if np.isscalar(xyz[0]):
        return xyz_rot_mat[0,0], xyz_rot_mat[1,0], xyz_rot_mat[2,0]
    else:
        return xyz_rot_mat[0,:], xyz_rot_mat[1,:], xyz_rot_mat[2,:]


def _convert_latlon_to_cartesian(lon, lat):
    """Convert latitude, longitude (degrees) to cartesian coordinates
    Note: conversion to cartesian differs from what is found at e.g. Wolfram 
    because here lat \in [-pi/2, pi/2] with 0 at equator, not [0, pi], pi/2 at equator

    Parameters
    ----------
    lon : numpy or dask array
        longitude in degrees
    lat : numpy or dask array
        latitude in degrees

    Returns
    -------
    x : numpy or dask array
        x- component of cartesian coordinate
    y : numpy or dask array
    z : numpy or dask array
    """

    # Convert to radians
    lon_r = np.radians(lon)
    lat_r = np.radians(lat)

    # Get cartesian
    x = np.cos(lat_r)*np.cos(lon_r)
    y = np.cos(lat_r)*np.sin(lon_r)
    z = np.sin(lat_r)

    return x, y, z

# -------------------------------------------------------------------------------
# Helper functions for the computing volume, heat, and salt transport 
# -------------------------------------------------------------------------------

def _parse_section_trsp_inputs(ds,pt1,pt2,maskW,maskS,section_name):
    """Handle inputs for computing volume, heat, or salt transport across
    a section

    Parameters
    ----------
    see calc_vol_trsp

    Returns
    -------
    maskW, maskS : xarray DataArray
        masks defining the section
    """

    use_predefined_section = False
    use_endpoints = False
    use_masks = False

    # Test if section name is in available basins
    if section_name in get_available_sections():
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
        _, maskW, maskS = get_section_line_masks(pt1, pt2, ds)
    else:
        # Secondly, try to use endpoints or mask
        if use_endpoints and use_masks:
            raise TypeError('Cannot provide more than one method for defining section')
        elif use_endpoints:
            _, maskW, maskS = get_section_line_masks(pt1, pt2, ds)



    return maskW, maskS

def _initialize_section_trsp_data_array(cds):
    """Create an xarray DataArray with time, depth, and latitude dims

    Parameters
    ----------
    cds : xarray Dataset
        contains LLC coordinates 'k' and (optionally) 'time'

    Returns
    -------
    da : xarray DataArray
        zero-valued DataArray with time and depth dimensions
    """

    coords = OrderedDict()
    dims = ()

    if 'time' in cds:
        coords.update( {'time': cds['time'].values} )
        dims += ('time',)
        zeros = np.zeros((len(cds['time'].values),
                          len(cds['k'].values)))
    else:
        zeros = np.zeros((len(cds['k'].values)))

    coords.update( {'k': cds['k'].values} )

    dims += ('k',)

    return xr.DataArray(data=zeros, coords=coords, dims=dims)
