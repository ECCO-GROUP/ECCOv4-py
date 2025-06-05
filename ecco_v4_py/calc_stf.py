"""
Module for computing meridional overturning streamfunctions
TBD: add barotropic streamfunction
"""

import numpy as np

from .ecco_utils import get_llc_grid
from .calc_meridional_trsp import _parse_coords, meridional_trsp_at_depth
from .calc_section_trsp import _parse_section_trsp_inputs, section_trsp_at_depth

# Define constants
METERS_CUBED_TO_SVERDRUPS = 10**-6

def calc_meridional_stf_dens(ds, lat_vals, sig_levels,
                             basin_name=None):
    """
    Compute meridonal streamfunction at each density level defined in sig_levels
    across latitude(s) defined in lat_vals

    Parameters
    ----------
    ds : xarray DataSet
        must contain vars UVELMASS,VVELMASS, UVELSTAR, VVELSTAR, SIG2, drF, dyG, dxG
        and coords YC, (and maskW, and maskS if basin_name not None)
    lat_vals : float or list
        latitude value(s) specifying where to compute the streamfunction
    sig_levels : list of length 50
        Target values of Sigma_2 specifying the density bands for the 
        computation of the streamfunction
    basin_name : string, optional
        denote ocean basin over which to compute streamfunction
        If not specified, compute global quantity
        see get_basin.get_available_basin_names for options

    Returns
    -------
    xds : xarray Dataset
        with the main variable
            'psi'
                meridional overturning streamfunction across the section in Sv
                with dimensions 'time' (if in given dataset), 'lat', and 'SIGMA_levs'
    """
    #Set up vars
    #velocities
    U = ds.UVELMASS
    u = ds.UVELSTAR
    V = ds.VVELMASS
    v = ds.VVELSTAR
    utot = U+u
    vtot = V+v
    #spacial
    dx = ds.dxG
    dy = ds.dyG
    dz = ds.drF
    sig2 = ds.SIG2
    
    #regrid density
    grid = ecco.get_llc_grid(ds)
    
    sig2W = grid.interp(sig2, axis='X')
    sig2S = grid.interp(sig2, axis='Y')

    #basin mask?
    if basin_name is not None:
        maskS = ds.maskS.compute()
        maskW = ds.maskW.compute()
        basin_maskW = ecco.get_basin_mask(basin_name= basin_name,mask=maskW.isel(k=0))
        basin_maskS = ecco.get_basin_mask(basin_name= basin_name,mask=maskS.isel(k=0))
        ubasin = utot*basin_maskW
        vbasin = vtot*basin_maskS
        sigWbasin = sig2W*basin_maskW
        sigSbasin = sig2S*basin_maskS
        u = ubasin
        v = vbasin
        sigW = sigWbasin
        sigS = sigSbasin
    else:
        u = utot
        v = vtot
        sigW = sig2W
        sigS = sig2S

    #compute streamfunction everywhere, summing over all densities greater than target density
    xvol = xr.zeros_like(u)
    xvol = xvol.rename({'k':'sig2'})
    yvol = xr.zeros_like(v)
    yvol = yvol.rename({'k':'sig2'})
    for ss in range(50):
        sig = sig_levels[ss]
        y = v*dz*dx*np.heaviside(sigS-sig,1)*-1
        yy = y.sum(dim='k')
        x = utot*dz*dy*np.heaviside(sigW-sig,1)*-1
        xx = x.sum(dim='k')
        if any(x in ['time','month','year'] for x in list(U.dims)):
            yvol[:,ss,:] = yy
            xvol[:,ss,:] = xx
        else:
            yvol[ss,:] = yy
            xvol[ss,:] = xx
        
    #now compute meridional streamfunction
    # Initialize empty DataArray with coordinates and dims
    ones = xr.ones_like(ds.YC)
    
    lats_da = xr.DataArray(lat_vals,coords={'lat':lats},dims=('lat',))
    
    xda = xr.zeros_like(yvol['sig2']*lats_da)

    if 'time' in list(U.dims):
        xda = xda.broadcast_like(yvol['time']).copy()
    elif 'month' in list(U.dims):
        xda = xda.broadcast_like(yvol['month']).copy()
    elif 'year' in list(U.dims):
        xda = xda.broadcast_like(yvol['year']).copy()
    
    # Convert to dataset to add sigma2 coordinate
    xds = xda.to_dataset(name='psi')
    xds = xds.assign_coords({'SIGMA_levs':('sig2', sig_levels)})
    
    #cycle through all lats
    for l in range(len(lat_vals)):
        lat = lat_vals[l]
        dome_maskC = ones.where(ds.YC>=lat,0).compute()
        lat_maskW = grid.diff(dome_maskC,'X',boundary='fill') #multiply by x
        lat_maskS = grid.diff(dome_maskC,'Y',boundary='fill') #multiply by y
        ytrsp_lat = (yvol * lat_maskS).sum(dim=['i','j_g','tile'])
        xtrsp_lat = (xvol * lat_maskW).sum(dim=['i_g','j','tile'])
        xds['psi'].loc[{'lat':lat}] = ytrsp_lat+xtrsp_lat

    return xds
                               

def calc_meridional_stf(ds,lat_vals,doFlip=True,
                        basin_name=None,coords=None,grid=None):
    """Compute the meridional overturning streamfunction in Sverdrups
    at specified latitude(s)

    Parameters
    ----------
    ds : xarray DataSet
        must contain UVELMASS,VVELMASS, drF, dyG, dxG
    lat_vals : float or list
        latitude value(s) rounded to the nearest degree
        specifying where to compute overturning streamfunction
    doFlip : logical, optional
        True: integrate from "bottom" by flipping Z dimension before cumsum(),
        then multiply by -1. False: flip neither dim nor sign.
    basin_name : string, optional
        denote ocean basin over which to compute streamfunction
        If not specified, compute global quantity
        see utils.get_available_basin_names for options
    coords : xarray Dataset
        separate dataset containing the coordinate information
        YC, drF, dyG, dxG, optionally maskW, maskS
    grid : xgcm Grid object, optional
        denotes LLC90 operations for xgcm, see ecco_utils.get_llc_grid
        see also the [xgcm documentation](https://xgcm.readthedocs.io/en/latest/grid_topology.html)

    Returns
    -------
    ds_out : xarray Dataset
        with the following variables
            moc
                meridional overturning strength as maximum of streamfunction
                in depth space, with dimensions 'time' (if in dataset), and 'lat'
            psi_moc
                meridional overturning streamfunction across the section in Sv
                with dimensions 'time' (if in given dataset), 'lat', and 'k'
            trsp_z
                freshwater transport across section at each depth level in Sv
                with dimensions 'time' (if in given dataset), 'lat', and 'k'
    """

    # get coords
    coords = _parse_coords(ds,coords,['Z','YC','drF','dyG','dxG'])

    # Compute volume transport
    trsp_x = ds['UVELMASS'] * coords['drF'] * coords['dyG']
    trsp_y = ds['VVELMASS'] * coords['drF'] * coords['dxG']

    # Creates an empty streamfunction
    ds_out = meridional_trsp_at_depth(trsp_x, trsp_y,
                                      lat_vals=lat_vals,
                                      coords=coords,
                                      basin_name=basin_name,
                                      grid=grid)

    psi_moc = ds_out['trsp_z'].copy(deep=True)

    # Flip depth dimension, take cumulative sum, flip back
    if doFlip:
        psi_moc = psi_moc.isel(k=slice(None,None,-1))

    # Should this be done with a grid object???
    psi_moc = psi_moc.cumsum(dim='k')

    if doFlip:
        psi_moc = -1 * psi_moc.isel(k=slice(None,None,-1))

    # Add to dataset
    ds_out['psi_moc'] = psi_moc

    # Compute overturning strength
    ds_out['moc'] = ds_out['psi_moc'].max(dim='k')

    # Convert to Sverdrups
    for fld in ['trsp_z','psi_moc','moc']:
        ds_out[fld] = ds_out[fld] * METERS_CUBED_TO_SVERDRUPS
        ds_out[fld].attrs['units'] = 'Sv'

    # Name the fields here, after unit conversion which doesn't keep attrs
    ds_out['trsp_z'].attrs['name'] = 'volumetric trsp per depth level'
    ds_out['psi_moc'].attrs['name'] = 'meridional overturning streamfunction'
    ds_out['moc'].attrs['name'] = 'meridional overturning strength'

    return ds_out

def calc_section_stf(ds,
                     pt1=None, pt2=None,
                     section_name=None,
                     maskW=None, maskS=None,
                     doFlip=True,coords=None,grid=None):
    """Compute the overturning streamfunction in plane normal to
    section defined by pt1 and pt2 in depth space

    See calc_section_trsp.calc_section_vol_trsp for the various ways
    to call this function

    All inputs are the same except:

    Parameters
    ----------
    doFlip : logical, optional
        True: integrate from "bottom" by flipping Z dimension before cumsum(),
        then multiply by -1. False: flip neither dim nor sign.

    Returns
    -------
    ds_out : xarray Dataset
        with the following variables
            moc
                meridional overturning strength as maximum of streamfunction
                in depth space, with dimensions 'time' (if in dataset), and 'lat'
            psi_moc
                overturning streamfunction across the section in Sv
                with dimensions 'time' (if in given dataset), 'lat', and 'k'
            trsp_z
                freshwater transport across section at each depth level in Sv
                with dimensions 'time' (if in given dataset), 'lat', and 'k'
            maskW, maskS
                defining the section
        and the section_name as an attribute if it is provided
    """

    coords = _parse_coords(ds,coords,['Z','YC','XC','drF','dyG','dxG'])

    # Compute volume transport
    trsp_x = ds['UVELMASS'] * coords['drF'] * coords['dyG']
    trsp_y = ds['VVELMASS'] * coords['drF'] * coords['dxG']

    maskW, maskS = _parse_section_trsp_inputs(coords,pt1,pt2,maskW,maskS,section_name,
                                              grid=grid)

    # Creates an empty streamfunction
    ds_out = section_trsp_at_depth(trsp_x, trsp_y,
                                   maskW, maskS,
                                   coords=coords)

    psi_moc = ds_out['trsp_z'].copy(deep=True)

    # Flip depth dimension, take cumulative sum, flip back
    if doFlip:
        psi_moc = psi_moc.isel(k=slice(None,None,-1))

    # Should this be done with a grid object???
    psi_moc = psi_moc.cumsum(dim='k')

    if doFlip:
        psi_moc = -1 * psi_moc.isel(k=slice(None,None,-1))

    # Add to dataset
    ds_out['psi_moc'] = psi_moc

    # Compute overturning strength
    ds_out['moc'] = ds_out['psi_moc'].max(dim='k')

    # Convert to Sverdrups
    for fld in ['trsp_z','psi_moc','moc']:
        ds_out[fld] = ds_out[fld] * METERS_CUBED_TO_SVERDRUPS
        ds_out[fld].attrs['units'] = 'Sv'

    # Name the fields here, after unit conversion which doesn't keep attrs
    ds_out['trsp_z'].attrs['name'] = 'volumetric trsp per depth level'
    ds_out['psi_moc'].attrs['name'] = 'overturning streamfunction'
    ds_out['moc'].attrs['name'] = 'overturning strength'

    # Add section name and masks to Dataset
    ds_out['maskW'] = maskW
    ds_out['maskS'] = maskS
    if section_name is not None:
        ds_out.attrs['name'] = section_name

    return ds_out
