"""
Module for computing meridional volume transport in density coordinates
"""

import numpy as np
import xarray as xr

from .get_basin import get_basin_mask
from .ecco_utils import get_llc_grid

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
        and coords YC, (maskW, and maskS if basin_name not None)
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
    
    #regrid
    grid = get_llc_grid(ds)
    
    sig2W = grid.interp(sig2, axis='X')
    sig2S = grid.interp(sig2, axis='Y')

    #basin mask?
    if basin_name is not None:
        maskS = ds.maskS.compute()
        maskW = ds.maskW.compute()
        basin_maskW = get_basin_mask(basin_name= basin_name,mask=maskW.isel(k=0))
        basin_maskS = get_basin_mask(basin_name= basin_name,mask=maskS.isel(k=0))
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
        x = u*dz*dy*np.heaviside(sigW-sig,1)*-1
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
        xds['psi'].loc[{'lat':lat}] = (ytrsp_lat+xtrsp_lat)*METERS_CUBED_TO_SVERDRUPS

    return xds
