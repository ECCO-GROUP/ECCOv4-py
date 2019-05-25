"""
Module for computing meridional overturning streamfunctions
TBD: add barotropic streamfunction
"""

import numpy as np

from .ecco_utils import get_llc_grid
from .calc_meridional_trsp import meridional_trsp_at_depth
from .calc_section_trsp import _parse_section_trsp_inputs, section_trsp_at_depth

# Define constants
METERS_CUBED_TO_SVERDRUPS = 10**-6

def calc_meridional_stf(ds,lat_vals,doFlip=True,basin_name=None,grid=None):
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
    grid : xgcm Grid object, optional
        denotes LLC90 operations for xgcm, see ecco_utils.get_llc_grid
        see also the [xgcm documentation](https://xgcm.readthedocs.io/en/latest/grid_topology.html)

    Returns
    -------
    ds_out : xarray Dataset
        with the following variables
            psi_moc
                meridional overturning streamfunction across the section in Sv
                with dimensions 'time' (if in given dataset), 'lat', and 'k'
            trsp_z
                freshwater transport across section at each depth level in Sv
                with dimensions 'time' (if in given dataset), 'lat', and 'k'
    """

    # Compute volume transport
    trsp_x = ds['UVELMASS'] * ds['drF'] * ds['dyG']
    trsp_y = ds['VVELMASS'] * ds['drF'] * ds['dxG']

    # Creates an empty streamfunction
    ds_out = meridional_trsp_at_depth(trsp_x, trsp_y, 
                                      lat_vals=lat_vals, 
                                      cds=ds.coords.to_dataset(), 
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

    # Convert to Sverdrups
    for fld in ['trsp_z','psi_moc']:
        ds_out[fld] = ds_out[fld] * METERS_CUBED_TO_SVERDRUPS
        ds_out[fld].attrs['units'] = 'Sv'

    return psi_moc

def calc_section_stf(ds, 
                     pt1=None, pt2=None, 
                     section_name=None,
                     maskW=None, maskS=None,
                     doFlip=True,grid=None):
    """Compute the overturning streamfunction in plane normal to 
    section defined by pt1 and pt2 in depth space

    See calc_section_trsp.calc_section_vol_trsp for the various ways 
    to call this function 

    All inputs are the same except:

    Parameters
    ----------
    ds : xarray DataSet
        must contain UVELMASS,VVELMASS, drF, dyG, dxG

    Returns
    -------
    ds_out : xarray Dataset
        with the following variables
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

    # Compute volume transport
    trsp_x = ds['UVELMASS'] * ds['drF'] * ds['dyG']
    trsp_y = ds['VVELMASS'] * ds['drF'] * ds['dxG']

    maskW, maskS = _parse_section_trsp_inputs(ds,pt1,pt2,maskW,maskS,section_name)

    # Creates an empty streamfunction
    ds_out = section_trsp_at_depth(trsp_x, trsp_y,
                                    maskW, maskS, 
                                    cds=ds.coords.to_dataset(), 
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

    # Convert to Sverdrups
    for fld in ['trsp_z','psi_moc']:
        ds_out[fld] = ds_out[fld] * METERS_CUBED_TO_SVERDRUPS
        ds_out[fld].attrs['units'] = 'Sv'

    # Add section name and masks to Dataset
    ds_out['maskW'] = maskW
    ds_out['maskS'] = maskS
    if section_name is not None:
        ds_out.attrs['name'] = section_name

    return ds_out
