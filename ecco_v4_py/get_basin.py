"""
Module for getting masks which denote specific ocean basins
"""

from __future__ import division, print_function
import os
import os.path
import warnings
import numpy as np
import xarray as xr
import datetime
import dateutil
import xgcm

from .read_bin_llc import read_llc_to_tiles

# Store the package directory for loading the basins binary
package_directory = os.path.dirname(os.path.abspath(__file__))

def get_basin_mask(basin_name, mask, basin_path='../binary_data'):
    """Return mask for ocean basin.
    Note: This mirrors gcmfaces/ecco_v4/v4_basin.m

    Parameters
    ----------
    basin_name : string or list
        name of basin to include, options are any of the following

        atlExt - atl, mexico, hudson, med, north, baffin, gin
        pacExt - pac, bering, okhotsk, japan, eastChina
        indExt - ind, southChina, java, timor, red, gulf
        arct, barents

    mask : xarray DataArray
        2D or 3D mask for open ocean
        Note: can be at centers, west, or south face

    basin_path : string, default : '../binary_data'
        name of the directory that contains 'basins.data' and 'basins.meta'
        
        If you don't have basins.data or basins.meta in your 'binary_data' directory
        you can download them from:
            https://github.com/ECCO-GROUP/ECCOv4-py/tree/master/binary_data
        or 
            https://figshare.com/articles/Binary_files_for_the_ecco-v4-py_Python_package_/9932162

    Returns
    -------
    basin_mask : xarray DataArray
        mask with values at cell centers, 1's for denoted ocean basin
        dimensions are the same as input field
    """

    if type(basin_name) is not list:
        basin_name = [basin_name]

    # Look for extended basins
    basin_name = _append_extended_basins(basin_name)

    # Get available names
    available_names = get_available_basin_names()


    # Read binary with the masks, from gcmfaces package
    bin_dir = os.path.join(package_directory, basin_path)
   
    if os.path.exists(bin_dir + '/basins.data'):
        all_basins = read_llc_to_tiles(bin_dir,'basins.data')
    else:
        print ('Cannot find basins.data in ' + bin_dir + '\n')
        print ('You can download basins.data and basins.meta here:')
        print ('   https://github.com/ECCO-GROUP/ECCOv4-py/tree/master/binary_data')
        print (' or ')
        print ('   https://figshare.com/articles/Binary_files_for_the_ecco-v4-py_Python_package_/9932162')
        print ('\n')               
        print ('Download these files and specify their path when calling this subroutine')
        return []


    # Handle vertical coordinate
    # If input mask is 3D in space, first get mask on top level
    if 'k' in mask.dims:
        mask_2d = mask.isel(k=0)
    elif 'k_u' in mask.dims:
        mask_2d = mask.isel(k_u=0)
    elif 'k_l' in mask.dims:
        mask_2d = mask.isel(k_l=0)
    elif 'k_p1' in mask.dims:
        mask_2d = mask.isel(k_p1=0)
    else:
        mask_2d = mask

    basin_mask = 0*mask_2d

    for name in basin_name:
        if name in available_names:
            basin_mask = basin_mask + mask_2d.where(all_basins == (available_names.index(name)+1),0)
        else:
            warnings.warn('\nIgnoring %s, not an available basin mask.\n '
                          'Available basin mask names are: %s' % (name,available_names))

    # Now multiply by original mask to get vertical coordinate back
    # (xarray multiplication implies union of dimensions)
    basin_mask = basin_mask * mask

    return basin_mask

def get_available_basin_names():
    """Return available basins to get a mask for
    ORDER MATTERS! Order is associated with value in basins.data

    Returns
    -------
    available_names : list
        strings denoting various basins
    """

    available_names = ['pac',
                       'atl',
                       'ind',
                       'arct',
                       'bering',
                       'southChina',
                       'mexico',
                       'okhotsk',
                       'hudson',
                       'med',
                       'java',
                       'north',
                       'japan',
                       'timor',
                       'eastChina',
                       'red',
                       'gulf',
                       'baffin',
                       'gin',
                       'barents']

    return available_names

def _append_extended_basins(basin_list):
    """Replace extended basins with components, e.g. atlExt with atl, mexico ...
    Note: atlExt etc are removed from the list for error checking later on.

    Parameters
    ----------
    basin_list : list of strings
        list of basin names potentially including 'atlExt', 'pacExt', 'indExt'

    Returns
    -------
    basin_list : list of strings
        list of basin names with "Ext" version replaced with "sub"basins
    """

    for name in basin_list:
        if name == 'atlExt':
            basin_list.remove('atlExt')
            basin_list.append('atl')
            basin_list.append('mexico')
            basin_list.append('hudson')
            basin_list.append('med')
            basin_list.append('north')
            basin_list.append('baffin')
            basin_list.append('gin')
        elif name == 'pacExt':
            basin_list.remove('pacExt')
            basin_list.append('pac')
            basin_list.append('bering')
            basin_list.append('okhotsk')
            basin_list.append('japan')
            basin_list.append('eastChina')
        elif name == 'indExt':
            basin_list.remove('indExt')
            basin_list.append('ind')
            basin_list.append('southChina')
            basin_list.append('java')
            basin_list.append('timor')
            basin_list.append('red')
            basin_list.append('gulf')

    return basin_list
