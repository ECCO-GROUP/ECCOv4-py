"""
ECCO v4 Python: tile_io

This module provides routines for loading ECCO netcdf files.
--- now with actual documentation!

.. _ecco_v4_py Documentation :
   https://github.com/ECCO-GROUP/ECCOv4-py

"""
from __future__ import division,print_function
import numpy as np
import xarray as xr
import glob
import os
import re
from pathlib import Path
from collections import OrderedDict 
 
from .netcdf_product_generation import update_ecco_dataset_geospatial_metadata
from .netcdf_product_generation import update_ecco_dataset_temporal_coverage_metadata


#%%
def load_ecco_grid_nc(grid_dir, grid_filename=[], \
                      tiles_to_load = 'all', \
                      k_subset = [], \
                      dask_chunk = False,\
                      less_output = True):
    
    """

    Loads the ECCOv4 NetCDF model grid parameters.
    All 13 tiles of the lat-lon-cap (llc) grid are present in this 
    type of file.  A subset of the vertical level (k_subset) and the 
    model grid tiles (tiles_to_load) are optional arguments.

    Simply pass the directory with the model grid file 

    Parameters
    ----------
    grid_dir : str
        path to a directory within which we will look for NetCDF grid file

    grid_filename : str
        name of model grid file
        filename should be something like : ECCOv4r3_grid.nc or ECCOv4r4_grid.nc
        
    tiles_to_load : int or list or range, optional, default range(13)
        a list of which tiles to load.  
    
    k_subset : list, optional, default = [] (load all)
        a list of which vertical levels to load. 
                
    dask_chunk : boolean, optional, default True
        whether or not to ask Dask to chunk the arrays into pieces 90x90.  
        WARNING: DON'T MESS WITH THIS NUMBER.

    less_output : boolean, default False
        A debugging flag.  False = less debugging output
        
    Returns
    -------
    g : Dataset
        an xarray Dataset

    """
    if not less_output:
        print ('tiles_to_load ', tiles_to_load)

    if isinstance(tiles_to_load,str):
        if 'all' in tiles_to_load:
            tiles_to_load = [0,1,2,3,4,5,6,7,8,9,10,11,12]
        else:
            tiles_to_load = list(int(tiles_to_load))
    elif isinstance(tiles_to_load,tuple):
        tiles_to_load = list(tiles_to_load)
    elif isinstance(tiles_to_load, int):
        tiles_to_load = [tiles_to_load]
    elif isinstance(tiles_to_load, range):
        tiles_to_load = list(tiles_to_load)
    elif not isinstance(tiles_to_load, list):
        raise Exception ('tiles_to_load has to be a tuple, int, list or string ' + \
                         'you passed a %s and I cannot handle that' % type(tiles_to_load))

    if not less_output:
        print(grid_filename, grid_dir)
 
    file = Path(grid_dir) / grid_filename

    if file.exists():
    
        if not less_output:
            print ('--- LOADING model grid file: %s' % file)

        if dask_chunk:
            g_i = xr.open_dataset(str(file), chunks=90)
        else:
            g_i = xr.open_dataset(str(file)).load()

        # pull out a k subset
        if len(k_subset) > 0 :
            if 'k' in g_i.coords.keys():
                g_i = g_i.isel(k=k_subset)

            if 'k_u' in g_i.coords.keys():
                g_i = g_i.isel(k_u=k_subset)

            if 'k_l' in g_i.coords.keys():
                g_i = g_i.isel(k_l=k_subset)

            if 'k_p1' in g_i.coords.keys():
                g_i = g_i.isel(k_p1=k_subset)


        # pull out the tile subset
        g_i = g_i.isel(tile=tiles_to_load)

        
        # update some metadata for fun.
        g_i = update_ecco_dataset_geospatial_metadata(g_i)
        g_i = update_ecco_dataset_temporal_coverage_metadata(g_i)
        g_i.attrs = OrderedDict(sorted(g_i.attrs.items()))

        return g_i

    # no files
    else:
        print('\n\n Attention!!')
        print('ECCO netcdf grid file not found in ' + str(file))
        return []
    

#%%
def load_ecco_var_from_years_nc(data_dir, var_to_load, \
                                years_to_load = 'all', \
                                tiles_to_load = 'all', \
                                k_subset = [], \
                                dask_chunk = True,\
                                less_output = True):
    
    """

    Loads one or more ECCOv4 NetCDF state estimate variable in the
    format of one file per variable per year.  All 13 tiles of the 
    lat-lon-cap (llc) grid are present in this type of file.

    Files in this format have names like 
        /eccov4r3_native_grid_netcdf/mon_mean/THETA_2010.nc
        /eccov4r3_native_grid_netcdf/day_mean/THETA_2010.nc
        /eccov4r3_native_grid_netcdf/mon_snapshot/THETA_2010.nc

    Simply point this routine at a directory with one or more 
    of these files and one or more years of var_to_load variables will
    be loaded. 
    * Used repeatedly by recursive_load_ecco_var_from_years_nc *


    Parameters
    ----------
    data_dir : str
        path to a directory within which we will look for NetCDF tile files

    tiles_to_load : int or list or range, optional, default range(13)
        a list of which tiles to load.  
    
    var_to_load : str
        string indicating which variable to load.
        
    years_to_load : str, int or list, optional, default 'all'
        a list of which years to load.  
    
    k_subset : list, optional, default = [] (load all)
        a list of which vertical levels to load. 
                
    dask_chunk : boolean, optional, default True
        whether or not to ask Dask to chunk the arrays into pieces 90x90.  
        WARNING: DON'T MESS WITH THIS NUMBER.

    less_output : boolean, default False
        A debugging flag.  False = less debugging output
        
    Returns
    -------
    g : Dataset
        an xarray Dataset

    """
    if not less_output:
        print ('tiles_to_load ', tiles_to_load)
        print ('years to load ', years_to_load)

    if isinstance(tiles_to_load,str):
        if 'all' in tiles_to_load:
            tiles_to_load = [0,1,2,3,4,5,6,7,8,9,10,11,12]
        else:
            tiles_to_load = list(int(tiles_to_load))
    elif isinstance(tiles_to_load,tuple):
        tiles_to_load = list(tiles_to_load)
    elif isinstance(tiles_to_load, int):
        tiles_to_load = [tiles_to_load]
    elif isinstance(tiles_to_load, range):
        tiles_to_load = list(tiles_to_load)
    elif not isinstance(tiles_to_load, list):
        raise Exception ('tiles_to_load has to be a tuple, int, list or string ' + \
                         'you passed a %s and I cannot handle that' % type(tiles_to_load))
 

    var_path = Path(data_dir) / var_to_load
    files = list(var_path.glob('**/*nc'))
    

    
    if not less_output:
        print ('--- LOADING %s FROM YEARS NC: %s' % \
               (var_to_load, data_dir))
    
    if isinstance(years_to_load, str):
        if 'all' not in years_to_load:        
            years_to_load = int(years_to_load)
    elif isinstance(years_to_load,tuple):
        years_to_load = list(years_to_load)
    elif isinstance(years_to_load, int):
        years_to_load = [years_to_load]
    elif isinstance(years_to_load, str):
        years_to_load = int(years_to_load)
    elif isinstance(years_to_load, range):
        years_to_load = list(years_to_load)
    elif not isinstance(years_to_load, list):
        raise Exception ('years_to_load has to be a tuple, int, list or string ' + \
                         'you passed a %s and I cannot handle that' % type(years_to_load))

    # g will be the DataArray for this variable and all times
    g = []
    if len(files) > 0:
        if not less_output:
            print ('---found %s nc files here.  loading ....' % var_to_load)
       
        for file in files:
            file_basename =  file.stem
            if not less_output:
                print('file basename : ', file_basename)
                
            # assumes format is VARNAME_YYYY_other stuff.nc
            file_year = int(file.stem.split(sep='_')[1])
            var_name_of_file = file.stem.split(sep='_')[0]
            
            if var_name_of_file == var_to_load:
                if not less_output:
                    print('var name matches ', var_name_of_file)
                    
                if 'all' in years_to_load or file_year in years_to_load:
                    if not less_output:
                        print ('year in years_to_load', file_year) 
                   
                    if dask_chunk:
                        g_i = xr.open_dataset(str(file), chunks=90)
                    else:
                        g_i = xr.open_dataset(str(file)).load()

                    # pull out a k subset
                    if len(k_subset) > 0 and 'k' in g_i.coords.keys():
                        g_i = g_i.isel(k=k_subset)
    
                    # pull out the tile subset
                    g_i = g_i.isel(tile=tiles_to_load)

                    
                    if isinstance(g, list):
                        g = g_i
                    else:
                        g = xr.concat((g, g_i),'time')
        

        # finished looping through files
        if len(g) == 0:
            if not less_output:
                print ('we had files but did not load any matching %s ' % \
                       var_to_load)
        
        else:
            # update some metadata for fun.
            g = update_ecco_dataset_geospatial_metadata(g)
            g = update_ecco_dataset_temporal_coverage_metadata(g)
            g.attrs = OrderedDict(sorted(g.attrs.items()))

    # no files
    else:
        if not less_output:
            print ('no files found with name "%s" in %s \n' % \
                   (var_to_load, data_dir ))
        
    return g

#%%
def recursive_load_ecco_var_from_years_nc(data_root_dir, 
                                          vars_to_load = 'all',
                                          tiles_to_load = 'all',
                                          years_to_load = 'all',
                                          k_subset = [],
                                          dask_chunk = True,
                                          less_output = True):
    
    """

    Loads one or more state estimate variables for one or more 
    years.  Appropriate for ECCOv4 NetCDF files stored in the
    format of one file per variable per year.  All 13 tiles of the 
    lat-lon-cap (llc) grid are present in this type of file.

    Files in this format have names like 
        /eccov4r3_native_grid_netcdf/mon_mean/THETA_2010.nc
        /eccov4r3_native_grid_netcdf/day_mean/THETA_2010.nc
        /eccov4r3_native_grid_netcdf/mon_snapshot/THETA_2010.nc

    Simply point this routine at a top-level directory, a list of
    variables you want, and a list of years and prepare to be dazzled.

    * Makes heavy use of load_ecco_var_from_years_nc *


    Parameters
    ----------
    data_root_dir : str
        path to a top-level directory below we will look for NetCDF tile files

    vars_to_load : list or str, optional, default 'all' 
        a list or string indicating which variables you want to load.
        
        Note: if 'all', data_root_dir must be a directory with one or more
        variable names. In the follow example, THETA, SALT, and ETAN
        will be loaded if the full path to `eccov4_native_grid_netcdf` is 
        provided:
            /eccov4r3_native_grid_netcdf/THETA/THETA_YYYY.nc 
            /eccov4r3_native_grid_netcdf/SALT/SALT_YYYY.nc 
            /eccov4r3_native_grid_netcdf/ETAN/ETAN_YYYY.nc 

    tiles_to_load : int or list or range, optional, default range(13)
        a list of which tiles to load.  
    
    years_to_load : int or list or range, optional, default 'all'
        a list of which tiles to load
        
    k_subset : list, optional, default = [] (load all)
        a list of which vertical levels to load. 
                
    dask_chunk : boolean, optional, default True
        whether or not to ask Dask to chunk the arrays into pieces 90x90.  
        WARNING: DON'T MESS WITH THIS NUMBER.

    less_output : boolean, default False
        A debugging flag.  False = less debugging output
        
    Returns
    -------
    g : Dataset
        an xarray Dataset

    """

    if not less_output:
        print ('tiles_to_load ', tiles_to_load)
        print ('years to load ', years_to_load)

    if isinstance(tiles_to_load,str):
        if 'all' in tiles_to_load:
            tiles_to_load = [0,1,2,3,4,5,6,7,8,9,10,11,12]
        else:
            tiles_to_load = list(int(tiles_to_load))
    elif isinstance(tiles_to_load,tuple):
        tiles_to_load = list(tiles_to_load)
    elif isinstance(tiles_to_load, int):
        tiles_to_load = [tiles_to_load]
    elif isinstance(tiles_to_load, range):
        tiles_to_load = list(tiles_to_load)
    elif not isinstance(tiles_to_load, list):
        raise Exception ('tiles_to_load has to be a tuple, int, list or string ' + \
                         'you passed a %s and I cannot handle that' % str(type(tiles_to_load)))

    if isinstance(years_to_load, str):
        if 'all' not in years_to_load:        
            years_to_load = int(years_to_load)
    elif isinstance(years_to_load,tuple):
        years_to_load = list(years_to_load)
    elif isinstance(years_to_load, int):
        years_to_load = [years_to_load]
    elif isinstance(years_to_load, str):
        years_to_load = int(years_to_load)
    elif isinstance(years_to_load, range):
        years_to_load = list(years_to_load)
    elif not isinstance(years_to_load, list):
        raise Exception ('years_to_load has to be a tuple, int, list or string ' + \
                         'you passed a %s and I cannot handle that' % type(years_to_load))
    
    if not isinstance(vars_to_load, list):
        vars_to_load = [vars_to_load]
        

    var_path = Path(data_root_dir)
    
    files = list(var_path.glob('**/*nc'))

    all_var_names = []
    
    files_to_load_dict = dict()
    
    if len(files) == 0:
        print ('no netcdf files found in this directory or subdirectories')
        return []
    
    # loop through all netcdf files found in subdirectories of 'data_root_dir'
    for file in files:
        if not less_output:
            print('file basename : ', file_stem)
            
        # assumes format is VARNAME_YYYY_other stuff.nc
        file_year = int(file.stem.split(sep='_')[1])
        var_name_of_file = file.stem.split(sep='_')[0]
        
        if not var_name_of_file in all_var_names:
            all_var_names.append(var_name_of_file)
            
        if 'all' in vars_to_load or var_name_of_file in vars_to_load:
            if 'all' in years_to_load or file_year in years_to_load:
                
                # add this file to a new list of files to read 
                if var_name_of_file not in files_to_load_dict.keys():
                    files_to_load_dict[var_name_of_file] = [file]
                else:
                    # add this file to the existing list of files to read
                    files_to_load_dict[var_name_of_file].append(file)

    # g_all will be a dataset containing all data arrays
    g_all = []
    
    # loop through all variables that we are going to load
    for var_to_load in files_to_load_dict.keys():
        
        # g is a data array that will contain all time levels
        # for this variable
        g = []
        print('loading files of ', var_to_load)
        
        # get all filenames corresponding to this variable to load
        files_for_var = files_to_load_dict[var_to_load]
        
        # loop through all of those variables
        for file in files_for_var:
         
            if not less_output:
                print('loading ', str(file))
                
            # g_i is a data arary for one time level of this variable
            if dask_chunk:
                g_i = xr.open_dataset(str(file), chunks=90)
            else:
                g_i = xr.open_dataset(str(file)).load()

            # pull out a k subset
            if len(k_subset) > 0 and 'k' in g_i.coords.keys():
                g_i = g_i.isel(k=k_subset)

            # pull out the tile subset
            g_i = g_i.isel(tile=tiles_to_load)

            # combine this time level (g_i) with the data array that contains
            # all time levels (g)
            if isinstance(g, list):
                g = g_i
            else:
                g = xr.concat((g, g_i),'time')
        

        # finished looping through files, check to make sure we have something
        # in g
        if len(g) == 0:
            if not less_output:
                print ('we had files but did not load any matching %s ' % \
                       var_to_load)
        
        else:
            # update some metadata for fun.
            g = update_ecco_dataset_geospatial_metadata(g)
            g = update_ecco_dataset_temporal_coverage_metadata(g)
     
            # Add this DataArray to the DataSet g_all
            # -- if this is the first DataArray, then make g_all, g
            if len(g_all) == 0:
                g_all = g
            
            # otherwise merge g into g_all
            else:
                g_all = xr.merge((g_all,g))
                
                # when you merge DataArrays or DataSets some metadata can
                # be lost.  This bit ensures that all the metadata is carried
                # over to g_all
                for g_attr_key in g.attrs.keys():
                    if g_attr_key not in g_all.attrs.keys():
                        g_all.attrs[g_attr_key] = g.attrs[g_attr_key]
    
    # finished loading all variable DataArrays
    # update metadata again
    if len(g_all) > 0:
        g_all = update_ecco_dataset_geospatial_metadata(g_all)
        g_all = update_ecco_dataset_temporal_coverage_metadata(g_all)
        g_all.attrs = OrderedDict(sorted(g_all.attrs.items()))
    else:
        print('we ended up with nothing loaded!')
        
    return g_all
        
