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
        Something like : ECCOv4r3_grid.nc or ECCOv4r4_grid.nc
        
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
 
    if len(grid_filename) == 0:
        files = np.sort(glob.glob(grid_dir + '/*nc'))
    else:
        files = np.sort(glob.glob(grid_dir + '/' + grid_filename))    
        
        
    if len(files) == 1:
        file = files[0]
        g = []
    
        if not less_output:
            print ('--- LOADING model grid file: %s' % files[0])

        if dask_chunk:
            g_i = xr.open_dataset(file, chunks=90)
        else:
            g_i = xr.open_dataset(file).load()

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

    # no files
    else:
        print('\n\n Attention!!')
        print('ECCO netcdf grid file not found in ' + grid_dir)
        print('Consider passing the grid file directory and grid filename as arguments')
        return []
    
    return g_i



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
 
    files = np.sort(glob.glob(data_dir + '/' + var_to_load + '*nc'))

    g = []
    
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

    if len(files) > 0:
        if not less_output:
            print ('---found %s nc files here.  loading ....' % var_to_load)
       
        for file in files:
            file_year = int(str.split(file,'.nc')[0][-4:])
            var_name_of_file = re.split(r"_\d+",str.split(file,'/')[-1])[0]
            
            if var_name_of_file == var_to_load:
                
                if 'all' in years_to_load or file_year in years_to_load:
                     
                    if dask_chunk:
                        g_i = xr.open_dataset(file, chunks=90)
                    else:
                        g_i = xr.open_dataset(file).load()

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
        
    # ecco_dataset to return
    g = []

    #if 'all' in vars_to_load:
    var_head_dirs = np.sort(glob.glob(data_root_dir + '/*'))
    var_names = []
      
    # search for variable names in data_root_dir
    if not less_output:
        print ('searching for variables in %s ' % data_root_dir)
    
    for var in var_head_dirs:
        var_names.append(str.split(var,'/')[-1])
       
    print ('searching %s for variables ... ' % data_root_dir)

    if len(var_names) == 0:
       print ('no variables found in %s' % data_root_dir)
    else:
       print ('found  %s \n' %  str(var_names))

    
    if not isinstance(years_to_load, list):
        years_to_load = [years_to_load]    
    
    # loop through the variable names found here.
    for var_to_load in var_names:
        g_var = []

        if 'all' in vars_to_load or var_to_load in vars_to_load:
            if not less_output:
                print ('trying to load %s ' % var_to_load)
                
            sub_dirs = np.sort(([[x[0] for x in os.walk(data_root_dir)]]))

            
            # loop through subdirectories look for var_to_load
            for sub_dir in sub_dirs[0]:
                if not less_output :
                    print ('searching for %s in %s ' % (var_to_load, sub_dir))
                
                g_i = load_ecco_var_from_years_nc(sub_dir, var_to_load, 
                                             years_to_load = years_to_load,
                                             tiles_to_load = tiles_to_load,
                                             k_subset =  k_subset,
                                             dask_chunk = dask_chunk,
                                             less_output =less_output)

                if len(g_i) > 0:
                    if isinstance(g_var, list):
                        g_var = g_i
                    else:
                        g_var = xr.concat((g_var,g_i),'time')
                
            print 
            
            # finshed looping through all dirs 
            if len(g_var) == 0:
                print ('finished searching for %s ... not found!' % var_to_load) 
            else:
                print ('finished searching for %s ... success!' % var_to_load) 
                
        
        # if we loaded var_to_load, then add it to g
        if len(g_var) > 0:
            # if g is [], make g = g_var
            if len(g) == 0 :
                g = g_var
            
            # otherwise merge
            else:
                g = xr.merge((g_var,g))
    # add some metadata
    if len(g) > 0:
        g = update_ecco_dataset_geospatial_metadata(g)
        g = update_ecco_dataset_temporal_coverage_metadata(g)
        
    return g
        
