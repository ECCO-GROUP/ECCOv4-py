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
import sys

from .netcdf_product_generation import update_ecco_dataset_geospatial_metadata
from .netcdf_product_generation import update_ecco_dataset_temporal_coverage_metadata

#%%
def load_ecco_grid_from_tiles_nc(grid_dir, 
                                 grid_base_name = 'ECCOv4r3_grid_tile_',
                                 tiles_to_load='all',
                                 k_subset = [], 
                                 dask_chunk = True, 
                                 coords_as_vars = False,
                                 less_output = True):
        
    """

    Loads 1-13 NetCDF tile files of the lat-lon-cap (LLC) grid used in ECCOv4.

    
    Parameters
    ----------
    grid_dir : str
        path to the NetCDF tile files

    grid_base_name : str, optional, default 'ECCOv4r3_grid_tile_' 
        the prefix string for the tile files.  Tile numbers are in the format 00, 01, .. 12
  
    tiles_to_load : str, int, list, or tuple, or range, optional, default range(13)
        a list of which tiles to load.  If not a list, code will try to make it a list
        
    k_subset : list, optional, default = [] (load all)
        a list of which vertical levels to load. 
                
    cmin/cmax : floats, optional, default calculate using the min/max of the data
        the minimum and maximum values to use for the colormap
        
    dask_chunk : boolean, optional, default True
        whether or not to ask Dask to chunk the arrays into pieces 90x90.  
        WARNING: DON'T MESS WITH THIS NUMBER.

    coords_as_vars : boolean, optional, deafult False
        boolean - do you want the non-dimension coords to actually be considered variables?
        your call.  Default is no.
        
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
    g = []
    
    for i in tiles_to_load:
        
        if dask_chunk:
            g_i = xr.open_dataset(grid_dir + '/' + \
                              grid_base_name + \
                              str(i).zfill(2) + '.nc', chunks = 90)
        else:
            g_i = xr.open_dataset(grid_dir + '/' + \
                              grid_base_name + \
                              str(i).zfill(2) + '.nc').load()

        if len(k_subset) > 0:
            g_i = g_i.isel(k=k_subset)
            
      
        if isinstance(g, list):
            g = g_i
            
        else:     
            g = xr.concat((g, g_i),'tile')
    
    if coords_as_vars:
        g.reset_coords()
            
    return g  


#%%

def recursive_load_ecco_var_from_tiles_nc(data_root_dir, 
                                          vars_to_load = 'all',
                                          tiles_to_load = 'all',
                                          years_to_load = 'all',
                                          k_subset = [],
                                          dask_chunk = True,
                                          less_output = True):
    
    """

    Loads one or more state estimate variables for one or more 
    years.  Appropriate for ECCOv4 NetCDF files stored as *tiles*
    * Makes use of load_ecco_var_from_tiles_nc *


    Parameters
    ----------
    data_root_dir : str
        path to a top-level directory below we will look for NetCDF tile files

    vars_to_load : list or str, optional, default 'all' 
        a list or string indicating which variables you want to load.
        
        Note: if 'all', data_root_dir must be a directory with one or more
        variable names.  e.g.,  `eccov4_native_grid_netcdf_tiles` in the
        below example
            /eccov4r3_native_grid_netcdf_tiles/THETA/2010
            /eccov4r3_native_grid_netcdf_tiles/SALT/2010
            /eccov4r3_native_grid_netcdf_tiles/ETAN/2010
        when pointed to eccov4r3_native_grid_netcdf_tiles in the above
        the variables THETA, SALT, and ETAN will be loaded.

    tiles_to_load : int or list or range, optional, default 'all'
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

    g = []

    if not less_output:
        print ('tiles_to_load ', tiles_to_load)
        print ('years to load ', years_to_load)
        print ('vars  to load ', vars_to_load)

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
 
    if not isinstance(vars_to_load, list):
        vars_to_load = [vars_to_load]
    
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
    
    if not isinstance(k_subset, list):
        k_subset = [k_subset]
        
    if 'all' in vars_to_load:
        var_head_dirs = np.sort(glob.glob(data_root_dir + '/*/'))
        var_names = []
       
        for var in var_head_dirs:
           var_names.append(str.split(var,'/')[-2])
       
        if not less_output:
            print ('searching for variables in ', data_root_dir)

        if not less_output:
            if len(var_names) == 0:
                print ('no variable names as top level directories in  %s' % \
                      data_root_dir)
            else:
               print ('variables found: %s ' % str(var_names))
           
        vars_to_load = var_names
        
    if not less_output:
        print ('vars to load ', vars_to_load)

    #%%
    
    dir_contents = next(os.walk(data_root_dir))
    # if we happen to have files in this directory
    if len(dir_contents[1]) == 0:
        print ('there are no subdirectories here.  ' + \
               'use load_ecco_var_from_tiles_nc to load tiles from a ' + \
               'directory of files of format VARNAME_YYYY_MM_DD_tile_NN.nc' + \
               'or VARNAME_YYYY_MM_tile_NN.nc')
    else:
        top_level_dirs = np.sort(dir_contents[1])
        
    top_level_dirs_with_var = dict()
    for var_to_load in vars_to_load:
        top_level_dirs_with_var[var_to_load] = []
    
    for top_level_dir in top_level_dirs:
#        if not less_output:
        print ('searching %s for variables ' % top_level_dir)
        
        keep_looking = True
        next_dir = data_root_dir + '/' + top_level_dir
        
        while keep_looking:
            if not less_output:
                print (next_dir)
            x = next(os.walk(next_dir))

            if len(x[2]) == 0 and len(x[1]) != 0:
                next_dir = x[0] + '/' + x[1][0]
            elif len(x[2]) > 0:
                keep_looking = False
                test_var_to_load = x[2][0]
                var_to_load_here = re.split(r"_\d+",str.split(test_var_to_load,'/')[-1])[0]
                if var_to_load_here in vars_to_load:
                    top_level_dirs_with_var[var_to_load_here].append(data_root_dir + '/' + top_level_dir + '/')
    
    if not less_output:
        print(top_level_dirs_with_var)
   
    #%%
        
      
    for var_to_load in vars_to_load:
        
        g_var = []

        #if not less_output:
        dirs_with_var = top_level_dirs_with_var[var_to_load]
        
        if len(dirs_with_var) > 0:
            print ('located directories with %s ' % var_to_load)
            
            for dir_with_var in dirs_with_var:
                if not less_output:
                    print ('searching %s ' % dir_with_var)

                sub_dirs = np.sort(([[x[0] for x in os.walk(dir_with_var)]]))[0]
                num_sub_dirs = len(sub_dirs)

                for idxs, sub_dir in enumerate(sub_dirs):
                 
                    if not less_output :
                        print (idxs / num_sub_dirs)
                    
                    files = np.sort(glob.glob(sub_dir + '/' + var_to_load + '*nc'))
                    
                    if len(files) > 0:

                        # by default, we'll load this directory of 
                        # netcdf tile files. 
                        load_this_directory = True
                        
                        # however, if the user specified a set of 
                        # years to load then first check to make sure that
                        # the year of these tile files is in the list...
                        if 'all' not in years_to_load:
                            # extract the first file
                            file = files[0]
                            # extract the year from the file name
                            # (regular expression match 4 consecutive 
                            # numbers) YYYY
                            file_year =  int(re.findall(r"[0-9]{4}", file)[0])
                                    
                            if file_year not in years_to_load:
                                load_this_directory = False

                        if load_this_directory:
                            if not less_output:
                                print ('found files here .. loading %s ' % str(files))
                            
                            g_i = load_ecco_var_from_tiles_nc(sub_dir, 
                                                              var_to_load, 
                                                              tiles_to_load=tiles_to_load,
                                                              k_subset=k_subset, 
                                                              dask_chunk=dask_chunk, 
                                                              less_output=less_output)
                            if isinstance(g_var, list):
                                g_var = g_i
                            else:
                                g_var = xr.concat((g_var, g_i),'time')
                        else:
                            if not less_output:
                                print ('skipping files from year %s ' \
                                       % file_year)
                            
        else:
            print ('no subdirectories with %s ' % var_to_load)          
        
        # if we loaded var_to_load, then add it to g
        if len(g_var) > 0:
            # if g is [], make g = g_var
            if len(g) == 0 :
                g = g_var
            
            # otherwise merge
            else:
                g = xr.merge((g_var, g))

    # add some metadata
    if len(g) > 0:
        g = update_ecco_dataset_geospatial_metadata(g)
        g = update_ecco_dataset_temporal_coverage_metadata(g)
        
    return g


#%%
def load_ecco_var_from_tiles_nc(data_dir, 
                                var_to_load,
                                tiles_to_load = 'all', 
                                k_subset = [], 
                                dask_chunk = True,
                                less_output = True):
    
        
    """

    Loads a single ECCOv4 NetCDF state estimate variable stored 
    in a directory of 13 tiles, one for each of the tiles of the 
    lat-lon-cap (llc) grid.  

    Files in this format have names like 
        /eccov4r3_native_grid_netcdf_tiles/mon_mean/2010/01/THETA_2010_01_tile_01.nc
        /eccov4r3_native_grid_netcdf_tiles/day_mean/2010/001/THETA_2010_01_01_tile_01.nc
        /eccov4r3_native_grid_netcdf_tiles/mon_snapshot/2010/01/THETA_2010_01_tile_01.nc

    Simply point this routine at a directory with tile files in it
    and one or more tiles will be loaded and concatenated.

    * Used repeatedly by recursive_load_ecco_var_from_tiles_nc *


    Parameters
    ----------
    data_dir : str
        path to a directory within which we will look for NetCDF tile files

    var_to_load : str
        string indicating which variable to load.
        
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
        print ('var   to_load ', var_to_load)

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
    
    g = []    
    files = np.sort(glob.glob(data_dir + '/' + var_to_load + '*nc'))

    if not less_output:
        print('files -- ')
        print(files)
    
    if not less_output:
        print('tiles to load -- ')
        print(tiles_to_load)

    for file in files:
        
        tile = int(file[-5:-3])
        var_name_of_file = re.split(r"_\d+",str.split(file,'/')[-1])[0]

        if var_to_load == var_name_of_file:
            if tile in tiles_to_load:
                if not less_output:
                    print ('loading tile %d' % tile)
    
                if dask_chunk:
                    g_i = xr.open_dataset(file, chunks = 90)

                    #print ('chunking')
                else:
                    g_i = xr.open_dataset(file).load()
                    #nt = len(g_i.time)
                    #g_i = g_i.chunk(nx, nx)
    
                if len(k_subset) > 0 and \
                    'k' in g_i.coords.keys():
                    g_i = g_i.isel(k=k_subset)
                                    
    
                if len(g) == 0:
                    g = g_i
                else:
                    g = xr.concat((g, g_i ),'tile')
                    
            elif not less_output:
                print ('not loading %s tile %d ' % (var_to_load, tile))
        else:
            print ('filename mismatch - trying to load %s and found %s ' % \
                  (var_to_load, var_name_of_file))
          
    if len(g) > 0:
        g = update_ecco_dataset_geospatial_metadata(g)
        g = update_ecco_dataset_temporal_coverage_metadata(g)
   

    return g  
 



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
        
