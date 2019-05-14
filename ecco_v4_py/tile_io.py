"""
ECCO v4 Python: tile_io

This module provides routines for loading ECCO netcdf files.

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
def load_ecco_grid_from_tiles_nc(grid_dir, 
                                 grid_base_name = 'ECCOv4r3_grid_tile_',
                                 tiles_to_load=range(13), 
                                 k_subset = [], 
                                 use_dask = True, 
                                 coords_as_vars = False):
        
    num_loaded_tiles = 0
    g = []
    
    for i in tiles_to_load:
        if use_dask:
            g_i = xr.open_dataset(grid_dir + '/' + grid_base_name + str(i).zfill(2) + '.nc', chunks=90)
        else:
            g_i = xr.open_dataset(grid_dir + '/' + grid_base_name + str(i).zfill(2) + '.nc')
       
        if len(k_subset) > 0:
                g_i = g_i.isel(k=k_subset)
                
        num_loaded_tiles += 1
        
        if num_loaded_tiles == 1:
            g = g_i
        else:
            g = xr.concat((g, g_i ),'tile')
            
#    for key, value in ds.variables.items():
#        if key not in ds.coords and len(ds[key].shape) >= 2:
#            ds.variables[key].attrs['grid_layout'] = 'original llc'
            
        if coords_as_vars:
            g.reset_coords()
            
            
    return g  


#%%

def recursive_load_ecco_var_from_tiles_nc(data_root_dir, 
                                          vars_to_load = 'all',
                                          tiles_to_load = range(13),
                                          k_subset = [],
                                          use_dask = True,
                                          less_output = True):
    
    g = []


    # if 'ALL' is passed as the var_names then search in data_door_dir
    # for variables.  
    if not isinstance(vars_to_load, list):
        vars_to_load = [vars_to_load]
    
    if 'all' in vars_to_load:
        var_head_dirs = np.sort(glob.glob(data_root_dir + '/*/'))
        var_names = []
       
        for var in var_head_dirs:
           var_names.append(str.split(var,'/')[-2])
       
        print ('searching for variables in ', data_root_dir)
        if len(var_names) == 0:
           print ('no variable names as top level directories in  %s' % \
                  data_root_dir)
        else:
           print ('variables found: %s ' % str(var_names))
           
        vars_to_load = var_names
        
    

    #%%
    
    dir_contents = os.walk(data_root_dir).next()
    # if we happen to have files in this directory
    if len(dir_contents[1]) == 0:
        print ('there are no subdirectories here.  ' + \
               'use load_ecco_var_from_tiles_nc to load tiles from a ' + \
               'directory of files of format VARNAME_YYYY_MM_DD_tile_NN.nc' + \
               'or VARNAME_YYYY_MM_tile_NN.nc')
    else:
        top_level_dirs = np.sort(dir_contents[1])
        
    top_level_dirs_with_var = dict()
    for var_name in vars_to_load:
        top_level_dirs_with_var[var_name] = []
    
    for top_level_dir in top_level_dirs:
        print ('searching %s for variables ' % top_level_dir)
        
        keep_looking = True
        next_dir = data_root_dir + '/' + top_level_dir
        
        while keep_looking:
            print (next_dir)
            x = next(os.walk(next_dir))

            if len(x[2]) == 0 and len(x[1]) != 0:
                next_dir = x[0] + '/' + x[1][0]
            elif len(x[2]) > 0:
                keep_looking = False
                test_var_name = x[2][0]
                var_name_here = re.split(r"_\d+",str.split(test_var_name,'/')[-1])[0]
               # print(var_name_here)
                if var_name_here in vars_to_load:
                    top_level_dirs_with_var[var_name_here].append(data_root_dir + '/' + top_level_dir + '/')
    
    print(top_level_dirs_with_var)
   
    #%%
        
      
    for var_name in vars_to_load:
        
        g_var = []

        #if not less_output:
        dirs_with_var = top_level_dirs_with_var[var_name]
        
        print ('loading %s ' % var_name)
        if len(dirs_with_var) > 0:
            print ('have directories with %s ' % var_name)
            
            for dir_with_var in dirs_with_var:
                print ('searching %s ' % dir_with_var)
                sub_dirs = np.sort(([[x[0] for x in os.walk(dir_with_var)]]))[0]

                for sub_dir in sub_dirs:
                    
                    if not less_output :
                        print (sub_dir)
                    
                    files = np.sort(glob.glob(sub_dir + '/' + var_name + '*nc'))
                   
                    if len(files) > 0:
                        if not less_output:
                            print ('found files here .. loading %s ' % str(files))
                        
                        g_i = load_ecco_var_from_tiles_nc(sub_dir, 
                                                           var_name, 
                                                           tiles_to_load,
                                                           k_subset, 
                                                           use_dask, 
                                                           less_output)
                        if len(g_var) == 0:
                            g_var = g_i
                        else:
                            g_var = xr.concat((g_var, g_i),'time')
        else:
            print ('no subdirectories with %s ' % var_name)          
        
        # if we loaded var_name, then add it to g
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
                                var_name, 
                                tiles_to_load = range(13), 
                                k_subset = [], 
                                use_dask = True,
                                less_output = True):
    
    g = []    
    files = np.sort(glob.glob(data_dir + '/' + var_name + '*nc'))
    
    if not isinstance(tiles_to_load, list):
        tiles_to_load = [tiles_to_load]
        
    for file in files:
        
        tile = int(file[-5:-3])
        var_name_of_file = re.split(r"_\d+",str.split(file,'/')[-1])[0]

        #print (var_name_of_file, var_name)
        if var_name == var_name_of_file:
            if tile in tiles_to_load:
                if not less_output:
                    print ('loading tile %d' % tile)
                    
                if use_dask:
                    g_i = xr.open_dataset(file, chunks=90)
                else:
                    g_i = xr.open_dataset(file)
               
                if len(k_subset) > 0 and \
                    'k' in g_i.coords.keys():
                    g_i = g_i.isel(k=k_subset)
                    
    
                if len(g) == 0:
                    g = g_i
                else:
                    g = xr.concat((g, g_i ),'tile')
                    
            elif not less_output:
                print ('not loading %s tile %d ' % (var_name, tile))
        else:
            print ('filename mismatch - trying to load %s and found %s ' % \
                  (var_name, var_name_of_file))
          
    if len(g) > 0:
        g = update_ecco_dataset_geospatial_metadata(g)
        g = update_ecco_dataset_temporal_coverage_metadata(g)
   

    return g  
 



#%%
def load_ecco_var_from_years_nc(data_dir, var_name, years_to_load = 'all',
                                tiles_to_load = range(13), k_subset = [],
                                use_dask = True, less_output = True):
    
    
    files = np.sort(glob.glob(data_dir + '/' + var_name + '*nc'))

    g = []
    
    if not less_output:
        print ('--- LOADING %s FROM YEARS NC: %s' % \
               (var_name, data_dir))
            
    if len(files) > 0:
        if not less_output:
            print ('---found %s nc files here.  loading ....' % var_name)
       
        for file in files:
            file_year = int(str.split(file,'.nc')[0][-4:])
            var_name_of_file = re.split(r"_\d+",str.split(file,'/')[-1])[0]
            
            if var_name_of_file == var_name:
                
                if 'all' in years_to_load or file_year in years_to_load:
                        
                    if use_dask:
                        g_i = xr.open_dataset(file,chunks=90)
                    else:
                        g_i = xr.open_dataset(file)
                    
                    # pull out a k subset
                    if len(k_subset) > 0 and 'k' in g_i.coords.keys():
                        g_i = g_i.isel(k=k_subset)
    
                    # pull out the tile subset
                    g_i = g_i.isel(tile=tiles_to_load)
                    
                    if len(g) == 0:
                        g = g_i
                    else:
                        g = xr.concat([g, g_i],'time')
        

        # finished looping through files
        if len(g) == 0:
            if not less_output:
                print ('we had files but did not load any matching %s ' % \
                       var_name)
        
        else:
            # update some metadata for fun.
            g = update_ecco_dataset_geospatial_metadata(g)
            g = update_ecco_dataset_temporal_coverage_metadata(g)
    # no files
    else:
        if not less_output:
            print ('no files found with name "%s" in %s \n' % \
                   (var_name, data_dir ))
        
    return g

#%%
def recursive_load_ecco_var_from_years_nc(data_root_dir, 
                                          vars_to_load = 'all',
                                          years_to_load = 'all',
                                          tiles_to_load = range(13),
                                          k_subset = [],
                                          use_dask = True,
                                          less_output = True):
 
    # ecco_dataset to return
    g = []

    if not isinstance(vars_to_load, list):
        vars_to_load = [vars_to_load]
        
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
    for var_name in var_names:
        g_var = []

        if 'all' in vars_to_load or var_name in vars_to_load:
            if not less_output:
                print ('trying to load %s ' % var_name)
                
            sub_dirs = np.sort(([[x[0] for x in os.walk(data_root_dir)]]))

            
            # loop through subdirectories look for var_name
            for sub_dir in sub_dirs[0]:
                if not less_output :
                    print ('searching for %s in %s ' % (var_name, sub_dir))
                
                g_i = load_ecco_var_from_years_nc(sub_dir, var_name, 
                                             years_to_load = years_to_load,
                                             tiles_to_load = tiles_to_load,
                                             k_subset =  k_subset,
                                             use_dask = use_dask,
                                             less_output =less_output)

                if len(g_i) > 0:
                    if len(g_var) == 0:
                        g_var = g_i
                    else:
                        g_var = xr.concat((g_var,g_i),'time')
                
            print 
            
            # finshed looping through all dirs 
            if len(g_var) == 0:
                print ('finished searching for %s ... not found!' % var_name) 
            else:
                print ('finished searching for %s ... success!' % var_name) 
                
        
        # if we loaded var_name, then add it to g
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
        