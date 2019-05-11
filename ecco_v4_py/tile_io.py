#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon Jul  3 16:11:15 2017

@author: ifenty
"""
from __future__ import division,print_function
import numpy as np
import xarray as xr
from copy import deepcopy
import glob
import os

#%%
def load_ecco_grid_tiles_from_nc(grid_dir, 
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

def recursive_load_ecco_var_tiles_from_nc(data_root_dir, 
                                          var_names,
                                          tiles_to_load = range(13),
                                          k_subset = [],
                                          use_dask = True,
                                          less_output = True):
    
    ecco_dataset = []
    ecco_dataset_var = []


    # if 'ALL' is passed as the var_names then search in data_door_dir
    # for variables.  
    if isinstance(var_names, str) and 'ALL' in var_names:
        var_head_dirs = np.sort(glob.glob(data_root_dir + '/*'))
        var_names = []
       
        for var in var_head_dirs:
           var_names.append(str.split(var,'/')[-1])
       
        print ('searching for variables in ', data_root_dir)
        if len(var_names) == 0:
           print ('none found')
        else:
           print ('found ',  var_names)
    
    
    # if only one var_name is passed (say as a string)) then
    # we convert the string to a list
    elif not isinstance(var_names, list):
        var_names = [var_names]
      
        
    for var_name in var_names:
        #if not less_output:
        print (var_name)
            
        ecco_dataset_var = []
     
        sub_dirs = np.sort(([[x[0] for x in os.walk(data_root_dir)]]))
       
        if not less_output :
            print (sub_dirs[0])
                    
        for sub_dir in sub_dirs[0]:
            if not less_output :
                print (sub_dir)
            
            files = np.sort(glob.glob(sub_dir + '/' + var_name + '*nc'))
           
            if len(files) > 0:
                if not less_output:
                    print ('found files here .. loading ', files)
               
                ecco_dataset_tmp = load_ecco_var_tiles_from_nc(sub_dir, 
                                                               var_name, 
                                                               tiles_to_load,
                                                               k_subset, 
                                                               use_dask, 
                                                               less_output)
               
                if len(ecco_dataset_var) == 0:
                    # create the dataset if it hasn't yet been initialized
                    ecco_dataset_var = ecco_dataset_tmp
                else:
                    # concat the new dataset along the time axis
                    ecco_dataset_var = \
                        xr.concat((ecco_dataset_var, ecco_dataset_tmp), 'time')
        
        #if len(ecco_dataset) == 0 and len(ecco_dataset_var) > 0:
        #    ecco_dataset = ecco_dataset_var
        #else:
        #    ecco_dataset = xr.merge((ecco_dataset, ecco_dataset_var))
        if len(ecco_dataset) == 0 and len(ecco_dataset_var) > 0:
            ecco_dataset = ecco_dataset_var
        elif len(ecco_dataset) > 0 and len(ecco_dataset_var) > 0:
            ecco_dataset = xr.merge((ecco_dataset, ecco_dataset_var))
    
    return ecco_dataset


#%%
def load_ecco_var_tiles_from_nc(data_dir, 
                                var_name, 
                                tiles_to_load = range(13), 
                                k_subset = [], 
                                use_dask = True,
                                less_output = True):
    
    num_loaded_tiles = 0
    g = []    
    files = np.sort(glob.glob(data_dir + '/' + var_name + '*nc'))
    
    for f in files:
        
        tile = int(f[-5:-3])
        if tile in tiles_to_load:
            if not less_output:
                print ('loading tile %d' % tile)
                
            if use_dask:
                g_i = xr.open_dataset(f, chunks=90)
            else:
                g_i = xr.open_dataset(f)
           
            if len(k_subset) > 0:
                g_i = g_i.isel(k=k_subset)
                
            num_loaded_tiles += 1

            #print (g_i.tile.values)
            if num_loaded_tiles == 1:
                g = g_i
            else:
                g = xr.concat((g, g_i ),'tile')
                
        elif not less_output:
            print ('not loading %s tile %d ' % (var_name, tile))
          
#    for key, value in ds.variables.items():
#        if key not in ds.coords and len(ds[key].shape) >= 2:
#            ds.variables[key].attrs['grid_layout'] = 'original llc'

    return g  
 



    
