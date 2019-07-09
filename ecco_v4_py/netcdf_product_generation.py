#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""ECCO v4 Python: Dataset Utililites

This module includes utility routines to create the ECCO product as netcdf and
for processing metadata.

.. _ecco_v4_py Documentation :
   https://github.com/ECCO-GROUP/ECCOv4-py
"""

from __future__ import division, print_function
import numpy as np
import xarray as xr
import datetime
import time
import xmitgcm as xm
import dateutil
import glob
import os
import sys

from .read_bin_llc import load_ecco_vars_from_mds
from .ecco_utils import extract_yyyy_mm_dd_hh_mm_ss_from_datetime64

#%%    
def create_nc_grid_files_on_native_grid_from_mds(grid_input_dir, 
                                                grid_output_dir, 
                                                meta_variable_specific = [],
                                                meta_common = [], 
                                                title_basename='ECCOv4r3_grid_tile_',
                                                title='ECCOv4R3 MITgcm grid information',
                                                mds_datatype = '>f4'):
    
    mds_files = ''
#    iters_to_load = []
#    vars_to_load = []
    
    grid =  load_ecco_vars_from_mds(grid_input_dir, 
                                    mds_files, 
                                    grid_input_dir,
                                    meta_variable_specific = meta_variable_specific,
                                    meta_common = meta_common,
                                    mds_datatype = mds_datatype)

    print(grid)
    
    for key in grid.attrs.keys():
        if 'geo' in key or 'time' in key or 'cell_method' in key:
            grid.attrs.pop(key)
            
    grid.attrs['title'] = title
    
    # create the path if it does not exist/
    if not os.path.exists(grid_output_dir):
        os.makedirs(grid_output_dir)

    for i in range(13):
        tmp = grid.sel(tile=i)
        new_fname = grid_output_dir + '/' + title_basename + \
                      str(i).zfill(2) + '.nc'
        print (new_fname)
        
        tmp.to_netcdf(new_fname)
        
    return grid



#%%
def get_time_steps_from_mds_files(mds_var_dir, mds_file):
   
    mds_search = mds_var_dir + '/' + mds_file + '*meta'
    
    tmp_files = np.sort(glob.glob(mds_search))
    
    time_steps = []
    
    for i in range(len(tmp_files)):
        time_step =  int(tmp_files[i][-15:-5])
        time_steps.append(time_step)
    
    return time_steps
#%%

def create_nc_variable_files_on_native_grid_from_mds(mds_var_dir, 
                                                     mds_files_to_load,
                                                     mds_grid_dir,
                                                     output_dir,
                                                     output_freq_code,
                                                     vars_to_load = 'all',
                                                     tiles_to_load = [0,1,2,3,4,5,6,7,8,9,10,11,12],
                                                     time_steps_to_load = [],
                                                     meta_variable_specific = dict(),
                                                     meta_common = dict(),
                                                     mds_datatype = '>f4'):

    # force mds_files_to_load to be a list (if str is passed)
    if isinstance(mds_files_to_load, str):
        mds_files_to_load = [mds_files_to_load]

    # force time_steps_to_load to be a list (if int is passed)
    if isinstance(time_steps_to_load, int):
        time_steps_to_load = [time_steps_to_load]

    # for ce tiles_to_load to be a list (if int is passed)
    if isinstance(tiles_to_load, int):
        tiles_to_load = [tiles_to_load]
        
    # loop through each mds file in mds_files_to_load
    for mds_file in mds_files_to_load:
        
        # if time steps to load is empty, load all time steps
        if len(time_steps_to_load ) == 0:
            # go through each file, pull out the time step, add the time step to a list,
            # and determine the start and end time of each record.
           
           time_steps_to_load = \
               get_time_steps_from_mds_files(mds_var_dir, mds_file)

            
        #print ('mds_file ', mds_file)
        
        # find all of the files in this directory matching the name 'mds_file'
        #mds_search = mds_var_dir + '/' + mds_file + '*meta'
        
        #print ('searching in ', mds_search)
        #tmp_files = np.sort(glob.glob(mds_search))
            
        #if len(tmp_files) == 0:
        #    print ('no files matching ', mds_file)
        #    break
        
        first_meta_fname  = mds_file + '.' + \
            str(time_steps_to_load[0]).zfill(10) + '.meta'    

       
        # get metadata for the first file and determine which variables
        # are present
        meta = xm.utils.parse_meta_file(mds_var_dir + '/' + first_meta_fname)
        vars_here =  meta['fldList']
        
        if not isinstance(vars_to_load, list):
            vars_to_load = [vars_to_load]
        
        if 'all' not in vars_to_load:
            num_vars_matching = len(np.intersect1d(vars_to_load, vars_here))
            
            print ('num vars matching ', num_vars_matching)
                    
            # only proceed if we are sure that the variable we want is in this
            # mds file
            if num_vars_matching == 0:
                print ('none of the variables you want are in ', mds_file)
                print (vars_to_load)
                print (vars_here)
                
                break
        
        # loop through time steps
        for time_step in time_steps_to_load:
            #print (mds_file, time_step)
 
            # load the dataset
            ecco_dataset =  \
                load_ecco_vars_from_mds(mds_var_dir, \
                                        mds_file,
                                        mds_grid_dir,
                                        vars_to_load = vars_to_load,
                                        tiles_to_load=tiles_to_load,
                                        model_time_steps_to_load=[time_step],
                                        output_freq_code = \
                                              output_freq_code, 
                                        meta_variable_specific = \
                                              meta_variable_specific,
                                        meta_common=meta_common,
                                        mds_datatype=mds_datatype,
                                        llc_method = 'bigchunks')
            
            # pull out the associated time
            year, mon, day, hh, mm, ss  = \
                 extract_yyyy_mm_dd_hh_mm_ss_from_datetime64(ecco_dataset.time.values)
                        
            print(year, mon, day)
            
            # if the field comes from an average, 
            # extract the time bounds -- we'll use it before we save
            # the variable
            
            if 'AVG' in output_freq_code:
                tb = ecco_dataset.time_bnds
                tb.name = 'tb'
                
            # loop through each variable in this dataset, 
            for var in ecco_dataset.keys():
                print ('    ' + var)                
                var_ds = ecco_dataset[var]
                
#                # give the variable name a little 
#                if output_freq_code   == 'AVG_MON':
#                    var_ds.name = var_ds.name + '_mon'
#                elif output_freq_code == 'AVG_DAY':
#                    var_ds.name = var_ds.name + '_day'
#                elif output_freq_code == 'AVG_WEEK':
#                    var_ds.name = var_ds.name + '_wk'
#                elif output_freq_code == 'AVG_YEAR':
#                    var_ds.name = var_ds.name + '_yr'
#                elif output_freq_code == 'SNAPSHOT_MON':
#                    var_ds.name = var_ds.name + '_inst'
#                else:
#                    print ('invalid output frequency code specified')
#                    print ('leaving the variable name unchanged')
                    
                # drop these ancillary fields -- they are in grid anyway    
                for key in var_ds.coords.keys():
                    if 'CS' in key or 'SN' in key or 'Depth' in key:
                        var_ds = var_ds.drop(key)
                    
                # save each tile separately
                for tile_i in range(13):
                
                    # pull out the tile
                    tmp = var_ds.isel(tile=tile_i)
                    
                    # create the new file path name
                    if 'MON' in output_freq_code:

                        fname = var + '_' + \
                                str(year) + '_' + \
                                str(mon).zfill(2) + '_tile_' + \
                                str(tile_i).zfill(2) + '.nc'
            
                        newpath = output_dir + '/' + var + '/' + \
                            str(year) + '/' + str(mon).zfill(2)
                    
                    elif ('WEEK' in output_freq_code) or \
                         ('DAY' in output_freq_code):
                                                
                        fname = var + '_' + \
                                str(year) + '_' + \
                                str(mon).zfill(2) + '_' + \
                                str(day).zfill(2) + '_tile_' + \
                                str(tile_i).zfill(2) + '.nc'
                        d0 = datetime.datetime(year, 1,1)
                        d1 = datetime.datetime(year, mon, day)
                        doy = (d1-d0).days + 1
                        
                        newpath = output_dir + '/' + var + '/' + \
                            str(year) + '/' + str(doy).zfill(3)
                   
                        #print (d0, d1)
                        
                    elif 'YEAR' in output_freq_code:
                      
                         fname = var + '_' + \
                                str(year) + '_' + '_tile_' + \
                                str(tile_i).zfill(2) + '.nc'
        
                         newpath = output_dir + '/' + var + '/' + \
                            str(year) 
                        
                    else:
                        print ('no valid output frequency code specified')
                        print ('saving to year/mon/day/tile')
                        fname = var + '_' + \
                            str(year) + '_' + \
                            str(mon).zfill(2) + '_' + \
                            str(day).zfill(2) + '_tile_' + \
                            str(tile_i).zfill(2) + '.nc'
                        
                        newpath = output_dir + '/' + var + '/' + \
                            str(year) + '/' + str(doy).zfill(3)
                        
    
                    # create the path if it does not exist/
                    if not os.path.exists(newpath):
                        os.makedirs(newpath)
                 
                    # convert the data array to a dataset.
                    tmp = tmp.to_dataset()

                    # add the time bounds field back in if we have an 
                    # average field
                    if 'AVG' in output_freq_code:
                        tmp = xr.merge((tmp, tb))
                        tmp = tmp.drop('tb')
    
                    # put the metadata back in
                    tmp.attrs = ecco_dataset.attrs
                    
                    # update the temporal and geospatial metadata
                    tmp = update_ecco_dataset_geospatial_metadata(tmp)
                    tmp = update_ecco_dataset_temporal_coverage_metadata(tmp)
                    
                    # save to netcdf.  it's that simple.
                    print ('saving to %s' % newpath + '/' + fname)
                    tmp.to_netcdf(newpath + '/' + fname)
                    
    return ecco_dataset, tmp


   
#%%
def update_ecco_dataset_temporal_coverage_metadata(ecco_dataset):
    """

    Adds high-level temporal coverage metadata to dataset object if the 
    dataset object has 'time_bnds' coordinates
    
    Input
    ----------    
    ecco_dataset : an xarray dataset


    Output: 
    ----------    
    ecco_dataset : dataset updated with 'time_coverage_start/end', if such
    bounds can be determined
    
    """
    
    if 'time_bnds' in ecco_dataset.coords.keys():
        
        # if there is only one time bounds
        if len(ecco_dataset.time_bnds.shape) == 1:
            ecco_dataset.attrs['time_coverage_start'] = \
                  str(ecco_dataset.time_bnds.values[0])[0:19]
            ecco_dataset.attrs['time_coverage_end']   = \
                    str(ecco_dataset.time_bnds.values[1])[0:19]

        else:
        # if there are many time bounds 
            ecco_dataset.attrs['time_coverage_start'] = \
                str(ecco_dataset.time_bnds.values[0][0])[0:19]
            ecco_dataset.attrs['time_coverage_end']   = \
                str(ecco_dataset.time_bnds.values[-1][-1])[0:19]
    #elif 'time' in ecco_dataset.coords.keys():
    #    ecco_dataset.attrs['time_coverage_start'] = str(ecco_dataset.time.values[0])[0:19]
    #    ecco_dataset.attrs['time_coverage_end']   = str(ecco_dataset.time.values[-1])[0:19]
    
    return ecco_dataset

#%%
def update_ecco_dataset_geospatial_metadata(ecco_dataset):
    """

    Adds high-level geographical coverage metadata to dataset object if the 
    dataset object has 'YG or YC' coordinates
    
    Input
    ----------    
    ecco_dataset : an xarray dataset


    Output: 
    ----------    
    ecco_dataset : dataset updated with 'geospatial extents', if such
    bounds can be determined
    
    """
    
        # set geospatial bounds
    if 'YG' in ecco_dataset.coords.keys() :
        ecco_dataset.attrs['geospatial_lat_max'] = ecco_dataset.YG.values.max()
        ecco_dataset.attrs['geospatial_lat_min'] = ecco_dataset.YG.values.min()     
        ecco_dataset.attrs['nx'] = ecco_dataset.YG.shape[-2]
        ecco_dataset.attrs['ny'] = ecco_dataset.YG.shape[-1]
    
    elif 'YC' in ecco_dataset.coords.keys() :
        ecco_dataset.attrs['geospatial_lat_max'] = ecco_dataset.YC.values.max()
        ecco_dataset.attrs['geospatial_lat_min'] = ecco_dataset.YC.values.min()     
        ecco_dataset.attrs['nx'] = ecco_dataset.YC.shape[-2]
        ecco_dataset.attrs['ny'] = ecco_dataset.YC.shape[-1]
    
    if 'XG' in ecco_dataset.coords.keys():
        ecco_dataset.attrs['geospatial_lon_max'] = ecco_dataset.XG.values.max()
        ecco_dataset.attrs['geospatial_lon_min'] = ecco_dataset.XG.values.min()
    elif 'XC' in ecco_dataset.coords.keys():
        ecco_dataset.attrs['geospatial_lon_max'] = ecco_dataset.XC.values.max()
        ecco_dataset.attrs['geospatial_lon_min'] = ecco_dataset.XC.values.min()
            
    if 'drF' in ecco_dataset.coords.keys():
        ecco_dataset.attrs['geospatial_vertical_max'] = \
            ecco_dataset.drF.cumsum().values[-1]
        ecco_dataset.attrs['geospatial_vertical_min'] = 0
        ecco_dataset.attrs['nz'] = len(ecco_dataset.k)
    else:
        ecco_dataset.attrs['geospatial_vertical_max'] = 0
        ecco_dataset.attrs['geospatial_vertical_min'] = 0
        ecco_dataset.attrs['nz'] = 1
    
    return ecco_dataset
