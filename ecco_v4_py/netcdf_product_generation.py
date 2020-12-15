#!/usr/bin/env python3
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
import pyresample as pr
import json
from pathlib import Path
import uuid

from .read_bin_llc import load_ecco_vars_from_mds
from .ecco_utils import extract_yyyy_mm_dd_hh_mm_ss_from_datetime64
from .resample_to_latlon import resample_to_latlon

#%%
def create_nc_grid_files_on_native_grid_from_mds(grid_input_dir,
                                                 grid_output_dir,
                                                 coordinate_metadata = dict(),
                                                 geometry_metadata = dict(),
                                                 global_metadata = dict(),
                                                 cell_bounds = None,
                                                 file_basename='ECCO-GRID',
                                                 title='ECCOv4 MITgcm grid information',
                                                 mds_datatype = '>f4',
                                                 write_to_disk = True,
                                                 less_output=True):

    # grid input dir and grid output dir should be of type pathlib.PosixPath

    mds_files = ''

    if isinstance(grid_input_dir, str):
        grid_input_dir = Path(grid_input_dir)
    if isinstance(grid_output_dir, str):
        grid_output_dir = Path(grid_output_dir)

    if not less_output:
        print(str(grid_input_dir))

    grid =  load_ecco_vars_from_mds(str(grid_input_dir),
                                    str(grid_input_dir),
                                    mds_files,
                                    vars_to_load = 'all',
                                    drop_unused_coords = False,
                                    grid_vars_to_coords = False,
                                    coordinate_metadata = coordinate_metadata,
                                    variable_metadata = geometry_metadata,
                                    global_metadata = global_metadata,
                                    cell_bounds = cell_bounds,
                                    mds_datatype = mds_datatype,
                                    less_output = less_output)

    if not less_output:
        print(grid)

#    for key in grid.attrs.keys():
#        if 'geo' in key or 'time' in key or 'cell_method' in key:
#            grid.attrs.pop(key)


    if 'time_coverage_end' in list(grid.attrs.keys()):
        grid.attrs.pop('time_coverage_end')

    if 'time_coverage_start' in list(grid.attrs.keys()):
        grid.attrs.pop('time_coverage_start')


    grid.attrs['title'] = title
    grid.attrs['product_name'] = file_basename + '.nc'
    grid.attrs['uuid'] = str(uuid.uuid1())


    # remove attributes that have TBD in their values
    for attr in list(grid.attrs.keys()):
        if type(grid.attrs[attr]) == str :
            if 'TBD' in grid.attrs[attr]:
                grid.attrs.pop(attr)

    # save to disk
    if write_to_disk:
        if not grid_output_dir.exists():
            try:
                grid_output_dir.mkdir()
            except:
                print ('cannot make %s ' % grid_output_dir)

        new_fname = grid_output_dir / (file_basename + '.nc')

        if not less_output:
            print('making single file grid netcdf')
            print(str(new_fname))

        if not less_output:
            print('\n... creating variable encodings')

        # PROVIDE SPECIFIC ENCODING DIRECTIVES FOR EACH DATA VAR
        dv_encoding = dict()
        for dv in grid.data_vars:
            dv_encoding[dv] =  {'zlib':True, \
                                'complevel':5,\
                                'shuffle':True,\
                                '_FillValue':None}


        # PROVIDE SPECIFIC ENCODING DIRECTIVES FOR EACH COORDINATE
        if not less_output:
            print('\n... creating coordinate encodings')
        coord_encoding = dict()

        for coord in grid.coords:
            # default encoding: no fill value, float32
            coord_encoding[coord] = {'_FillValue':None, 'dtype':'float32'}

            if grid[coord].values.dtype == np.int64:
                grid[coord].values[:] = grid[coord].astype(np.int32)
                coord_encoding[coord]['dtype'] ='int32'

        # MERGE ENCODINGS for coordinates and variables
        encoding = {**dv_encoding, **coord_encoding}


        if not less_output:
            print('\n... saving single file to netcdf') 
        grid.to_netcdf(str(new_fname), encoding=encoding)

        # save as 13 tiles
        if not less_output:
            print('making 13 file grid netcdf')

        for i in range(13):
            tmp = grid.sel(tile=i)
            tmp2 = file_basename  + '_TILE_' + str(i).zfill(2) + '.nc'
            tmp.attrs['product_name'] = tmp2

            new_fname = grid_output_dir  / tmp2
            if not less_output:
                print (new_fname)

            tmp.to_netcdf(str(new_fname), encoding=encoding)

    return grid



#%%
def get_time_steps_from_mds_files(mds_var_dir, mds_file):

    if isinstance(mds_var_dir, str):
        mds_var_dir = Path(mds_var_dir)

    tmp_files = np.sort(list(mds_var_dir.glob(mds_file + '*meta')))

    time_steps = []

    print ('get time steps')
    print (tmp_files)
    for i in range(len(tmp_files)):
        time_steps.append(int(tmp_files[i].stem[-10:]))

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
                                                     mds_datatype = '>f4',
                                                     verbose=True,
                                                     method = 'time_interval_and_combined_tiles',
                                                     less_output=True):

    #%%
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

        if not less_output:
            print(mds_file)
        # if time steps to load is empty, load all time steps
        if len(time_steps_to_load ) == 0:
            # go through each file, pull out the time step, add the time step to a list,
            # and determine the start and end time of each record.

           time_steps_to_load = \
               get_time_steps_from_mds_files(mds_var_dir, mds_file)


        first_meta_fname  = mds_file + '.' + \
            str(time_steps_to_load[0]).zfill(10) + '.meta'


        # get metadata for the first file and determine which variables
        # are present
        meta = xm.utils.parse_meta_file(str(mds_var_dir / first_meta_fname))
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
        #%%


        ecco_dataset_all =  \
                load_ecco_vars_from_mds(mds_var_dir, \
                                         mds_grid_dir,
                                         mds_file,
                                         vars_to_load = vars_to_load,
                                         tiles_to_load=tiles_to_load,
                                         model_time_steps_to_load=time_steps_to_load,
                                         output_freq_code = \
                                              output_freq_code,
                                         meta_variable_specific = \
                                              meta_variable_specific,
                                         meta_common=meta_common,
                                         mds_datatype=mds_datatype,
                                         llc_method = 'bigchunks')

        if(verbose):
            print ('loaded ecco dataset....')
        # loop through time steps, one at a time.
        for time_step in time_steps_to_load:

            i, = np.where(ecco_dataset_all.timestep == time_step)
            if(verbose):
                print (ecco_dataset_all.timestep.values)
                print ('time step ', time_step, i)

            # load the dataset
            ecco_dataset = ecco_dataset_all.isel(time=i)

            # pull out the year, month day, hour, min, sec associated with
            # this time step
            if type(ecco_dataset.time.values) == np.ndarray:
                cur_time = ecco_dataset.time.values[0]
            else:
                cur_time = ecco_dataset.time.values

            #print (type(cur_time))
            year, mon, day, hh, mm, ss  = \
                 extract_yyyy_mm_dd_hh_mm_ss_from_datetime64(cur_time)

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

                # drop these ancillary fields -- they are in grid anyway
                keys_to_drop = ['CS','SN','Depth','rA','PHrefC','hFacC',\
                                'maskC','drF', 'dxC', 'dyG', 'rAw', 'hFacW',\
                                'rAs','hFacS','maskS','dxG','dyC', 'maskW']

                for key_to_drop in keys_to_drop:
                    #print (key_to_drop)
                    if key_to_drop in var_ds.coords.keys():
                        var_ds = var_ds.drop(key_to_drop)
                #%%
                # METHOD 'TIME_INTERVAL_AND_COMBINED_TILES'
                # --> MAKES ONE FILE PER TIME RECORD, KEEPS TILES TOGETHER

                if method == 'time_interval_and_combined_tiles':
                    # create the new file path name
                        if 'MON' in output_freq_code:

                            fname = var + '_' +  str(year) + '_' + \
                                str(mon).zfill(2) + '.nc'

                            newpath = output_dir  /  var /  \
                                str(year)

                        elif ('WEEK' in output_freq_code) or \
                             ('DAY' in output_freq_code):

                            fname = var + '_' + \
                                    str(year) + '_' + \
                                    str(mon).zfill(2) + '_' + \
                                    str(day).zfill(2) +  '.nc'
                            d0 = datetime.datetime(year, 1,1)
                            d1 = datetime.datetime(year, mon, day)
                            doy = (d1-d0).days + 1

                            if not less_output:
                                print('--- making one file per time record')
                                print(output_dir)

                            newpath = output_dir / var / str(year) / \
                                str(doy).zfill(3)

                        elif 'YEAR' in output_freq_code:

                             fname = var + '_' + str(year) + '.nc'

                             newpath = output_dir  /  var  / str(year)

                        else:
                            print ('no valid output frequency code specified')
                            print ('saving to year/mon/day/tile')
                            fname = var + '_' + \
                                str(year) + '_' + \
                                str(mon).zfill(2) + '_' + \
                                str(day).zfill(2) + '.nc'
                            d0 = datetime.datetime(year, 1,1)
                            d1 = datetime.datetime(year, mon, day)
                            doy = (d1-d0).days + 1

                            newpath = output_dir  /  var /  \
                                str(year)  / str(doy).zfill(3)

                        # create the path if it does not exist/
                        if not newpath.exists():
                            newpath.mkdir(parents=True, exist_ok=True)

                        # convert the data array to a dataset.
                        tmp = var_ds.to_dataset()

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
                        if(verbose):
                            print ('saving to %s' % str(newpath  /  fname))
                        tmp.to_netcdf(str(newpath  /  fname), engine='netcdf4')

                # METHOD 'TIME_INTERVAL_AND_SEPARATED_TILES'
                # --> MAKES ONE FILE PER TIME RECORD PER TILE

                if method == 'time_interval_and_separate_tiles':

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
                            d0 = datetime.datetime(year, 1,1)
                            d1 = datetime.datetime(year, mon, day)
                            doy = (d1-d0).days + 1

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
                        if(verbose):
                            print ('saving to %s' % newpath + '/' + fname)
                        tmp.to_netcdf(newpath + '/' + fname, engine='netcdf4')

#%%
    ecco_dataset_all.close()
    return ecco_dataset, tmp

# create the interpolated fields. Default is on 0.5 degrees by 0.5 degrees.
def create_nc_variable_files_on_regular_grid_from_mds(mds_var_dir,
                                                     mds_files_to_load,
                                                     mds_grid_dir,
                                                     output_dir,
                                                     output_freq_code,
                                                     vars_to_load = 'all',
                                                     tiles_to_load = [0,1,2,3,4,5,6,7,8,9,10,11,12],
                                                     time_steps_to_load = [],
                                                     meta_variable_specific = dict(),
                                                     meta_common = dict(),
                                                     mds_datatype = '>f4',
                                                     dlon=0.5, dlat=0.5,
                                                     radius_of_influence = 120000,
                                                     express=1,
                                                     kvarnmidx = 2, # coordinate idx for vertical axis
                                                     # method now is only a place holder.
                                                     # This can be expanded. For example,
                                                     # the global interpolated fields can
                                                     # split to tiles, similarly to
                                                     # the tiled native fields, to
                                                     # reduce the size of each file.
                                                     verbose=True,
                                                     method = ''):
    #%%
    # force mds_files_to_load to be a list (if str is passed)
    if isinstance(mds_files_to_load, str):
        mds_files_to_load = [mds_files_to_load]

    # force time_steps_to_load to be a list (if int is passed)
    if isinstance(time_steps_to_load, int):
        time_steps_to_load = [time_steps_to_load]

    # for ce tiles_to_load to be a list (if int is passed)
    if isinstance(tiles_to_load, int):
        tiles_to_load = [tiles_to_load]

    # if no specific file data passed, read default metadata from json file
    # -- variable specific meta data
    script_dir = os.path.dirname(__file__)  # <-- absolute dir the script is in
    if not meta_variable_specific:
        meta_variable_rel_path = '../meta_json/ecco_meta_variable.json'
        abs_meta_variable_path = os.path.join(script_dir, meta_variable_rel_path)
        with open(abs_meta_variable_path, 'r') as fp:
            meta_variable_specific = json.load(fp)

    # --- common meta data
    if not meta_common:
        meta_common_rel_path = '../meta_json/ecco_meta_common.json'
        abs_meta_common_path = os.path.join(script_dir, meta_common_rel_path)
        with open(abs_meta_common_path, 'r') as fp:
            meta_common = json.load(fp)

    # info for the regular grid
    new_grid_min_lat = -90+dlat/2.
    new_grid_max_lat = 90-dlat/2.
    new_grid_min_lon = -180+dlon/2.
    new_grid_max_lon = 180-dlon/2.
    new_grid_ny = np.int((new_grid_max_lat-new_grid_min_lat)/dlat + 1 + 1e-4*dlat)
    new_grid_nx = np.int((new_grid_max_lon-new_grid_min_lon)/dlon + 1 + 1e-4*dlon)
    j_reg = new_grid_min_lat + np.asarray(range(new_grid_ny))*dlat
    i_reg = new_grid_min_lon + np.asarray(range(new_grid_nx))*dlon
    j_reg_idx = np.asarray(range(new_grid_ny))
    i_reg_idx = np.asarray(range(new_grid_nx))
    if (new_grid_ny < 1) or (new_grid_nx < 1):
        raise ValueError('You need to have at least one grid point for the new grid.')

    # loop through each mds file in mds_files_to_load
    for mds_file in mds_files_to_load:

        # if time steps to load is empty, load all time steps
        if len(time_steps_to_load ) == 0:
            # go through each file, pull out the time step, add the time step to a list,
            # and determine the start and end time of each record.

           time_steps_to_load = \
               get_time_steps_from_mds_files(mds_var_dir, mds_file)


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
        #%%
        # load the MDS fields
        ecco_dataset_all =  \
                load_ecco_vars_from_mds(mds_var_dir, \
                                         mds_grid_dir,
                                         mds_file,
                                         vars_to_load = vars_to_load,
                                         tiles_to_load=tiles_to_load,
                                         model_time_steps_to_load=time_steps_to_load,
                                         output_freq_code = \
                                              output_freq_code,
                                         meta_variable_specific = \
                                              meta_variable_specific,
                                         meta_common=meta_common,
                                         mds_datatype=mds_datatype,
                                         llc_method = 'bigchunks')

        # do the actual loading. Otherwise, the code may be slow.
        ecco_dataset_all.load()

        # print(ecco_dataset_all.keys())
        # loop through each variable in this dataset,
        for var in ecco_dataset_all.keys():
            print ('    ' + var)
            # obtain the grid information (use fields from time=0)
            # Note that nrtmp would always equal to one,
            # since each outfile will include only one time-record (e.g. daily, monthly avgs.).

            ecco_dataset = ecco_dataset_all.isel(time=[0])

            var_ds = ecco_dataset[var]

            shapetmp = var_ds.shape

            lenshapetmp = len(shapetmp)
            nttmp = 0
            nrtmp = 0
            if(lenshapetmp==4):
                nttmp = shapetmp[0]
                nrtmp = 0
            elif(lenshapetmp==5):
                nttmp = shapetmp[0]
                nrtmp = shapetmp[1]
            else:
                print('Error! ', var_ds.shape)
                sys.exit()

            # Get X,Y of the original grid. They could be XC/YC, XG/YC, XC/YG, etc.
            # Similar for mask.
            # default is XC, YC
            if 'i' in var_ds.coords.keys():
                XX = ecco_dataset['XC']
                XXname = 'XC'
            if 'j' in var_ds.coords.keys():
                YY = ecco_dataset['YC']
                YYname = 'YC'
            varmask = 'maskC'
            iname = 'i'
            jname = 'j'

            if 'i_g' in var_ds.coords.keys():
                XX = ecco_dataset['XG']
                XXname = 'XG'
                varmask = 'maskW'
                iname = 'i_g'
            if 'j_g' in var_ds.coords.keys():
                YY = ecco_dataset['YG']
                YYname = 'YG'
                varmask = 'maskS'
                jname = 'j_g'

            # interpolation
            # To do it fast, set express==1 (default)
            if(express==1):
                orig_lons_1d = XX.values.ravel()
                orig_lats_1d = YY.values.ravel()
                orig_grid = pr.geometry.SwathDefinition(lons=orig_lons_1d,
                                                        lats=orig_lats_1d)

                if (new_grid_ny > 0) and (new_grid_nx > 0):
                    # 1D grid values
                    new_grid_lon, new_grid_lat = np.meshgrid(i_reg, j_reg)

                    # define the lat lon points of the two parts.
                    new_grid  = pr.geometry.GridDefinition(lons=new_grid_lon,
                                                           lats=new_grid_lat)

                    # Get the neighbor info once.
                    # It will be used repeatedly late to resample data
                    # fast for each of the datasets that is based on
                    # the same swath, e.g. for a model variable at different times.
                    valid_input_index, valid_output_index, index_array, distance_array = \
                    pr.kd_tree.get_neighbour_info(orig_grid,
                                               new_grid, radius_of_influence,
                                               neighbours=1)

            # loop through time steps, one at a time.
            for time_step in time_steps_to_load:

                i, = np.where(ecco_dataset_all.timestep == time_step)
                if(verbose):
                    print (ecco_dataset_all.timestep.values)
                    print ('time step ', time_step, i)

                # load the dataset
                ecco_dataset = ecco_dataset_all.isel(time=i)

                # pull out the year, month day, hour, min, sec associated with
                # this time step
                if type(ecco_dataset.time.values) == np.ndarray:
                    cur_time = ecco_dataset.time.values[0]
                else:
                    cur_time = ecco_dataset.time.values
                #print (type(cur_time))
                year, mon, day, hh, mm, ss  = \
                     extract_yyyy_mm_dd_hh_mm_ss_from_datetime64(cur_time)

                print(year, mon, day)

                # if the field comes from an average,
                # extract the time bounds -- we'll use it before we save
                # the variable
                if 'AVG' in output_freq_code:
                    tb = ecco_dataset.time_bnds
                    tb.name = 'tb'

                var_ds = ecco_dataset[var]

                # 3d fields (with Z-axis) for each time record
                if(nttmp != 0 and nrtmp != 0):
                    tmpall = np.zeros((nttmp, nrtmp,new_grid_ny,new_grid_nx))
                    for ir in range(nrtmp): # Z-loop
                        # mask
                        maskloc = ecco_dataset[varmask].values[ir,:]

                        for it in range(nttmp): # time loop
                            # one 2d field at a time
                            var_ds_onechunk = var_ds[it,ir,:]
                            # apply mask
                            var_ds_onechunk.values[maskloc==0]=np.nan
                            orig_field = var_ds_onechunk.values
                            if(express==1):
                                tmp = pr.kd_tree.get_sample_from_neighbour_info(
                                        'nn', new_grid.shape, orig_field,
                                        valid_input_index, valid_output_index,
                                        index_array)

                            else:
                                new_grid_lon, new_grid_lat, tmp = resample_to_latlon(XX, YY, orig_field,
                                                                  new_grid_min_lat,
                                                                  new_grid_max_lat, dlat,
                                                                  new_grid_min_lon,
                                                                  new_grid_max_lon, dlon,
                                                                  nprocs_user=1,
                                                                  mapping_method = 'nearest_neighbor',
                                                                  radius_of_influence=radius_of_influence)
                            tmpall[it,ir,:] = tmp
                # 2d fields (without Z-axis) for each time record
                elif(nttmp != 0):
                    tmpall = np.zeros((nttmp, new_grid_ny,new_grid_nx))
                    # mask
                    maskloc = ecco_dataset[varmask].values[0,:]
                    for it in range(nttmp): # time loop
                        var_ds_onechunk = var_ds[it,:]
                        var_ds_onechunk.values[maskloc==0]=np.nan
                        orig_field = var_ds_onechunk.values
                        if(express==1):
                            tmp = pr.kd_tree.get_sample_from_neighbour_info(
                                    'nn', new_grid.shape, orig_field,
                                    valid_input_index, valid_output_index,
                                    index_array)
                        else:
                            new_grid_lon, new_grid_lat, tmp = resample_to_latlon(XX, YY, orig_field,
                                                              new_grid_min_lat,
                                                              new_grid_max_lat, dlat,
                                                              new_grid_min_lon,
                                                              new_grid_max_lon, dlon,
                                                              nprocs_user=1,
                                                              mapping_method = 'nearest_neighbor',
                                                              radius_of_influence=radius_of_influence)
                        tmpall[it,:] = tmp

                else:
                    print('Error! both nttmp and nrtmp are zeros.')
                    sys.exit()
                # set the coordinates for the new (regular) grid
                # 2d fields
                if(lenshapetmp==4):
                    var_ds_reg = xr.DataArray(tmpall,
                                              coords = {'time': var_ds.coords['time'].values,
                                                        'j': j_reg_idx,
                                                        'i': i_reg_idx},\
                                              dims = ('time', 'j', 'i'))
                # 3d fields
                elif(lenshapetmp==5):
                    # Get the variable name (kvarnm) for Z-axis: k, k_l
                    kvarnm = var_ds.coords.keys()[kvarnmidx]

                    if(kvarnm[0]!='k'):
                        kvarnmidxnew = kvarnmidx
                        for iktmp, ktmp in enumerate(var_ds.coords.keys()):
                            if(ktmp[0]=='k'):
                                kvarnmidxnew = iktmp
                        if(kvarnmidxnew==kvarnmidx):
                            print('Error! Seems ', kvarnm, ' is not the vertical axis.')
                            print(var_ds)
                            sys.exit()
                        else:
                            kvarnmidx = kvarnmidxnew
                            kvarnm = var_ds.coords.keys()[kvarnmidx]

                    var_ds_reg = xr.DataArray(tmpall,
                                              coords = {'time': var_ds.coords['time'].values,
                                                        kvarnm: var_ds.coords[kvarnm].values,
                                                        'j': j_reg_idx,
                                                        'i': i_reg_idx},\
                                              dims = ('time', kvarnm,'j', 'i'))
                # set the attrs for the new (regular) grid
                var_ds_reg['j'].attrs = var_ds[jname].attrs
                var_ds_reg['i'].attrs = var_ds[iname].attrs
                var_ds_reg['j'].attrs['long_name'] = 'y-dimension'
                var_ds_reg['i'].attrs['long_name'] = 'x-dimension'
                var_ds_reg['j'].attrs['swap_dim'] = 'latitude'
                var_ds_reg['i'].attrs['swap_dim'] = 'longitude'

                var_ds_reg['latitude'] = (('j'), j_reg)
                var_ds_reg['longitude'] = (('i'), i_reg)
                var_ds_reg['latitude'].attrs = ecco_dataset[YYname].attrs
                var_ds_reg['longitude'].attrs = ecco_dataset[XXname].attrs
                var_ds_reg['latitude'].attrs['long_name'] = "latitude at center of grid cell"
                var_ds_reg['longitude'].attrs['long_name'] = "longitude at center of grid cell"

                var_ds_reg.name = var_ds.name

                #keys_to_drop = ['tile','j','i','XC','YC','XG','YG']
                # drop these ancillary fields -- they are in grid anyway
                keys_to_drop = ['CS','SN','Depth','rA','PHrefC','hFacC',\
                                'maskC','drF', 'dxC', 'dyG', 'rAw', 'hFacW',\
                                'rAs','hFacS','maskS','dxG','dyC', 'maskW', \
                                'tile','XC','YC','XG','YG']

                for key_to_drop in keys_to_drop:
                    #print (key_to_drop)
                    if key_to_drop in var_ds.coords.keys():
                        var_ds = var_ds.drop(key_to_drop)

                # any remaining fields, e.g. time, would be included in the interpolated fields.
                for key_to_add in var_ds.coords.keys():
                    if(key_to_add not in var_ds_reg.coords.keys()):
                        if(key_to_add != 'i_g' and key_to_add != 'j_g'):
                            var_ds_reg[key_to_add] = var_ds[key_to_add]

                # use the same global attributs
                var_ds_reg.attrs = var_ds.attrs


                #print(var_ds.coords.keys())
                #%%

                # create the new file path name
                if 'MON' in output_freq_code:

                    fname = var + '_' +  str(year) + '_' + str(mon).zfill(2) + '.nc'

                    newpath = output_dir + '/' + var + '/' + \
                        str(year) + '/'

                elif ('WEEK' in output_freq_code) or \
                     ('DAY' in output_freq_code):

                    fname = var + '_' + \
                            str(year) + '_' + \
                            str(mon).zfill(2) + '_' + \
                            str(day).zfill(2) +  '.nc'
                    d0 = datetime.datetime(year, 1,1)
                    d1 = datetime.datetime(year, mon, day)
                    doy = (d1-d0).days + 1

                    newpath = output_dir + '/' + var + '/' + \
                        str(year) + '/' + str(doy).zfill(3)

                elif 'YEAR' in output_freq_code:

                     fname = var + '_' + str(year) + '.nc'

                     newpath = output_dir + '/' + var + '/' + \
                        str(year)

                else:
                    print ('no valid output frequency code specified')
                    print ('saving to year/mon/day/tile')
                    fname = var + '_' + \
                        str(year) + '_' + \
                        str(mon).zfill(2) + '_' + \
                        str(day).zfill(2) + '.nc'
                    d0 = datetime.datetime(year, 1,1)
                    d1 = datetime.datetime(year, mon, day)
                    doy = (d1-d0).days + 1

                    newpath = output_dir + '/' + var + '/' + \
                        str(year) + '/' + str(doy).zfill(3)


                # create the path if it does not exist/
                if not os.path.exists(newpath):
                    os.makedirs(newpath)

                # convert the data array to a dataset.
                tmp = var_ds_reg.to_dataset()

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
                if(verbose):
                    print ('saving to %s' % newpath + '/' + fname)
                # do not include _FillValue
                encoding = {i: {'_FillValue': False} for i in tmp.variables.keys()}

                tmp.to_netcdf(newpath + '/' + fname, engine='netcdf4',encoding=encoding)



#%%
    ecco_dataset_all.close()
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

    if 'k' in ecco_dataset.coords.keys():
        ecco_dataset.attrs['geospatial_vertical_max'] = \
            ecco_dataset.Z.values[0]
        ecco_dataset.attrs['geospatial_vertical_min'] = \
            ecco_dataset.Z.values[-1]
        ecco_dataset.attrs['nz'] = len(ecco_dataset.k.values)
    elif 'k_l' in ecco_dataset.coords.keys():
        ecco_dataset.attrs['geospatial_vertical_max'] = \
            ecco_dataset.Zl.values[0]
        ecco_dataset.attrs['geospatial_vertical_min'] = \
            ecco_dataset.Zl.values[-1]
        ecco_dataset.attrs['nz'] = len(ecco_dataset.k_l.values)
    elif 'k_u' in ecco_dataset.coords.keys():
        ecco_dataset.attrs['geospatial_vertical_max'] = \
            ecco_dataset.Zu.values[0]
        ecco_dataset.attrs['geospatial_vertical_min'] = \
            ecco_dataset.Zu.values[-1]
        ecco_dataset.attrs['nz'] = len(ecco_dataset.k_u.values)
    else:
        ecco_dataset.attrs['geospatial_vertical_max'] = 0
        ecco_dataset.attrs['geospatial_vertical_min'] = 0
        ecco_dataset.attrs['nz'] = 1

    return ecco_dataset

