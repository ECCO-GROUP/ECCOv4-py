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
import pyresample as pr
import json
from pathlib import Path

from .read_bin_llc import load_ecco_vars_from_mds
from .ecco_utils import extract_yyyy_mm_dd_hh_mm_ss_from_datetime64
from .ecco_utils import make_1D_latlon_bounds_from_ecco_dataset
from .resample_to_latlon import resample_to_latlon



# %%
def create_nc_grid_files_on_native_grid_from_mds(grid_input_dir,
                                                 grid_output_dir,
                                                 meta_variable_specific=dict(),
                                                 meta_common=dict(),
                                                 title_basename='ECCO-GRID',
                                                 title='ECCOv4 MITgcm grid information',
                                                 mds_datatype='>f4',
                                                 less_output=True):
    # grid input dir and grid output dir should be of type pathlib.PosixPath

    mds_files = ''

    if isinstance(grid_input_dir, str):
        grid_input_dir = Path(grid_input_dir)
    if isinstance(grid_output_dir, str):
        grid_output_dir = Path(grid_output_dir)

    print(str(grid_input_dir))
    grid = load_ecco_vars_from_mds(str(grid_input_dir),
                                   str(grid_input_dir),
                                   mds_files,
                                   meta_variable_specific=meta_variable_specific,
                                   meta_common=meta_common,
                                   mds_datatype=mds_datatype,
                                   less_output=less_output)

    print(grid)

    for key in grid.attrs.keys():
        if 'geo' in key or 'time' in key or 'cell_method' in key:
            grid.attrs.pop(key)

    grid.attrs['title'] = title

    if not grid_output_dir.exists():
        try:
            grid_output_dir.mkdir()
        except:
            print('cannot make %s ' % grid_output_dir)

    new_fname = grid_output_dir / (title_basename + '.nc')
    if not less_output:
        print('making single file grid netcdf')
        print(str(new_fname))

    grid.to_netcdf(str(new_fname))

    if not less_output:
        print('making 13 file grid netcdf')

    for i in range(13):
        tmp = grid.sel(tile=i)
        tmp2 = title_basename + '_' + str(i).zfill(2) + '.nc'
        new_fname = grid_output_dir / tmp2
        if not less_output:
            print(new_fname)

        tmp.to_netcdf(str(new_fname))

    return grid


# %%
def get_time_steps_from_mds_files(mds_var_dir, mds_file):
    if isinstance(mds_var_dir, str):
        mds_var_dir = Path(mds_var_dir)

    tmp_files = np.sort(list(mds_var_dir.glob(mds_file + '*meta')))

    time_steps = []

    print('get time steps')
    print(tmp_files)
    for i in range(len(tmp_files)):
        time_steps.append(int(tmp_files[i].stem[-10:]))

    return time_steps


# %%

def create_nc_variable_files_on_native_grid_from_mds(mds_var_dir,
                                                     mds_files_to_load,
                                                     mds_grid_dir,
                                                     output_dir,
                                                     output_freq_code,
                                                     vars_to_load='all',
                                                     tiles_to_load=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12],
                                                     time_steps_to_load=[],
                                                     meta_variable_specific=dict(),
                                                     meta_common=dict(),
                                                     mds_datatype='>f4',
                                                     verbose=True,
                                                     method='time_interval_and_combined_tiles',
                                                     less_output=True):
    # %%
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
        if len(time_steps_to_load) == 0:
            # go through each file, pull out the time step, add the time step to a list,
            # and determine the start and end time of each record.

            time_steps_to_load = \
                get_time_steps_from_mds_files(mds_var_dir, mds_file)

        first_meta_fname = mds_file + '.' + \
                           str(time_steps_to_load[0]).zfill(10) + '.meta'

        # get metadata for the first file and determine which variables
        # are present
        meta = xm.utils.parse_meta_file(str(mds_var_dir / first_meta_fname))
        vars_here = meta['fldList']

        if not isinstance(vars_to_load, list):
            vars_to_load = [vars_to_load]

        if 'all' not in vars_to_load:
            num_vars_matching = len(np.intersect1d(vars_to_load, vars_here))

            print('num vars matching ', num_vars_matching)

            # only proceed if we are sure that the variable we want is in this
            # mds file
            if num_vars_matching == 0:
                print('none of the variables you want are in ', mds_file)
                print(vars_to_load)
                print(vars_here)

                break
        # %%

        ecco_dataset_all = \
            load_ecco_vars_from_mds(mds_var_dir, \
                                    mds_grid_dir,
                                    mds_file,
                                    vars_to_load=vars_to_load,
                                    tiles_to_load=tiles_to_load,
                                    time_steps_to_load=time_steps_to_load,
                                    output_freq_code= \
                                        output_freq_code,
                                    meta_variable_specific= \
                                        meta_variable_specific,
                                    meta_common=meta_common,
                                    mds_datatype=mds_datatype,
                                    llc_method='bigchunks',
                                    less_output=less_output)

        if not less_output:
            print('loaded ecco dataset....')
            ecco_dataset_all
            print(ecco_dataset_all.time)

        # loop through time steps, one at a time.
        for time_step in time_steps_to_load:

            i, = np.where(ecco_dataset_all.timestep == time_step)
            if (verbose):
                print(ecco_dataset_all.timestep.values)
                print('time step ', time_step, i)

            # load the dataset
            ecco_dataset = ecco_dataset_all.isel(time=i)

            # pull out the year, month day, hour, min, sec associated with
            # this time step
            if type(ecco_dataset.time.values) == np.ndarray:
                cur_time = ecco_dataset.time.values[0]
            else:
                cur_time = ecco_dataset.time.values

            if (not less_output):
                print(type(cur_time))
                print(cur_time)
                print(time_step)

            year, mon, day, hh, mm, ss = \
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
                print('    ' + var)
                var_ds = ecco_dataset[var]

                # drop these ancillary fields -- they are in grid anyway    
                keys_to_drop = ['CS', 'SN', 'Depth', 'rA', 'PHrefC', 'hFacC', \
                                'maskC', 'drF', 'dxC', 'dyG', 'rAw', 'hFacW', \
                                'rAs', 'hFacS', 'maskS', 'dxG', 'dyC', 'maskW']

                for key_to_drop in keys_to_drop:
                    # print (key_to_drop)
                    if key_to_drop in var_ds.coords.keys():
                        var_ds = var_ds.drop(key_to_drop)
                # %%
                # METHOD 'TIME_INTERVAL_AND_COMBINED_TILES'
                # --> MAKES ONE FILE PER TIME RECORD, KEEPS TILES TOGETHER

                if method == 'time_interval_and_combined_tiles':
                    # create the new file path name
                    if 'MON' in output_freq_code:

                        fname = var + '_' + str(year) + '_' + \
                                str(mon).zfill(2) + '.nc'

                        newpath = output_dir / var / \
                                  str(year)

                    elif ('WEEK' in output_freq_code) or \
                            ('DAY' in output_freq_code):

                        fname = var + '_' + \
                                str(year) + '_' + \
                                str(mon).zfill(2) + '_' + \
                                str(day).zfill(2) + '.nc'
                        d0 = datetime.datetime(year, 1, 1)
                        d1 = datetime.datetime(year, mon, day)
                        doy = (d1 - d0).days + 1

                        if not less_output:
                            print('--- making one file per time record')
                            print(output_dir)

                        newpath = output_dir / var / str(year) / \
                                  str(doy).zfill(3)

                    elif 'YEAR' in output_freq_code:

                        fname = var + '_' + str(year) + '.nc'

                        newpath = output_dir / var / str(year)

                    else:
                        print('no valid output frequency code specified')
                        print('saving to year/mon/day/tile')
                        fname = var + '_' + \
                                str(year) + '_' + \
                                str(mon).zfill(2) + '_' + \
                                str(day).zfill(2) + '.nc'
                        d0 = datetime.datetime(year, 1, 1)
                        d1 = datetime.datetime(year, mon, day)
                        doy = (d1 - d0).days + 1

                        newpath = output_dir / var / \
                                  str(year) / str(doy).zfill(3)

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
                    if (verbose):
                        print('saving to %s' % str(newpath / fname))
                    tmp.to_netcdf(str(newpath / fname), engine='netcdf4')

    return ecco_dataset, tmp, ecco_dataset_all


# create the interpolated fields. Default is on 0.5 degrees by 0.5 degrees.


def create_nc_variable_files_on_regular_grid_from_mds(mds_var_dir,
                                                      mds_files_to_load,
                                                      mds_grid_dir,
                                                      output_dir,
                                                      output_freq_code,
                                                      vars_to_load='all',
                                                      tiles_to_load=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12],
                                                      time_steps_to_load=[],
                                                      meta_variable_specific=dict(),
                                                      meta_common=dict(),
                                                      mds_datatype='>f4',
                                                      dlon=0.5, dlat=0.5,
                                                      radius_of_influence=120000,
                                                      express=1,
                                                      kvarnmidx=2,
                                                      # coordinate idx for vertical axis
                                                      # method now is only a place holder.
                                                      # This can be expanded. For example,
                                                      # the global interpolated fields can
                                                      # split to tiles, similarly to
                                                      # the tiled native fields, to
                                                      # reduce the size of each file.
                                                      less_output=True,
                                                      method=''):
    # %%
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
    if not meta_variable_specific:
       script_dir = os.path.dirname(__file__)  # <-- absolute dir the script is in

       meta_variable_rel_path = '../meta_json/ecco_meta_variable.json'
       abs_meta_variable_path = os.path.join(script_dir, meta_variable_rel_path)
       with open(abs_meta_variable_path, 'r') as fp:
           meta_variable_specific = json.load(fp)

       # --- common meta data
    if not meta_common:
       script_dir = os.path.dirname(__file__)  # <-- absolute dir the script is in
       meta_common_rel_path = '../meta_json/ecco_meta_common.json'
       abs_meta_common_path = os.path.join(script_dir, meta_common_rel_path)
       with open(abs_meta_common_path, 'r') as fp:
           meta_common = json.load(fp)

    # info for the regular grid
    new_grid_min_lat = -90 + dlat / 2.
    new_grid_max_lat = 90 - dlat / 2.
    new_grid_min_lon = -180 + dlon / 2.
    new_grid_max_lon = 180 - dlon / 2.
    new_grid_n_lat = np.int((new_grid_max_lat - new_grid_min_lat) / dlat + 1 + 1e-4 * dlat)
    new_grid_n_lon = np.int((new_grid_max_lon - new_grid_min_lon) / dlon + 1 + 1e-4 * dlon)
    new_lat_1D = new_grid_min_lat + np.asarray(range(new_grid_n_lat)) * dlat
    new_lon_1D = new_grid_min_lon + np.asarray(range(new_grid_n_lon)) * dlon


    if (new_grid_n_lat < 1) or (new_grid_n_lon < 1):
        raise ValueError('You need to have at least one grid point for the new grid.')

    # loop through each mds file in mds_files_to_load
    for mds_file in mds_files_to_load:

        # if time steps to load is empty, load all time steps
        if len(time_steps_to_load) == 0:
            # go through each file, pull out the time step, add the time step to a list,
            # and determine the start and end time of each record.

            time_steps_to_load = \
                get_time_steps_from_mds_files(mds_var_dir, mds_file)

        first_meta_fname = mds_file + '.' + \
                           str(time_steps_to_load[0]).zfill(10) + '.meta'

        # get metadata for the first file and determine which variables
        # are present
        meta = xm.utils.parse_meta_file(str(mds_var_dir / first_meta_fname))
        vars_here = meta['fldList']

        if not isinstance(vars_to_load, list):
            vars_to_load = [vars_to_load]

        if 'all' not in vars_to_load:
            num_vars_matching = len(np.intersect1d(vars_to_load, vars_here))

            print('num vars matching ', num_vars_matching)

            # only proceed if we are sure that the variable we want is in this
            # mds file
            if num_vars_matching == 0:
                print('none of the variables you want are in ', mds_file)
                print(vars_to_load)
                print(vars_here)

                break
        # %%
        # load the MDS fields
        ecco_dataset_all = \
            load_ecco_vars_from_mds(mds_var_dir, \
                                    mds_grid_dir,
                                    mds_file,
                                    vars_to_load=vars_to_load,
                                    tiles_to_load=tiles_to_load,
                                    time_steps_to_load=time_steps_to_load,
                                    output_freq_code= \
                                        output_freq_code,
                                    meta_variable_specific= \
                                        meta_variable_specific,
                                    meta_common=meta_common,
                                    mds_datatype=mds_datatype,
                                    llc_method='bigchunks',
                                    less_output=less_output)

        # do the actual loading. Otherwise, the code may be slow.
        ecco_dataset_all.load()

        # loop through each variable in this dataset,
        for var in ecco_dataset_all.keys():
            print('-- processing ', var)
            # obtain the grid information (use fields from time=0)
            # Note that nrtmp would always equal to one,
            # since each outfile will include only one time-record (e.g. daily, monthly avgs.).

            ecco_dataset = ecco_dataset_all.isel(time=[0])

            var_ds = ecco_dataset[var]

            shapetmp = var_ds.shape
            print('shape of var_ds')
            print(shapetmp)

            lenshapetmp = len(shapetmp)
            nttmp = 0
            nrtmp = 0

            # 2D (time, tile, i j)
            if (lenshapetmp == 4):
                nttmp = shapetmp[0]
                nrtmp = 0

            # 3D (time, depth, tile, i j)
            elif (lenshapetmp == 5):
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

            if 'i_g' in var_ds.coords.keys():
                XX = ecco_dataset['XG']
                XXname = 'XG'
                varmask = 'maskW'

            if 'j_g' in var_ds.coords.keys():
                YY = ecco_dataset['YG']
                YYname = 'YG'
                varmask = 'maskS'

            # interpolation
            # To do it fast, set express==1 (default)
            if (express == 1):
                orig_lons_1d = XX.values.ravel()
                orig_lats_1d = YY.values.ravel()
                orig_grid = pr.geometry.SwathDefinition(lons=orig_lons_1d,
                                                        lats=orig_lats_1d)

                if (new_grid_n_lat > 0) and (new_grid_n_lon > 0):
                    # 1D grid values 
                    new_grid_lon, new_grid_lat = np.meshgrid(new_lon_1D, new_lat_1D)

                    # define the lat lon points of the two parts.
                    new_grid = pr.geometry.GridDefinition(lons=new_grid_lon,
                                                          lats=new_grid_lat)

                    # Get the neighbor info once. 
                    # It will be used repeatedly later to resample data
                    # fast for each of the datasets that is based on 
                    # the same swath, e.g. for a model variable at different times. 
                    valid_input_index, valid_output_index, index_array, distance_array = \
                        pr.kd_tree.get_neighbour_info(orig_grid,
                                                      new_grid, radius_of_influence,
                                                      neighbours=1)

            # loop through time steps, one at a time.
            for time_step in time_steps_to_load:

                i, = np.where(ecco_dataset_all.timestep == time_step)
                if not less_output:
                    print(ecco_dataset_all.timestep.values)
                    print('time step ', time_step, i)

                # load the dataset
                ecco_dataset = ecco_dataset_all.isel(time=i)

                # pull out the year, month day, hour, min, sec associated with
                # this time step
                if type(ecco_dataset.time.values) == np.ndarray:
                    cur_time = ecco_dataset.time.values[0]
                else:
                    cur_time = ecco_dataset.time.values

                year, mon, day, hh, mm, ss = \
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
                if (nttmp != 0 and nrtmp != 0):

                    # make empty array
                    tmpall = np.zeros((nttmp, nrtmp, new_grid_n_lat, new_grid_n_lon))

                    # loop through z
                    for ir in range(nrtmp):  # Z-loop
                        # mask
                        maskloc = ecco_dataset[varmask].values[ir, :]

                        for it in range(nttmp):  # time loop
                            # one 2d field at a time
                            var_ds_onechunk = var_ds[it, ir, :]
                            # apply mask
                            var_ds_onechunk.values[maskloc == 0] = np.nan
                            orig_field = var_ds_onechunk.values

                            if (express == 1):
                                tmp = pr.kd_tree.get_sample_from_neighbour_info(
                                    'nn', new_grid.shape, orig_field,
                                    valid_input_index, valid_output_index,
                                    index_array)

                            else:
                                new_grid_lon, new_grid_lat, tmp = \
                                    resample_to_latlon(XX, YY, orig_field,
                                                       new_grid_min_lat,
                                                       new_grid_max_lat, dlat,
                                                       new_grid_min_lon,
                                                       new_grid_max_lon, dlon,
                                                       nprocs_user=1,
                                                       mapping_method='nearest_neighbor',
                                                       radius_of_influence=radius_of_influence)

                            # put the interpolated 2D array into the big array
                            tmpall[it, ir, :] = tmp

                # 2d fields (without Z-axis) for each time record      
                elif (nttmp != 0):

                    tmpall = np.zeros((nttmp, new_grid_n_lat, new_grid_n_lon))
                    # mask
                    maskloc = ecco_dataset[varmask].values[0, :]

                    for it in range(nttmp):  # time loop
                        var_ds_onechunk = var_ds[it, :]
                        var_ds_onechunk.values[maskloc == 0] = np.nan
                        orig_field = var_ds_onechunk.values

                        if (express == 1):
                            tmp = pr.kd_tree.get_sample_from_neighbour_info(
                                'nn', new_grid.shape, orig_field,
                                valid_input_index, valid_output_index,
                                index_array)
                        else:
                            new_grid_lon, new_grid_lat, tmp = \
                                resample_to_latlon(XX, YY, orig_field,
                                                   new_grid_min_lat,
                                                   new_grid_max_lat, dlat,
                                                   new_grid_min_lon,
                                                   new_grid_max_lon, dlon,
                                                   nprocs_user=1,
                                                   mapping_method='nearest_neighbor',
                                                   radius_of_influence=radius_of_influence)
                        tmpall[it, :] = tmp

                else:
                    print('Error! both nttmp and nrtmp are zeros.')
                    sys.exit()

                # set the coordinates for the new (regular) grid
                # 2d fields
                if (nrtmp == 0):
                    var_da_reg = xr.DataArray(tmpall,
                                              coords={'time': var_ds.coords['time'].values,
                                                      'latitude': new_lat_1D,
                                                      'longitude': new_lon_1D},
                                              dims=('time', 'latitude', 'longitude'))

                # 3d fields
                elif (nrtmp > 0):

                    var_coords = list(var_ds.coords.keys())

                    if 'k' in var_coords:
                        kvarnm = 'k'
                    elif 'k_l' in var_coords:
                        kvarnm = 'k_l'
                    elif 'k_u' in var_coords:
                        kvarnm = 'k_u'
                    else:
                        print("no valid k index name found")
                        sys.exit()

                    var_da_reg = xr.DataArray(tmpall,
                                              coords={'time': var_ds.coords['time'].values,
                                                      kvarnm: var_ds.coords[kvarnm].values,
                                                      'latitude': new_lat_1D,
                                                      'longitude': new_lon_1D},
                                              dims=('time', kvarnm, 'latitude', 'longitude'))


                print('make 1D latlon bounds')
                lat_bnds_da, lon_bnds_da = make_1D_latlon_bounds_from_ecco_dataset(var_da_reg, dlat, dlon)

                print('pre-merge')
                print(var_da_reg)
                print(lat_bnds_da)
                print(lon_bnds_da)
                #var_da_reg.assign_coords(lat_bnds_da)
                #var_da_reg.assign_coords(lon_bnds_da)


                # keys_to_drop = ['tile','j','i','XC','YC','XG','YG']
                # drop these ancillary fields -- they are in grid anyway    
                keys_to_drop = ['CS', 'SN', 'Depth', 'rA', 'PHrefC', 'hFacC', \
                                'maskC', 'drF', 'dxC', 'dyG', 'rAw', 'hFacW', \
                                'rAs', 'hFacS', 'maskS', 'dxG', 'dyC', 'maskW', \
                                'tile', 'XC', 'YC', 'XG', 'YG']

                for key_to_drop in keys_to_drop:
                    # print (key_to_drop)
                    if key_to_drop in var_ds.coords.keys():
                        var_ds = var_ds.drop(key_to_drop)

                # any remaining fields, e.g. time, would be included in the interpolated fields.

                for key_to_add in var_ds.coords.keys():
                    if (key_to_add not in var_da_reg.coords.keys()):
                        if (key_to_add != 'i_g' and key_to_add != 'j_g' and key_to_add != 'i' and key_to_add != 'j'):
                            var_da_reg[key_to_add] = var_ds[key_to_add]

                # use the same global attributs                
                var_da_reg.attrs = var_ds.attrs

                # create the new file path name
                if 'MON' in output_freq_code:

                    fname = var + '_' + str(year) + '_' + str(mon).zfill(2) + '.nc'
                    newpath = output_dir / var / str(year)

                elif ('WEEK' in output_freq_code) or \
                        ('DAY' in output_freq_code):

                    fname = var + '_' + \
                            str(year) + '_' + \
                            str(mon).zfill(2) + '_' + \
                            str(day).zfill(2) + '.nc'
                    d0 = datetime.datetime(year, 1, 1)
                    d1 = datetime.datetime(year, mon, day)
                    doy = (d1 - d0).days + 1

                    newpath = output_dir / var / str(year) / str(doy).zfill(3)

                elif 'YEAR' in output_freq_code:

                    fname = var + '_' + str(year) + '.nc'

                    newpath = output_dir / var / str(year)

                else:
                    print('no valid output frequency code specified')
                    print('saving to year/mon/day/tile')
                    fname = var + '_' + \
                            str(year) + '_' + \
                            str(mon).zfill(2) + '_' + \
                            str(day).zfill(2) + '.nc'
                    d0 = datetime.datetime(year, 1, 1)
                    d1 = datetime.datetime(year, mon, day)
                    doy = (d1 - d0).days + 1

                    newpath = output_dir / var / str(year) / str(doy).zfill(3)

                # create the path if it does not exist/
                if not newpath.exists():
                    newpath.mkdir(parents=True, exist_ok = True)

                # convert the data array to a dataset.


                tmp = var_da_reg.to_dataset()

                # add the time bounds field back in if we have an
                # average field
                if 'AVG' in output_freq_code:
#                    tmp = xr.merge((tmp, tb))
#                    tmp = tmp.drop('tb')
                    tmp.attrs['cell_method'] = meta_common['cell_methods']['interp_mn']
                else:
                    tmp.attrs['cell_method'] = meta_common['cell_methods']['interp_snapshot']

                # put the metadata back in
                tmp.attrs = ecco_dataset.attrs

                # update the temporal and geospatial metadata
                tmp = update_ecco_dataset_geospatial_metadata(tmp)
                tmp = update_ecco_dataset_temporal_coverage_metadata(tmp)

                tmp.latitude.attrs.update(meta_common["latitude-with-bounds"])
                tmp.longitude.attrs.update(meta_common["longitude-with-bounds"])

                # save to netcdf.  it's that simple.
                if not less_output:
                    print('saving to %s' % newpath + '/' + fname)
                # do not include _FillValue
                encoding = {i: {'_FillValue': False} for i in tmp.variables.keys()}
                encoding['time']['units'] = 'days since 1992-01-01 00:00:00'
                encoding['time_bnds']['calendar'] = "proleptic_gregorian"

                tmp.to_netcdf(str(newpath / fname), engine='netcdf4', encoding=encoding)

        ecco_dataset_all.close()
    return ecco_dataset, tmp


##%%
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
            ecco_dataset.attrs['time_coverage_end'] = \
                str(ecco_dataset.time_bnds.values[1])[0:19]

        else:
            # if there are many time bounds
            ecco_dataset.attrs['time_coverage_start'] = \
                str(ecco_dataset.time_bnds.values[0][0])[0:19]
            ecco_dataset.attrs['time_coverage_end'] = \
                str(ecco_dataset.time_bnds.values[-1][-1])[0:19]
    # elif 'time' in ecco_dataset.coords.keys():
    #    ecco_dataset.attrs['time_coverage_start'] = str(ecco_dataset.time.values[0])[0:19]
    #    ecco_dataset.attrs['time_coverage_end']   = str(ecco_dataset.time.values[-1])[0:19]

    return ecco_dataset


# %%
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
    if 'YG' in ecco_dataset.coords.keys():
        ecco_dataset.attrs['geospatial_lat_max'] = ecco_dataset.YG.values.max()
        ecco_dataset.attrs['geospatial_lat_min'] = ecco_dataset.YG.values.min()
        ecco_dataset.attrs['nx'] = ecco_dataset.YG.shape[-2]
        ecco_dataset.attrs['ny'] = ecco_dataset.YG.shape[-1]

    elif 'YC' in ecco_dataset.coords.keys():
        ecco_dataset.attrs['geospatial_lat_max'] = ecco_dataset.YC.values.max()
        ecco_dataset.attrs['geospatial_lat_min'] = ecco_dataset.YC.values.min()
        ecco_dataset.attrs['nx'] = ecco_dataset.YC.shape[-2]
        ecco_dataset.attrs['ny'] = ecco_dataset.YC.shape[-1]
    elif 'latitude' in ecco_dataset.coords.keys():
        ecco_dataset.attrs['geospatial_lat_max'] = ecco_dataset.latitude.values.max()
        ecco_dataset.attrs['geospatial_lat_min'] = ecco_dataset.latitude.values.min()

        if len(ecco_dataset.latitude.dims) == 1:
            ecco_dataset.attrs['nx'] = len(ecco_dataset.latitude)
        else:
            ecco_dataset.attrs['nx'] = ecco_dataset.latitude.shape[0]
            ecco_dataset.attrs['ny'] = ecco_dataset.latitude.shape[1]

    if 'XG' in ecco_dataset.coords.keys():
        ecco_dataset.attrs['geospatial_lon_max'] = ecco_dataset.XG.values.max()
        ecco_dataset.attrs['geospatial_lon_min'] = ecco_dataset.XG.values.min()
    elif 'XC' in ecco_dataset.coords.keys():
        ecco_dataset.attrs['geospatial_lon_max'] = ecco_dataset.XC.values.max()
        ecco_dataset.attrs['geospatial_lon_min'] = ecco_dataset.XC.values.min()
    elif 'longitude' in ecco_dataset.coords.keys():
        ecco_dataset.attrs['geospatial_lon_max'] = ecco_dataset.longitude.values.max()
        ecco_dataset.attrs['geospatial_lon_min'] = ecco_dataset.longitude.values.min()

        if len(ecco_dataset.longitude.dims) == 1:
            ecco_dataset.attrs['nx'] = len(ecco_dataset.longitude)
        else:
            ecco_dataset.attrs['nx'] = ecco_dataset.longitude.shape[0]
            ecco_dataset.attrs['ny'] = ecco_dataset.longitude.shape[1]

    ecco_dataset.attrs["geospatial_lat_units"]: "degrees_north"
    ecco_dataset.attrs["geospatial_lon_units"]: "degrees_east"


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
        #        ecco_dataset.attrs['geospatial_vertical_max'] = 0
        #        ecco_dataset.attrs['geospatial_vertical_min'] = 0
        #        ecco_dataset.attrs['geospatial_vertical_min'] = 0

        ecco_dataset.attrs['nz'] = 1

    return ecco_dataset
