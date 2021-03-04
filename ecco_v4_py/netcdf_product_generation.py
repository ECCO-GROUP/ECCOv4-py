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
import netCDF4 as nc4
from collections import OrderedDict
import dask as dask
from dask import delayed

from .read_bin_llc import load_ecco_vars_from_mds
from .ecco_utils import extract_yyyy_mm_dd_hh_mm_ss_from_datetime64
from .resample_to_latlon import resample_to_latlon

#%%
def sort_attrs(attrs):
    od = OrderedDict()

    keys = sorted(list(attrs.keys()),key=str.casefold)

    for k in keys:
        od[k] = attrs[k]

    return od

#%%


def create_native_grid_netcdf_files(mds_grid_dir, mds_var_dir, mds_filename,
                                    mds_freq_code,
                                    vars_to_load,
                                    dataset_name = 'by_variable',
                                    time_steps_to_load = 'all',
                                    tiles_to_load = 'all',
                                    output_array_precision = np.float32,
                                    global_metadata = 'default',
                                    coordinate_metadata = 'default',
                                    geometry_metadata = 'default',
                                    variable_metadata = 'default'):


    # if no specific file data passed, read default metadata from json file
    # -- variable specific meta data
    script_dir = Path(__file__).resolve().parent

    #if not meta_variable_specific:
    #    meta_variable_rel_path = '../meta_json/ecco_meta_variable.json'
    #    abs_meta_variable_path = os.path.join(script_dir, meta_variable_rel_path)
    #    with open(abs_meta_variable_path, 'r') as fp:
    #        meta_variable_specific = json.load(fp)


    ## METADATA
    metadata_json_dir = Path('/home/ifenty/git_repos_others/ECCO-GROUP/ECCO-ACCESS/metadata/ECCOv4r4_metadata_json')

    metadata_fields = ['ECCOv4_global_metadata_for_all_datasets',
                       'ECCOv4_global_metadata_for_latlon_datasets',
                       'ECCOv4_global_metadata_for_native_datasets',
                       'ECCOv4rcoordinate_metadata_for_native_datasets',
                       'ECCOv4r4_geometry_metadata_for_native_datasets',
                       'ECCOv4r4_variable_metadata']

    print('\nLOADING METADATA')
    # load METADATA
    metadata = dict()

    for mf in metadata_fields:
        mf_e = mf + '.json'
        print(mf_e)
        with open(str(metadata_json_dir / mf_e), 'r') as fp:
            metadata[mf] = json.load(fp)


    # metadata for different variables
    global_metadata_for_all_datasets = metadata['ECCOv4r4_global_metadata_for_all_datasets']
    global_metadata_for_latlon_datasets = metadata['ECCOv4r4_global_metadata_for_latlon_datasets']
    global_metadata_for_native_datasets = metadata['ECCOv4r4_global_metadata_for_native_datasets']

    coordinate_metadata_for_1D_datasets = metadata['ECCOv4r4_coordinate_metadata_for_1D_datasets']
    coordinate_metadata_for_latlon_datasets = metadata['ECCOv4r4_coordinate_metadata_for_latlon_datasets']
    coordinate_metadata_for_native_datasets = metadata['ECCOv4r4_coordinate_metadata_for_native_datasets']

    geometry_metadata_for_latlon_datasets = metadata['ECCOv4r4_geometry_metadata_for_latlon_datasets']
    geometry_metadata_for_native_datasets = metadata['ECCOv4r4_geometry_metadata_for_native_datasets']

    groupings_for_1D_datasets = metadata['ECCOv4r4_groupings_for_1D_datasets']
    groupings_for_latlon_datasets = metadata['ECCOv4r4_groupings_for_latlon_datasets']
    groupings_for_native_datasets = metadata['ECCOv4r4_groupings_for_native_datasets']

    variable_metadata_latlon = metadata['ECCOv4r4_variable_metadata_for_latlon_datasets']
    variable_metadata = metadata['ECCOv4r4_variable_metadata']

    global_metadata = global_metadata_for_all_datasets + global_metadata_for_native_datasets

    variable_metadata_combined = variable_metadata + geometry_metadata_for_native_datasets


    #  #variable_metadata = variable_metadata + geometry_metadata_for_native_datasets
    short_mds_name = 'ETAN_mon_mean'
    output_freq_code= 'AVG_MON'
    cur_ts = 'all'
    vars_to_load = 'ETAN' #['ETAN']
    F_DS = []

    ecco_grid =  load_ecco_vars_from_mds(str(mds_grid_dir),
                                    str(mds_grid_dir),
                                    '',
                                    vars_to_load = 'all',
                                    drop_unused_coords = False,
                                    grid_vars_to_coords = False,
                                    coordinate_metadata = coordinate_metadata_for_native_datasets,
                                    variable_metadata = geometry_metadata_for_native_datasets,
                                    global_metadata = global_metadata,
                                    less_output=False).load()


    F_DS = \
      load_ecco_vars_from_mds(mds_var_dir,\
                                   mds_grid_dir = mds_grid_dir, \
                                   mds_files = short_mds_name,\
                                   vars_to_load = vars_to_load,
                                   drop_unused_coords = True,\
                                   grid_vars_to_coords = False,\
                                   variable_metadata = variable_metadata_combined,
                                   coordinate_metadata = coordinate_metadata_for_native_datasets,
                                   #global_metadata = [[global_metadata]],
                                   output_freq_code=output_freq_code,\
                                   model_time_steps_to_load=cur_ts,
                                   less_output = True)

    print(F_DS)
    #  vars_to_drop = set(F_DS.data_vars).difference(set([var]))
    #  F_DS.drop_vars(vars_to_drop)


    save_ecco_dataset_to_netcdf(F_DS.isel(time=[0]),
                                    Path('/home/ifenty/tmp/'),
                                    dataset_name = 'by_variable',
                                    time_method = 'by_year',
                                    output_freq_code='AVG_MON')




#%%%%%%%%%%%%%%%%%%%%
def save_ecco_dataset_to_netcdf(ecco_ds,
                                output_dir,
                                dataset_name = 'by_variable',
                                time_method = 'by_record',
                                output_array_precision = np.float32,
                                output_freq_code=None):

    """Saves an ECCO dataset to one or more NetCDF files

    NetCDF files will be written with the following options
    -------------------------------------------------------
    * compression level 5
    * shuffle = True
    * zlib = True

    Parameters
    ----------
    ecco_ds: xarray DataSet
        the DataSet to save.  Can have one or more 'data variables'

    output_dir: String
        root directory for saved files.  New files will be saved in
        a (new) subdirectory of output_dir

    dataset_name : String, optional.  Default 'by_variable'
        name to use for NetCDF files.  'by_variable' will create a
        name based on the data variables present by concatenating
        all data variable names together separated by '_'
        For example, if ecco_ds has both 'ETAN' and 'SSH' the
        dataset_name will be 'ETAN_SSH'

	time_method : String, optional. Default 'by_record'
        options include
            'by_record' - one file per time level
            'by_year'   - one file per calendar year

    output_array_precision : numpy type. Default np.float32
        precision to use when saving data variables of type float
        options include
            np.float32
            np.float64

    output_freq_code: String, optional. Default = None
        a string code indicating the time level of averaging of the
        data variables
        options include
            'AVG_MON'  - monthly-averaged file
            'AVG_DAY'  - daily-averaged files
            'SNAP'     - snapshot files (instantaneous)

    RETURNS:
    ----------
    nothing.  files should be saved to disk

    """


    # Create a name of the files if not specified
    # ---------------------------------------------
    if dataset_name =='by_variable':
        # concat all data variables together into a single string
        dataset_name = '_'.join(list(ecco_ds.data_vars))


    # force load coordinate values in case they are in dask array
    # -----------------------------------------------------------
    for coord in ecco_ds.coords:
        ecco_ds[coord].load()


    # Define fill values for NaN
    # ---------------------------------------------
    if output_array_precision == np.float32:
        netcdf_fill_value = nc4.default_fillvals['f4']

    elif output_array_precision == np.float64:
        netcdf_fill_value = nc4.default_fillvals['f8']


    # Create NetCDF encoding directives
    # ---------------------------------------------
    print('\n... creating variable encodings')
    # ... data variable encoding directives
    dv_encoding = dict()
    for dv in ecco_ds.data_vars:
        dv_encoding[dv] =  {'zlib':True, \
                            'complevel':5,\
                            'shuffle':True,\
                            '_FillValue':netcdf_fill_value}

    # ... coordinate encoding directives
    print('\n... creating coordinate encodings')
    coord_encoding = dict()
    for coord in ecco_ds.coords:
        # set default no fill value for coordinate
        if output_array_precision == np.float32:
            coord_encoding[coord] = {'_FillValue':None, 'dtype':'float32'}
        elif output_array_precision == np.float64:
            coord_encoding[coord] = {'_FillValue':None, 'dtype':'float64'}

        # force 64 bit ints to be 32 bit ints
        if (ecco_ds[coord].values.dtype == np.int32) or \
           (ecco_ds[coord].values.dtype == np.int64) :
            coord_encoding[coord]['dtype'] ='int32'

        # fix encoding of time
        if coord == 'time' or coord == 'time_bnds':
            coord_encoding[coord]['dtype'] ='int32'

            if 'units' in ecco_ds[coord].attrs:
                # apply units as encoding for time
                coord_encoding[coord]['units'] = ecco_ds[coord].attrs['units']
                # delete from the attributes list
                del ecco_ds[coord].attrs['units']

        elif coord == 'time_step':
            coord_encoding[coord]['dtype'] ='int32'

    # ... combined data variable and coordinate encoding directives
    encoding = {**dv_encoding, **coord_encoding}


    # Create directory for output files
    # ---------------------------------------------
    filepath = output_dir  / dataset_name

    if not filepath.exists():
        filepath.mkdir(parents=True, exist_ok=True)


    # Determine output freqency code.
    # ---------------------------------------------
    # user can specify directory or it can be found if the dataset
    # has the 'time_coverage_resolution' global attribute
    if output_freq_code == None:
        if 'time_coverage_resolution' in ecco_ds.attrs:

            print('dataset time averaging from metadata')
            time_coverage_resolution = ecco_ds.attrs['time_coverage_resolution']
            if time_coverage_resolution == 'P1M':
                output_freq_code='AVG_MON'
            elif time_coverage_resolution == 'P1D':
                output_freq_code='AVG_DAY'
            elif time_coverage_resolution == 'P0S':
                output_freq_code='SNAP'
        else:
            print('output_freq_code not defined and not available in dataset metadata')
            print('... using full record time in filename')


    # Write records to disk as NetCDF
    # ---------------------------------------------
    # one file per time level

    if time_method == 'by_record':
        for time_i, rec_time in enumerate(ecco_ds.time):

            cur_ds = ecco_ds.isel(time=time_i)

            # cast data variables to desired precision (if necessary)
            #for data_var in cur_ds.data_vars:
            #    if cur_ds[data_var].values.dtype != output_array_precision:
            #        cur_ds[data_var].values = cur_ds[data_var].astype(output_array_precision)

            time_date_info  =\
                make_date_str_from_dt64(cur_ds.time.values, output_freq_code)

           # sort comments alphabetically
            print('\n... sorting global attributes')
            cur_ds.attrs = sort_attrs(cur_ds.attrs)

            # add one final comment (PODAAC request)
            cur_ds.attrs["coordinates_comment"] = \
                "Note: the global 'coordinates' attribute descibes auxillary coordinates."

            fname = dataset_name + '_' + time_date_info['short'] +\
                    '_' + time_date_info['ppp_tttt'] + '.nc'

            print(fname)
            print(cur_ds)
            netcdf_output_filename = filepath / fname

            # SAVE
            print('\n... saving to netcdf ', netcdf_output_filename)
            cur_ds.to_netcdf(netcdf_output_filename, encoding=encoding)
            cur_ds.close()

    # one file per year
    elif time_method == 'by_year':
        unique_years = np.unique(ecco_ds.time.dt.year)
        print(unique_years)

        for year in unique_years:
            # pull out only records for this year
            cur_ds = ecco_ds.sel(time=slice(str(year), str(year)))

            first_time = cur_ds.time.values[0]
            last_time = cur_ds.time.values[-1]

            first_time_date_info =\
                make_date_str_from_dt64(first_time, output_freq_code)

            last_time_date_info =\
                make_date_str_from_dt64(last_time, output_freq_code)

           # sort comments alphabetically
            print('\n... sorting global attributes')
            cur_ds.attrs = sort_attrs(cur_ds.attrs)

            # add one final comment (PODAAC request)
            cur_ds.attrs["coordinates_comment"] = \
                "Note: the global 'coordinates' attribute descibes auxillary coordinates."

            fname = dataset_name + '_' +\
                first_time_date_info['short'] + '_' +\
                last_time_date_info['short'] + '_' +\
                first_time_date_info['ppp_tttt']+ '.nc'

            print(fname)
            print(cur_ds)
            netcdf_output_filename = filepath / fname

            # SAVE
            print('\n... saving to netcdf ', netcdf_output_filename)
            cur_ds.to_netcdf(netcdf_output_filename, encoding=encoding)
            cur_ds.close()


#%%%%
def make_date_str_from_dt64(dt64_time, output_freq_code):
    """Extracts components of a numpy date time 64 object

    Parameters
    ----------
    dt_64_time: numpy.datetime64
        a single datetime64 object

    output_freq_code: string
        a string code indicating the time level of averaging of the
        data variables
        options include
            'AVG_MON'  - monthly-averaged file
            'AVG_DAY'  - daily-averaged files
            'SNAPSHOT' - snapshot files (instantaneous)

    RETURNS:
    ----------
    a dictionary with the following string entries (all zero padded)
        date_str_full  : YYYY-MM-DDTHHMMSS
        date_str_short : YYYY-MM    (for AVG_MON)
                         YYYY-MM-DD (for AVG_DAY)
                         YYYY-MM-DDTHHMMSS (for SNAP)
        year           : YYYY
        month          : MM
        day            : DD
        hour           : HH
        ppp_tttt  : one of 'mon_mean','day_mean','snap'

    """

    print(dt64_time)
    date_str_full = str(dt64_time)[0:19].replace(':','')
    year  = date_str_full[0:4]
    month = date_str_full[5:7]
    day   = date_str_full[8:10]
    hour  = date_str_full[11:13]

    print(year, month, day, hour)
    ppp_tttt = ""
    date_str_short =""

    if output_freq_code == 'AVG_MON':
        date_str_short = str(np.datetime64(dt64_time,'M'))
        ppp_tttt = 'mon_mean'

    # --- AVG DAY
    elif output_freq_code == 'AVG_DAY':
        date_str_short = str(np.datetime64(dt64_time,'D'))
        ppp_tttt = 'day_mean'

    # --- SNAPSHOT
    elif 'SNAP' in output_freq_code:
        # convert from oroginal
        #   '1992-01-16T12:00:00.000000000'
        # to new format
        # '1992-01-16T120000'
        date_str_short = str(dt64_time)[0:19].replace(':','')
        ppp_tttt = 'snap'

    date_str = dict()
    date_str['full'] = date_str_full
    date_str['short'] = date_str_short
    date_str['year']  = year
    date_str['month'] = month
    date_str['day'] = day
    date_str['hour'] = hour
    date_str['ppp_tttt'] = ppp_tttt

    return date_str


#%%
def create_nc_grid_files_on_native_grid_from_mds(grid_input_dir,
                                                 grid_output_dir,
                                                 coordinate_metadata = dict(),
                                                 geometry_metadata = dict(),
                                                 global_metadata = dict(),
                                                 cell_bounds = None,
                                                 file_basename='ECCO-GRID',
                                                 title='llc grid geometry',
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



