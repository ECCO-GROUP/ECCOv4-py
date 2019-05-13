"""
ECCO v4 Python: read_bin_llc

This module includes utility routines for loading binary files in the 
llc 13-tile native flat binary layout.  This layout is the default for 
MITgcm input and output for global setups using lat-lon-cap (llc) layout. 
The llc layout is used for ECCO v4. 

.. _ecco_v4_py Documentation :
   https://github.com/ECCO-GROUP/ECCOv4-py

"""

from __future__ import division,print_function
from xmitgcm import open_mdsdataset
import xmitgcm as xm
import xmitgcm
import numpy as np
import xarray as xr
import time

from .llc_array_conversion  import llc_compact_to_tiles, \
    llc_compact_to_faces, llc_faces_to_tiles, llc_faces_to_compact, \
    llc_tiles_to_faces, llc_tiles_to_compact

from .ecco_utils import make_time_bounds_and_center_times_from_ecco_dataset


def load_ecco_vars_from_mds(mds_var_dir, 
                            mds_files, 
                            mds_grid_dir,
                            vars_to_load = 'all', 
                            tiles_to_load = range(13),
                            model_time_steps_to_load = 'all',
                            output_freq_code = '', 
                            meta_variable_specific=dict(),
                            meta_common=dict(),
                            mds_datatype = '>f4',
                            llc_method = 'bigchunks',
                            less_output=True):
                                 
    """

    Uses xmitgcm's *open_mdsdataset* routine to load ecco variable(s) from 
    MITgcm's MDS binary output into xarray Dataset/DataArray objects.  

    The main benefit of using this routine over open_mdsdataset is that this
    routine allows for 
        
        - proper centering of the *time* variable for time-averaged fields

        - creation of the *time-bnds* fields in time-averaged fields

        - specification of extra variable-specific and globl metadata


    xmitgcm.open_mdsdataset uses the model step number from the file name 
    (e.g., 732 from the file VAR.000000000732.data) to construct the 
    'time' field.  For time-averaged fields, this model step 
    corresponds to END of the averaging period, not the time averaging mid 
    point. This routine fixes the 'time' field of time-averaged fields
    to be the mid point of the time averaging period when the appropriate
    *output_freq_code* is passed.

    
    Parameters
    ----------
    mds_var_dir : str
        directory where the .data/.meta files are stored
    
    mds_files   : str or list
        a string or list of file names to load.  
        Note :  the name is everything BEFORE the time step
        the mds_file name for 'var.000000000732.data' is 'var'
    
    mds_grid_dir : str
        the directory where the model binary (.data) grid fields
        are stored
    
    vars_to_load : str or list, optional, default 'all'
        a string or list of the variable names to read from the mds_files

        - if 'all' then all variables in the files are loaded

    tiles_to_load : int or list of ints, optional, default range(13)
        an int or list of ints indicating which tiles to load

    model_time_steps_to_load : int or list of ints, optional, default 'all'
        an int or list of ints indicating which model time steps to load

        Note : the model time step indicates the time step when the the file was written.
        when the field is a time average, this time step shows the END of the averaging period.

    output_freq_code : str, optional, default empty string
        a code used to create the proper time indices on the fields after loading
        ('AVG' or 'SNAPSHOT') + '_' + ('DAY','WEEK','MON', or 'YEAR')
        
        valid options :

        - AVG_DAY, AVG_WEEK, AVG_MON, AVG_YEAR
        - SNAPSHOT_DAY, SNAPSHOT_WEEK, SNAPSHOT_MON, SNAPSHOT_YEAR
        
    meta_variable_specific : dict, optional, default empty dictionary
        a dictionary with variable-specific metadata.  used when creating
        the offical ECCO products

    meta_common : dict, optional, default empty dictionary
        a dictionary with globally-valid metadata for the ECCO fields.
        useful when creating the offical ECCO netcdf fields

    mds_datatype : string, optional, default '>f4'
        code indicating what type of field to load if the xmitgcm cannot 
        determine the format from the .meta file.  '>f4' means big endian
        32 bit float.

    llc_method : string, optional, default 'big_chunks'
        refer to the xmitgcm documentation.
    
    less_output : logical, optional
        if True (default), omit additional print statements

    Returns
    =======

    ecco_dataset : xarray Dataset


    """


    #ECCO v4 r3 starts 1992/1/1 12:00:00
    ecco_v4_start_year = 1992
    ecco_v4_start_mon  = 1
    ecco_v4_start_day  = 1
    ecco_v4_start_hour = 12
    ecco_v4_start_min  = 0
    ecco_v4_start_sec  = 0
    
    # ECCO v4 r3 has 1 hour (3600 s) time steps    
    delta_t = 3600
    
    # define reference date for xmitgcm
    ref_date = str(ecco_v4_start_year) + '-' + str(ecco_v4_start_mon)  + '-' + \
        str(ecco_v4_start_day)  + ' ' +  str(ecco_v4_start_hour) + ':' +  \
        str(ecco_v4_start_min)  + ':' + str(ecco_v4_start_sec)


    if model_time_steps_to_load == 'all':

        ecco_dataset = open_mdsdataset(data_dir = mds_var_dir, 
                                       grid_dir = mds_grid_dir,
                                       read_grid = True,
                                       prefix = mds_files,
                                       geometry = 'llc', 
                                       iters = 'all',
                                       ref_date = ref_date, 
                                       delta_t  = delta_t,
                                       default_dtype = np.dtype(mds_datatype),
                                       grid_vars_to_coords=True,
                                       llc_method = llc_method)
    
    else:
        if isinstance(model_time_steps_to_load, int):
            iters_to_load = [iters_to_load]
        
        if isinstance(model_time_steps_to_load, list):
            ecco_dataset = open_mdsdataset(data_dir = mds_var_dir, 
                                           grid_dir = mds_grid_dir,
                                           read_grid = True,
                                           prefix = mds_files, 
                                           geometry = 'llc', 
                                           iters = model_time_steps_to_load,
                                           ref_date = ref_date, 
                                           delta_t = delta_t,
                                           default_dtype = np.dtype(mds_datatype),
                                           grid_vars_to_coords=True,
                                           llc_method=llc_method)
        else:
            print ('not a valid iters_to_load.  must be "all", an "int", or a list of "int"')
            return []


    # replace the xmitgcm coordinate name of 'FACE' with 'TILE'
    if 'face' in ecco_dataset.coords.keys():
        ecco_dataset = ecco_dataset.rename({'face': 'tile'})
        ecco_dataset.tile.attrs['standard_name'] = 'tile_index'
       
    # if vars_to_load is an empty list, keep all variables.  otherwise,
    # only keep those variables in the vars_to_load list.
    
    vars_ignored = []
    vars_loaded = []
    
    if not isinstance(vars_to_load, list):
        vars_to_load = [vars_to_load]

    if not less_output:
        print ('vars to load ', vars_to_load)
    
    if 'all' not in vars_to_load:
        if not less_output:
            print ('loading subset of variables: ', vars_to_load)
    
        # remove variables that are not on the vars_to_load_list
        for ecco_var in ecco_dataset.keys():
            
            if ecco_var not in vars_to_load:
                vars_ignored.append(ecco_var)
                ecco_dataset = ecco_dataset.drop(ecco_var)
            
            else:
                vars_loaded.append(ecco_var)

        if not less_output:
            print ('loaded  : ', vars_loaded)
            print ('ignored : ', vars_ignored)
    
    else:
        if not less_output:
            print ('loaded all variables  : ', ecco_dataset.keys())
        
    # keep tiles in the 'tiles_to_load' list.
    if not isinstance(tiles_to_load, list):
        tiles_to_load = [tiles_to_load]

    ecco_dataset = ecco_dataset.sel(tile = tiles_to_load)
    
    #ecco_dataset = ecco_dataset.isel(time=0)
    if 'AVG' in output_freq_code and \
        'time_bnds' not in ecco_dataset.keys():

        time_bnds_ds, center_times = \
            make_time_bounds_and_center_times_from_ecco_dataset(ecco_dataset,\
                                                                output_freq_code)
        
        if isinstance(ecco_dataset.time.values, np.datetime64):
            ecco_dataset.time.values = center_times
        
        elif isinstance(center_times, np.datetime64):
            center_times = np.array(center_times)
            ecco_dataset.time.values[:] = center_times
        
        if 'ecco-v4-time-average-center-no-units' in meta_common:
            ecco_dataset.time.attrs = \
                meta_common['ecco-v4-time-average-center-no-units']
          
        ecco_dataset = xr.merge((ecco_dataset, time_bnds_ds))
        
        if 'time_bnds-no-units' in meta_common:
            ecco_dataset.time_bnds.attrs=meta_common['time_bnds-no-units']
      
        ecco_dataset = ecco_dataset.set_coords('time_bnds')
        
    elif  'SNAPSHOT' in output_freq_code:
         if 'ecco-v4-time-snapshot-no-units' in meta_common:
            ecco_dataset.time.attrs = \
                meta_common['ecco-v4-time-snapshot-no-units']
       

    #%% DROP SOME EXTRA FIELDS THAT DO NOT NEED TO BE IN THE DATASET
    if 'maskCtrlS' in ecco_dataset.coords.keys():
        ecco_dataset=ecco_dataset.drop('maskCtrlS')
    if 'maskCtrlW' in ecco_dataset.coords.keys():
        ecco_dataset=ecco_dataset.drop('maskCtrlW')
    if 'maskCtrlC' in ecco_dataset.coords.keys():
        ecco_dataset=ecco_dataset.drop('maskCtrlC')
        
    # UPDATE THE VARIABLE SPECIFIC METADATA USING THE 'META_VARSPECIFIC' DICT.
    # if it exists
    for ecco_var in ecco_dataset.variables.keys():
        if ecco_var in meta_variable_specific.keys():
            ecco_dataset[ecco_var].attrs = meta_variable_specific[ecco_var]


    #%% UPDATE THE GLOBAL METADATA USING THE 'META_COMMON' DICT, if it exists
    ecco_dataset.attrs = dict()

    if 'ecco-v4-global' in meta_common:
        ecco_dataset.attrs.update(meta_common['ecco-v4-global'])

    if 'k' in ecco_dataset.dims.keys() and \
        'ecco-v4-global-3D' in meta_common:
        ecco_dataset.attrs.update(meta_common['ecco-v4-global-3D'])
   
    ecco_dataset.attrs['date_created'] = time.ctime()
    
    # give it a hug?
    ecco_dataset = ecco_dataset.squeeze()

    return ecco_dataset



#%%
def read_llc_to_tiles(fdir, fname, llc=90, skip=0, nk=1, nl=1, 
                      filetype = '>f4', less_output = True):
    """
    Loads an MITgcm binary file in the 'tiled' format of the 
    lat-lon-cap (LLC) grids via xmitgcm.  

    Array is returned with the following dimension order:

        [N_tiles, N_recs, N_z, llc, llc]

    where if either N_z or N_recs =1, then that dimension is collapsed
    and not present in the returned array.

    Parameters
    ----------
    fdir : string
        A string with the directory of the binary file to open
    
    fname : string
        A string with the name of the binary file to open
    
    llc : int, optional, default 90
        the size of the llc grid.  For ECCO v4, we use the llc90 domain 
        so `llc` would be `90`.  
        Default: 90
    
    skip : int, optional, default 0
        the number of 2D slices (or records) to skip.  Records could be vertical levels of a 3D field, or different 2D fields, or both.
    
    nk : int, optional, default 1 [singleton]
        number of 2D slices (or records) to load in the depth dimension.  
        Default: 1 [singleton]
    
    nl : int, optional, default 1 [singleton]
        number of 2D slices (or records) to load in the "record" dimension.  
        Default: 1 [singleton] 
    
    filetype: string, default '>f4'
        the file type, default is big endian (>) 32 bit float (f4)
        alternatively, ('<d') would be little endian (<) 64 bit float (d)
            
    less_output : boolean, optional, default False
        A debugging flag.  False = less debugging output
           
        
    Returns
    -------
    data_tiles : dask Array
        a dask array of dimension nk 13 x nl x nk x llc x llc, one llc x llc array 
        for each of the 13 tiles and nl and nk levels.  

    Note
    ----
        The dask array can be converted to a numpy array via
        data = np.asarray(data_tiles)
        
    """

    full_filename = '%s/%s' % (fdir,fname)

    if not less_output:
        print (full_filename)
    
    # Handle "skipped" records by reading up until that record, and
    # dropping preceding records afterward
    #
    # Note: that xmitgcm looks at recs including a full 3D chunk
    # while "skip" refers to 2D chunks.
    nrecs = nl
    skip_3d = int(skip/nk)
    nrecs += skip_3d

    # Reads data into dask array as numpy memmap 
    # [Nrecs x Nz x Ntiles x llc x llc]
    data_tiles = xmitgcm.utils.read_3d_llc_data(full_filename, nx=llc, nz=nk,
                                                nrecs=nrecs, dtype=filetype)

    # Handle cases of single or multiple records, and skip>0
    # Also, swap so that Ntiles dim is ALWAYS first 
    # for ecco_v4_py convention
    if nl==1:
        # Only want 1 record
        data_tiles = data_tiles[skip_3d,...]
        if nk>1:
            data_tiles = data_tiles.swapaxes(0,1)

    else:
        # Want more than one record
        data_tiles = data_tiles[skip_3d:skip_3d+nl,...]

        if nk>1:
            data_tiles = data_tiles.swapaxes(1,2)

        data_tiles = data_tiles.swapaxes(0,1)

    # return the array
    return data_tiles

#%%
def read_llc_to_compact(fdir, fname, llc=90, skip=0, nk=1, nl=1, 
            filetype = '>f4', less_output = False ):
    """

    Loads an MITgcm binary file in the 'tiled' format of the 
    lat-lon-cap (LLC) grids, then converts to the compact form

    Array is returned with the following dimension order:

        [N_recs, N_z, N_tiles*llc, llc]

    where if either N_z or N_recs =1, then that dimension is collapsed 
    and not present in the returned array.

    Parameters
    ----------
    fdir : string
        A string with the directory of the binary file to open
    
    fname : string
        A string with the name of the binary file to open
    
    llc : int, optional, default 90
        the size of the llc grid.  For ECCO v4, we use the llc90 domain 
        so `llc` would be `90`.  
        Default: 90
    
    skip : int, optional, default 0
        the number of 2D slices (or records) to skip.  Records could be vertical levels of a 3D field, or different 2D fields, or both.
    
    nk : int, optional, default 1 [singleton]
        number of 2D slices (or records) to load in the depth dimension.  
        Default: 1 [singleton]
    
    nl : int, optional, default 1 [singleton]
        number of 2D slices (or records) to load in the "record" dimension.  
        Default: 1 [singleton] 
    
    filetype: string, default '>f4'
        the file type, default is big endian (>) 32 bit float (f4)
        alternatively, ('<d') would be little endian (<) 64 bit float (d)
        
    less_output : boolean, optional, default False
        A debugging flag.  False = less debugging output
        
        
    Returns
    -------
    data_compact : ndarray
        a numpy array of dimension nl x nk x 13*llc x llc 

    """

    data_tiles = read_llc_to_tiles(fdir,fname,llc=llc,nk=nk,nl=nl,skip=skip,
                                   filetype=filetype)

    data_compact = llc_tiles_to_compact(data_tiles,less_output=less_output)
    
    # return the array
    return data_compact




def read_llc_to_faces(fdir, fname, llc=90, skip=0, nk=1, nl=1,
        filetype = '>f4', less_output = False):
    """

    Loads an MITgcm binary file in the 'compact' format of the 
    lat-lon-cap (LLC) grids and converts it to the '5 faces' format
    of the LLC grids.  

    Can load 2D and 3D arrays.

    Array is returned with the following dimension order:
    - [N_faces][N_recs, N_z, N_y, N_x]

    where if either N_z or N_recs =1, then that dimension is collapsed
    and not present in the returned array.

    Parameters
    ----------
    fdir : string
        A string with the directory of the binary file to open
    
    fname : string
        A string with the name of the binary file to open
    
    llc : int, optional, default 90
        the size of the llc grid.  For ECCO v4, we use the llc90 domain (llc=90)
    
    skip : int, optional, default 0
        the number of 2D slices (or records) to skip.  Records could be vertical levels of a 3D field, or different 2D fields, or both.
    
    nk : int, optional, default 1 [singleton]
        number of 2D slices (or records) to load in the depth dimension.  
    
    nl : int, optional, default 1 [singleton]
        number of 2D slices (or records) to load in the "record" dimension.  
    
    filetype : string, default '>f4'
        the file type, default is big endian (>) 32 bit float (f4)
        alternatively, ('<d') would be little endian (<) 64 bit float (d)
        
    less_output : boolean, optional, default False
        A debugging flag.  False = less debugging output
        

    Returns
    -------
    data_faces : dict
        a dictionary containing the five lat-lon-cap faces data_faces[n]
        dimensions of each 2D slice of data_faces
        f1,f2: 3*llc x llc
        f3: llc x llc
        4,f5: llc x 3*llc  

    """
    
    data_tiles = read_llc_to_tiles(fdir,fname,llc=llc,nk=nk,nl=nl,skip=skip,
                                   filetype=filetype)

    data_faces = llc_tiles_to_faces(data_tiles, less_output = less_output)

    return data_faces



