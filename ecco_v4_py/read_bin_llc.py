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
import xmitgcm
import numpy as np
import xarray as xr
import time
import sys
import datetime

from .llc_array_conversion  import llc_compact_to_tiles, \
    llc_compact_to_faces, llc_faces_to_tiles, llc_faces_to_compact, \
    llc_tiles_to_faces, llc_tiles_to_compact

from .read_bin_gen import load_binary_array

from .ecco_utils import make_time_bounds_and_center_times_from_ecco_dataset
from .ecco_utils import add_global_metadata
from .ecco_utils import add_variable_metadata
from .ecco_utils import add_coordinate_metadata
from .ecco_utils import sort_all_attrs, sort_attrs


def load_ecco_vars_from_mds(mds_var_dir,
                            mds_grid_dir=None,
                            mds_files=None,
                            vars_to_load = 'all',
                            tiles_to_load = [0,1,2,3,4,5,6,7,8,9,10,11,12],
                            model_time_steps_to_load = 'all',
                            output_freq_code = '',
                            drop_unused_coords = False,
                            grid_vars_to_coords = True,
                            coordinate_metadata = [],
                            variable_metadata = [],
                            global_metadata = [],
                            cell_bounds = None,
                            mds_datatype = '>f4',
                            llc_method = 'bigchunks',
                            less_output=True,
                            **kwargs):

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

    mds_grid_dir : str, optional, default 'None'
        the directory where the model binary (.data) grid fields
        are stored, default is same directory as mds_var_dir

    mds_files   : str or list or None, optional
        either: a string or list of file names to load,
        or None to load all files
        Note :  the name is everything BEFORE the time step
        the mds_file name for 'var.000000000732.data' is 'var'

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

    drop_unused_coords : boolean, optional, default False
        drop coordinates that have dimensions that do not appear in any
        data variables

    grid_vars_to_coords : boolean, optional, default True
        promote grid variables to coordinates. grid variables are coordinates
        according to xarray's definition but not CF conventions

    coordinate_metadata : list, option, default empty list
        tuples with information that is specific to coordinate fields

    variable_metadata : list, option, default empty list
        tuples with information specific to variable fields

    global_metadata : list, option, default empty list
        tuples with global metadata information

    cell_bounds: a DataSet object, optional, default None
        DataSet object with two DataArrays, XC_bnds, YC_bnds
        If passed, the auxillary XC_bnds and YC_bnds coordinates are
        added

    mds_datatype : string, optional, default '>f4'
        code indicating what type of field to load if the xmitgcm cannot
        determine the format from the .meta file.  '>f4' means big endian
        32 bit float.

    llc_method : string, optional, default 'big_chunks'
        refer to the xmitgcm documentation.

    less_output : logical, optional
        if True (default), omit additional print statements

    **kwargs: optional
        extra inputs passed to xmitgcm.open_mdsdataset

    Returns
    =======

    ecco_dataset : xarray Dataset


    """

    # range object is different between python 2 and 3
    if sys.version_info[0] >= 3 and isinstance(tiles_to_load, range):
        tiles_to_load = list(tiles_to_load)

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
        if not less_output:
            print ('loading all model time steps')
            print('read bin_llc:')
            print(mds_var_dir)
            print(mds_grid_dir)

        ecco_dataset = open_mdsdataset(data_dir = mds_var_dir,
                                       grid_dir = mds_grid_dir,
                                       read_grid = True,
                                       prefix = mds_files,
                                       geometry = 'llc',
                                       iters = 'all',
                                       ref_date = ref_date,
                                       delta_t  = delta_t,
                                       default_dtype = np.dtype(mds_datatype),
                                       grid_vars_to_coords=grid_vars_to_coords,
                                       llc_method = llc_method,
                                       ignore_unknown_vars=False,
                                       **kwargs)

    else:
        if not less_output:
            print ('loading subset of  model time steps')

        if isinstance(model_time_steps_to_load, int):
            model_time_steps_to_load = [model_time_steps_to_load]

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
                                           grid_vars_to_coords = grid_vars_to_coords,
                                           llc_method=llc_method,
                                           ignore_unknown_vars=False,
                                           **kwargs)
        else:
            print('model time steps to load ', model_time_steps_to_load)
            raise TypeError('not a valid model_time_steps_to_load.  must be "all", an "int", or a list of "int"')

    if isinstance(cell_bounds, xr.core.dataset.Dataset):
        if 'XC_bnds' in cell_bounds.data_vars:
            ecco_dataset = ecco_dataset.assign_coords({"XC_bnds": (("tile","j","i","nb"), cell_bounds['XC_bnds'])})
        if 'YC_bnds' in cell_bounds.data_vars:
            ecco_dataset = ecco_dataset.assign_coords({"YC_bnds": (("tile","j","i","nb"), cell_bounds['YC_bnds'])})

    # replace the xmitgcm coordinate name of 'FACE' with 'TILE'
    if 'face' in ecco_dataset.coords.keys():
        ecco_dataset = ecco_dataset.rename({'face': 'tile'})
        ecco_dataset.tile.attrs['long_name'] = 'index of llc grid tile'



    if grid_vars_to_coords == False:
        # A bunch of grid geometry fields that aren't actually coordinates in
        # any sense are loaded by default as coordinates. We'll kick them all
        # down to data variables and then promote a few that are actually
        # coordinates.

        # Promote some variables as CF convention compliant coordinates.
        CF_legal_coords = ['XC','YC','XG','YG','Z','Zp1','Zu','Zl','time', 'iter']
        for legal_coord in CF_legal_coords:
            if legal_coord in list(ecco_dataset.data_vars):
                ecco_dataset = ecco_dataset.set_coords(legal_coord)

    # rename iter to timestep
    if 'iter' in ecco_dataset.coords.keys():
        ecco_dataset = ecco_dataset.rename({'iter': 'timestep'})



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
                ecco_dataset = ecco_dataset.drop_vars(ecco_var)

            else:
                vars_loaded.append(ecco_var)

        if not less_output:
            print ('loaded  : ', vars_loaded)
            print ('ignored : ', vars_ignored)

    else:
        if not less_output:
            print ('loaded all variables  : ', ecco_dataset.keys())

    # keep tiles in the 'tiles_to_load' list.
    if not isinstance(tiles_to_load, list) and not isinstance(tiles_to_load,range):
        tiles_to_load = [tiles_to_load]

    if not less_output:
        print ('subsetting tiles to ', tiles_to_load)

    ecco_dataset = ecco_dataset.sel(tile = tiles_to_load)

    # add time bounds for datasets with time-averaged quantities
    if not less_output:
        print ('creating time bounds .... ')

    if 'AVG' in output_freq_code and \
        'time_bnds' not in ecco_dataset.keys():

        if not less_output:
            print ('avg in output freq code and time bounds not in ecco keys')

        time_bnds_ds, center_times = \
            make_time_bounds_and_center_times_from_ecco_dataset(ecco_dataset,\
                                                                output_freq_code)

        ecco_dataset = xr.merge((ecco_dataset, time_bnds_ds))

        ecco_dataset = ecco_dataset.set_coords('time_bnds')

        if not less_output:
            print ('time bounds -----')
            print (time_bnds_ds)
            print ('center times -----')
            print (center_times)
            print ('ecco dataset time values type', type(ecco_dataset.time.values))
            print ('ecco dataset time_bnds typ   ', type(ecco_dataset.time_bnds))
            print ('ecco dataset time_bnds       ', ecco_dataset.time_bnds)

        if isinstance(ecco_dataset.time.values, np.datetime64):
            if not less_output:
                print ('replacing time.values....')
            ecco_dataset['time'].values = center_times

        elif isinstance(center_times, np.datetime64):
            if not less_output:
                print ('replacing time.values....')
            center_times = np.array(center_times)
            ecco_dataset['time'].values[:] = center_times

        elif isinstance(ecco_dataset.time.values, np.ndarray) and \
              isinstance(center_times, np.ndarray):
            if not less_output:
                print ('replacing time.values....')
            ecco_dataset = ecco_dataset.assign_coords({'time': center_times})
            #ecco_dataset['time'] = center_times


    # Drop mask Ctrl fields
    if 'maskCtrlS' in list(ecco_dataset.data_vars):
        ecco_dataset=ecco_dataset.drop_vars('maskCtrlS')
    if 'maskCtrlW' in list(ecco_dataset.data_vars):
        ecco_dataset=ecco_dataset.drop_vars('maskCtrlW')
    if 'maskCtrlC' in list(ecco_dataset.data_vars):
        ecco_dataset=ecco_dataset.drop_vars('maskCtrlC')

    # determine all of the dimensions used by data variables
    all_var_dims = set([])
    for ecco_var in ecco_dataset.data_vars:
        all_var_dims = set.union(all_var_dims, set(ecco_dataset[ecco_var].dims))

    if not less_output:
        print('all_var_dims ', all_var_dims)


    # update global, coordinate, and variable metadata
    # ... first, drop whatever is in 'standard_name' because those fields
    #     are generally not CF compliant.
    for ecco_var in ecco_dataset.data_vars.keys():
        if 'standard_name' in ecco_dataset[ecco_var].attrs.keys():
            ecco_dataset[ecco_var].attrs.pop('standard_name')
    for ecco_var in ecco_dataset.coords.keys():
        if 'standard_name' in ecco_dataset[ecco_var].attrs.keys():
            ecco_dataset[ecco_var].attrs.pop('standard_name')

    # update coordinate metadata
    if len(coordinate_metadata) > 0:
        ecco_dataset = add_coordinate_metadata(coordinate_metadata, ecco_dataset)

    # update variable metadata
    if len(variable_metadata) > 0:
        ecco_dataset, gcmd_keywords = \
            add_variable_metadata(variable_metadata, ecco_dataset)

    # update global metadata
    # possible keys
    keys_2D = set(['i','j','i_g','j_g'])
    keys_3D = set(['k','k_l','k_u','k_p1'])

    if len(all_var_dims.intersection(keys_3D)) > 0:
        dataset_dim = '3D'
    elif len(all_var_dims.intersection(keys_2D)) > 0:
        dataset_dim = '2D'
    else:
        #print('no 2D or 3D dimensions')
        print('cannot find 2D or 3D dims in dataset')
        sys.exit()

    # drop 3D and dims coordinates that do not appear in any data variable
    if drop_unused_coords:

        if dataset_dim == '2D':
            if not less_output:
                print('\n only 2D variables, dropping 3D dims and coords')
                print(all_var_dims)

            # drop 3D dimensions
            dims_to_drop = set(ecco_dataset.dims).intersection(set(['k','k_u','k_l','k_p1']))
            if not less_output:
                print('\n dropping 3D dims and coords')
                print(dims_to_drop)

            for dim in dims_to_drop:
                if not less_output:
                    print('--> dropping', dim)
                ecco_dataset = ecco_dataset.drop(dim)

            # drop 3D coords
            coords_to_drop = set(ecco_dataset.coords).intersection(set(['Z','Zp1','Zu','Zl']))
            for coord in coords_to_drop:
                if not less_output:
                    print('--> dropping ', coord)
                ecco_dataset = ecco_dataset.drop(coord)

    # apply global metadata
    if len(global_metadata) > 0:

        #print('dataset dim: ', dataset_dim)
        # some metadata is 3D dataset specific.
        ecco_dataset = \
            add_global_metadata(global_metadata, ecco_dataset, dataset_dim,\
                                less_output=less_output)

    if not less_output:
        print('dataset dim: ', dataset_dim)
        print('dataset_coords : ', ecco_dataset.coords)

    # add current time and date
    current_time = datetime.datetime.now().isoformat()[0:19]
    ecco_dataset.attrs['date_created'] = current_time
    ecco_dataset.attrs['date_modified'] = current_time
    ecco_dataset.attrs['date_metadata_modified'] = current_time
    ecco_dataset.attrs['date_issued'] = current_time

    if not less_output:
        print('output_freq_code: ', output_freq_code)
    # set the long name of the time attribute
    if 'AVG' in output_freq_code:
        ecco_dataset.time.attrs['long_name'] = 'center time of averaging period'
    elif 'SNAP' in output_freq_code:
        ecco_dataset.time.attrs['long_name'] = 'snapshot time'

    # set averaging period duration and resolution
    print('\n... setting time coverage resolution')
    # --- AVG DAY
    if output_freq_code == 'AVG_MON':
        ecco_dataset.attrs['time_coverage_duration'] = 'P1M'
        ecco_dataset.attrs['time_coverage_resolution'] = 'P1M'
    # --- AVG DAY
    elif output_freq_code == 'AVG_DAY':
        ecco_dataset.attrs['time_coverage_duration'] = 'P1D'
        ecco_dataset.attrs['time_coverage_resolution'] = 'P1D'
    # --- SNAPSHOT
    elif output_freq_code == 'SNAPSHOT':
        ecco_dataset.attrs['time_coverage_duration'] = 'P0S'
        ecco_dataset.attrs['time_coverage_resolution'] = 'P0S'

    ecco_dataset.attrs['original_mds_var_dir'] = str(mds_var_dir)
    ecco_dataset.attrs['original_mds_grid_dir'] = str(mds_grid_dir)

    # alaphbetically sort global attributes
    ecco_dataset.attrs = sort_attrs(ecco_dataset.attrs)


    if not less_output:
        for dim in ecco_dataset.dims:
            print(dim, ecco_dataset[dim].attrs)

    return ecco_dataset

#%%
def read_llc_to_compact(fdir, fname, llc=90, skip=0, nk=1, nl=1,
            filetype = '>f4', less_output = False ):
    """

    Loads an MITgcm binary file in the compact format of the
    lat-lon-cap (LLC) grids.  Note, does not use Dask.

    Array is returned with the following dimension order:

        [nl, nk, N_tiles*llc, llc]

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
    data_compact = load_binary_array(fdir, fname, llc, 13*llc, nk=nk, nl=nl,
                                     skip=skip, filetype = filetype,
                                     less_output = less_output)
    #data_tiles = read_bin_to_tiles(fdir,fname,llc=llc,nk=nk,nl=nl,skip=skip,
    #                skip=skip, filetype = filetype, less_output = less_output)

    #data_tiles = read_llc_to_tiles(fdir,fname,llc=llc,nk=nk,nl=nl,skip=skip,
    #                               filetype=filetype)

    #data_compact = llc_tiles_to_compact(data_tiles,less_output=less_output)

    # return the array
    return data_compact


#%%
def read_llc_to_tiles(fdir, fname, llc=90, skip=0, nk=1, nl=1,
              	      filetype = '>f', less_output = False,
                      use_xmitgcm=False):
    """


    Loads an MITgcm binary file in the 'tiled' format of the
    lat-lon-cap (LLC) grids with dimension order:

        [N_recs, N_z, N_tiles, llc, llc]

    where if either N_z or N_recs =1, then that dimension is collapsed
    and not present in the returned array.

    if use_xmitgcm == True

        data are read in via the low level routine
        xmitgcm.utils.read_3d_llc_data and returned as dask array.

        Hint: use data_tiles.compute() to load into memory.

    if use_xmitgcm == False

        Loads an MITgcm binary file in the 'compact' format of the
        lat-lon-cap (LLC) grids and converts it to the '13 tiles' format
        of the LLC grids.

    Parameters
    ----------
    fdir : string
        A string with the directory of the binary file to open
    fname : string
        A string with the name of the binary file to open
    llc : int
        the size of the llc grid.  For ECCO v4, we use the llc90 domain
        so `llc` would be `90`.
        Default: 90
    skip : int
        the number of 2D slices (or records) to skip.
        Records could be vertical levels of a 3D field, or different 2D fields, or both.
    nk : int
        number of 2D slices (or records) to load in the third dimension.
        if nk = -1, load all 2D slices
        Default: 1 [singleton]
    nl : int
        number of 2D slices (or records) to load in the fourth dimension.
        Default: 1 [singleton]
    filetype: string
        the file type, default is big endian (>) 32 bit float (f)
        alternatively, ('<d') would be little endian (<) 64 bit float (d)
    less_output : boolean
        A debugging flag.  False = less debugging output
        Default: False
    use_xmitgcm : boolean
        option to use the routine xmitgcm.utils.read_3d_llc_data into a dask
        array, i.e. not into memory.
        Otherwise read in as a compact array, convert to faces, then to tiled format
        Default: False

    Returns
    -------
    data_tiles
        a numpy array of dimension 13 x nl x nk x llc x llc, one llc x llc array
        for each of the 13 tiles and nl and nk levels.
    """

    if use_xmitgcm:

        full_filename = '%s/%s' % (fdir,fname)

        if not less_output:
            print('read_llc_to_tiles: full_filename: ',full_filename)

        # Handle "skipped" records by reading up until that record, and
        # dropping preceding records afterward
        #
        # Note: that xmitgcm looks at recs including a full 3D chunk
        # while "skip" refers to 2D chunks.
        nrecs = nl
        skip_3d = int(np.ceil(skip/nk))
        nrecs += skip_3d

        # Reads data into dask array as numpy memmap
        # [Nrecs x Nz x Ntiles x llc x llc]
        data_tiles = xmitgcm.utils.read_3d_llc_data(full_filename, nx=llc, nz=nk,
                                                    nrecs=nrecs, dtype=filetype,
                                                    memmap=False)

        # Handle cases of single or multiple records, and skip>0
        if skip>0:
            if nk>1 and nl>1:
                data_tiles = np.reshape(data_tiles,(nk*nrecs,)+data_tiles.shape[-3:])
                data_tiles = data_tiles[skip:skip+nl*nk]
                data_tiles = np.reshape(data_tiles,(nl,nk)+data_tiles.shape[-3:])
            else:
                data_tiles = data_tiles[skip:skip+nl,...]

    else:

        data_compact = read_llc_to_compact(fdir, fname, llc=llc, skip=skip, nk=nk, nl=nl,
           				    filetype = filetype, less_output=less_output)

        data_tiles   = llc_compact_to_tiles(data_compact, less_output=less_output)

    # return the array
    return data_tiles

#%%
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
        the number of 2D slices (or records) to skip.
        Records could be vertical levels of a 3D field, or different 2D fields, or both.

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

    data_compact = read_llc_to_compact(fdir, fname, llc=llc, skip=skip, nk=nk, nl=nl,
                                    filetype = filetype, less_output=less_output)

    data_faces = llc_compact_to_faces(data_compact, less_output = less_output)

    #data_tiles = read_llc_to_tiles(fdir,fname,llc=llc,nk=nk,nl=nl,skip=skip,
    #                               filetype=filetype)

    #data_faces = llc_tiles_to_faces(data_tiles, less_output = less_output)

    return data_faces
