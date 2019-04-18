"""ECCO v4 Python: read_bin_llc

This module includes utility routines for loading binary files in the 
llc 13-tile native flat binary layout.  This layout is the default for 
MITgcm input and output for global setups using lat-lon-cap (llc) layout. 
The llc layout is used for ECCO v4. 

.. _ecco_v4_py Documentation :
   https://github.com/ECCO-GROUP/ECCOv4-py
"""

from __future__ import division,print_function
import xmitgcm

from .llc_array_conversion  import llc_compact_to_tiles, \
    llc_compact_to_faces, llc_faces_to_tiles, llc_faces_to_compact, \
    llc_tiles_to_faces, llc_tiles_to_compact

#%%
def read_llc_to_tiles(fdir, fname, llc=90, skip=0, nk=1, nl=1, 
                      filetype = '>f'):
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
    llc : int
        the size of the llc grid.  For ECCO v4, we use the llc90 domain 
        so `llc` would be `90`.  
        Default: 90
    skip : int
        the number of 2D slices (or records) to skip.  Records could be vertical levels of a 3D field, or different 2D fields, or both.
    nk : int
        number of 2D slices (or records) to load in the depth dimension.  
        Default: 1 [singleton]
    nl : int
        number of 2D slices (or records) to load in the "record" dimension.  
        Default: 1 [singleton] 
    filetype: string
        the file type, default is big endian (>) 32 bit float (f)
        alternatively, ('<d') would be little endian (<) 64 bit float (d)
        
    Returns
    -------
    data_tiles
        a numpy array of dimension 13 x nl x nk x llc x llc, one llc x llc array 
        for each of the 13 tiles and nl and nk levels.  

    """

    full_filename = '%s/%s' % (fdir,fname)

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
            filetype = '>f', less_output = False ):
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
    llc : int
        the size of the llc grid.  For ECCO v4, we use the llc90 domain 
        so `llc` would be `90`.  
        Default: 90
    skip : int
        the number of 2D slices (or records) to skip.  Records could 
        be vertical levels of a 3D field, or different 2D fields, or both.
        Default: 0
    nk : int
        number of 2D slices (or records) to load in the third dimension.  
        Default: 1 [singleton]
    nl : int
        number of 2D slices (or records) to load in the fourth dimension.  
        Default: 1 [singleton]
    filetype: string
        the file type, default is big endian (>) 32 bit float (f)
        alternatively, ('<d') would be little endian (<) 64 bit float (d)
        Deafult '>f'
    less_output : boolean
        A debugging flag.  False = less debugging output
        Default: False
        
    Returns
    -------
    data_compact
        a numpy array of dimension nl x nk x 13*llc x llc 

    """

    data_tiles = read_llc_to_tiles(fdir,fname,llc=llc,nk=nk,nl=nl,skip=skip,
                                   filetype=filetype)

    data_compact = llc_tiles_to_compact(data_tiles,less_output=less_output)
    
    # return the array
    return data_compact



#%%
def read_llc_to_faces(fdir, fname, llc=90, skip=0, nk=1, nl=1,
        filetype = '>f', less_output = False):
    """

    Loads an MITgcm binary file in the 'compact' format of the 
    lat-lon-cap (LLC) grids and converts it to the '5 faces' format
    of the LLC grids.  

    Can load 2D and 3D arrays.

    Array is returned with the following dimension order:

        [N_faces][N_recs, N_z, N_y, N_x]

    where if either N_z or N_recs =1, then that dimension is collapsed
    and not present in the returned array.

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
        the number of 2D slices (or records) to skip.  Records could be 
        vertical levels of a 3D field, or different 2D fields, or both.
    nk : int
        number of 2D slices (or records) to load in the third dimension.  
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
        
    Returns
    -------
    data_faces : a dictionary containing the five lat-lon-cap faces
        data_faces[n] is a numpy array of face n, n in [1..5]

    - dimensions of each 2D slice of data_faces
        f1,f2: 3*llc x llc
           f3: llc x llc
        f4,f5: llc x 3*llc  
        
    """
    
    data_tiles = read_llc_to_tiles(fdir,fname,llc=llc,nk=nk,nl=nl,skip=skip,
                                   filetype=filetype)

    data_faces = llc_tiles_to_faces(data_tiles, less_output = less_output)

    return data_faces



